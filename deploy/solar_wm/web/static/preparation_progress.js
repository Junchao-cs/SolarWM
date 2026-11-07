/* Display-only compiler telemetry. No model commands and no readiness overrides. */
(()=>{
  let sample=null, received=0, ageAtReceive=0, unavailable=false, unsupportedDeployment=null, endpoint='/preparation-progress';
  const minutes=seconds=>Math.max(1,Math.ceil(seconds/60));
  const detail=document.createElement('small');detail.id='prepareDetail';
  const health=document.createElement('small');health.id='prepareHealth';
  document.querySelector('.prepare-copy').append(detail,health);
  async function poll(){
    try{
      if(typeof lastStatus==='undefined'||!lastStatus?.scene_preparation?.busy||lastStatus.vae_options_ready)return;
      let response=await fetch(endpoint,{cache:'no-store',signal:AbortSignal.timeout(5000)});
      // Older resident servers can expose a read-only deployment-bound bridge.
      if(response.status===404&&endpoint==='/preparation-progress'){
        endpoint='/preparation_progress_data.js';
        response=await fetch(endpoint,{cache:'no-store',signal:AbortSignal.timeout(5000)});
      }
      if(!response.ok)throw Error('Progress unavailable');
      const row=await response.json();
      if(row.deployment_id!==lastStatus.deployment_id)throw Error('Stale deployment');
      if(row.supported===false){unsupportedDeployment=row.deployment_id;sample=null;detail.textContent='';health.textContent='';detail.hidden=health.hidden=true;return;}
      unsupportedDeployment=null;
      const serverDate=Date.parse(response.headers.get('Date'));
      if(!Number.isFinite(serverDate)||!Number.isFinite(row.observed_at))throw Error('Missing telemetry time');
      ageAtReceive=Math.max(0,serverDate/1000-row.observed_at);
      sample=row;received=performance.now();unavailable=false;
    }catch(_){unavailable=true;}
    finally{setTimeout(poll,4000);}
  }
  window.solarwmPreparationProgress=(status)=>{
    // This cumulative compiler log describes cold startup, not the currently
    // selected image. It cannot estimate a later resident scene switch.
    const busy=!!status.scene_preparation?.busy&&!status.vae_options_ready&&unsupportedDeployment!==status.deployment_id;
    detail.hidden=health.hidden=!busy;
    if(!busy)return null;
    const age=ageAtReceive+(performance.now()-received)/1000;
    const valid=sample&&sample.deployment_id===status.deployment_id&&!unavailable&&age<20&&!sample.error;
    if(!valid){
      detail.textContent='Detailed compiler progress is unavailable; this does not confirm a model failure.';
      health.textContent='Waiting for a fresh progress report. Do not restart solely because the counter is unchanged.';
      return null;
    }
    const row=sample;
    if(!row.caught_up){detail.textContent='Reading compiler progress…';health.textContent='Progress endpoint connected';return null;}
    const recent=row.last_event_age_s==null?'No compiler event yet':
      'Last compiler event '+minutes(row.last_event_age_s+age)+' min ago';
    health.textContent='Progress endpoint connected · '+recent+' (not a worker-health check)';
    if(row.completed>=12){
      detail.textContent='DiT compilation complete on all '+row.ranks+' ranks; Graph/VAE warmup and checks may still be running.';
      return null;
    }
    if(row.current_phase==null&&row.completed===0){
      detail.textContent='DiT compilation has not started yet; preparing model or scene inputs.';
      return null;
    }
    detail.textContent=row.current_phase?
      'Phase '+row.current_phase+'/12 · '+row.operation+' · '+minutes(row.phase_elapsed_s+age)+' min in this phase':
      'Waiting for the first compiler phase';
    let estimate='Estimating from completed phases; first startup can take hours.';
    if(row.overdue)estimate='This phase is taking longer than recent phases. Completion time is uncertain; inspect logs if it persists.';
    else if(row.remaining_low_s!=null&&row.remaining_high_s>age){
      const low=minutes(Math.max(0,row.remaining_low_s-age)),high=Math.max(low,minutes(row.remaining_high_s-age));
      estimate='Estimated DiT compilation remaining: '+low+'–'+high+' min. Graph/VAE warmup is additional; this is not a deadline.';
    }
    return {title:'Preparing DiT · '+row.completed+'/12 phases complete on all '+row.ranks+' ranks',estimate};
  };
  poll();
})();
