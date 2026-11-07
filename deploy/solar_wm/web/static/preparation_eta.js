/* Display-only, end-to-end preparation ETA. Never changes model readiness. */
((root)=>{
  class PreparationETA {
    // Conservative initial references. Successful resident preparations replace
    // them with observed costs; cold compilation is deliberately separate.
    constructor({referenceSeconds=25,encodingSeconds=35}={}){
      this.referenceSeconds=referenceSeconds;
      this.encodingSeconds=encodingSeconds;
      this.history=new Map();this.encodingHistory=[];
      this.current=null;this.pending=null;this.deployment=null;
    }
    resetDeployment(status){
      if(this.deployment!==status.deployment_id){
        this.history.clear();this.encodingHistory=[];
        this.current=null;this.pending=null;this.deployment=status.deployment_id;
      }
    }
    median(values,fallback){
      if(!values.length)return fallback;
      const a=[...values].sort((x,y)=>x-y),i=Math.floor(a.length/2);
      return a.length%2?a[i]:(a[i-1]+a[i])/2;
    }
    forecast(scene){
      const known=this.history.get(scene);
      const warmup=known?.warmup??this.median([...this.history.values()].map(x=>x.warmup).filter(x=>x>0),this.referenceSeconds);
      const encoding=known?0:this.median(this.encodingHistory,this.encodingSeconds);
      return {warmup,encoding,total:warmup+encoding,source:known?.warmup||this.encodingHistory.length?'observed':'reference'};
    }
    begin(scene,status,now){
      this.resetDeployment(status);
      this.pending={scene,started:now,...this.forecast(scene)};
    }
    pendingEstimate(scene,status,now){
      this.resetDeployment(status);
      const p=this.pending;
      if(!p||p.scene!==scene)return {remaining:null,source:'unavailable'};
      const remaining=Math.max(0,p.total-Math.max(0,now-p.started)/1000);
      return {remaining,source:p.source,overrun:remaining===0};
    }
    observe(status,now){
      this.resetDeployment(status);
      const p=status?.scene_preparation;
      if(!p)return {remaining:null,source:'unavailable',overrun:false};
      if(this.current?.id!==p.id){
        const pending=this.pending?.scene===p.scene_id&&p.busy?this.pending:null;
        const f=pending||this.forecast(p.scene_id);
        this.current={id:p.id,scene:p.scene_id,anchor:now,
          elapsed:Math.max(0,p.elapsed_s||0,pending?(now-pending.started)/1000:0),
          deadline:f.total,warmup:f.warmup,encoding:f.encoding,source:f.source,
          resident:!!status.vae_options_ready,
          sawStart:!!pending||p.stage==='queued'||(p.elapsed_s||0)<2,
          sawBusy:!!p.busy,recorded:false};
        if(pending)this.pending=null;
      }
      const c=this.current;
      if(p.busy)c.sawBusy=true;
      if(Number.isFinite(p.elapsed_s)&&p.elapsed_s>c.elapsed){c.elapsed=p.elapsed_s;c.anchor=now;}
      const elapsed=c.elapsed+Math.max(0,now-c.anchor)/1000;
      if(p.stage==='encoding'&&!c.hadEncoding){
        c.hadEncoding=true;
        // Correct an invalidated cache forecast once, not on each poll.
        if(c.encoding===0){
          c.encoding=this.median(this.encodingHistory,this.encodingSeconds);
          c.deadline+=c.encoding;c.revised=true;
        }
      }
      if(p.busy&&!['queued','encoding'].includes(p.stage)&&c.warmupStart==null&&(c.hadEncoding||c.sawStart)){
        c.warmupStart=c.hadEncoding?elapsed:0;
        // One request clock. Replace the extraction allowance with observed
        // cost (or zero for a cache hit), without starting another timer.
        const deadline=c.warmupStart+c.warmup;
        c.revised=deadline>c.deadline+1;c.deadline=deadline;
      }
      if(p.ready&&!c.recorded){
        let warmup=null;
        if(c.sawBusy&&c.resident&&c.warmupStart!=null&&p.elapsed_s>c.warmupStart){
          warmup=p.elapsed_s-c.warmupStart;
          if(c.hadEncoding&&c.warmupStart>0){
            this.encodingHistory.push(c.warmupStart);
            if(this.encodingHistory.length>16)this.encodingHistory.shift();
          }
        }
        // Ready confirms cached inputs, even if this tab missed preparation.
        // Never learn cold compilation or aborted requests as resident costs.
        const previous=this.history.get(p.scene_id);
        this.history.delete(p.scene_id);this.history.set(p.scene_id,{warmup:warmup??previous?.warmup??null});
        if(this.history.size>32)this.history.delete(this.history.keys().next().value);
        c.recorded=true;
      }
      if(!p.busy)return {remaining:0,source:'complete',overrun:false};
      if(Number.isFinite(p.estimated_remaining_s)&&p.estimated_remaining_s>=0){
        return {remaining:p.estimated_remaining_s,source:'server',
          overrun:!!p.estimate_overrun||p.estimated_remaining_s===0};
      }
      // Cold model loading is not comparable to a resident scene switch.
      if(!status.ready||!c.resident)return {remaining:null,source:'startup',overrun:false};
      const remaining=Math.max(0,c.deadline-elapsed);
      return {remaining,source:c.source,overrun:remaining===0,revised:!!c.revised};
    }
  }
  if(typeof module==='object'&&module.exports)module.exports={PreparationETA};
  else root.SolarWMPreparationETA=PreparationETA;
})(globalThis);
