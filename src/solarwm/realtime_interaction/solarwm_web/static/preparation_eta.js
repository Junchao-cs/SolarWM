/* Display-only ETA. Does not prepare scenes, release Start or touch model state. */
((root)=>{
  class PreparationETA {
    constructor({referenceSeconds=50}={}){
      this.referenceSeconds=referenceSeconds;
      this.history=new Map();
      this.current=null;
    }
    observe(status, now){
      const p=status?.scene_preparation;
      if(!p)return {remaining:null,source:'unavailable',overrun:false};
      if(this.current?.id!==p.id){
        const known=this.history.get(p.scene_id);
        this.current={id:p.id,scene:p.scene_id,anchor:now,elapsed:Math.max(0,p.elapsed_s||0),
          estimate:known??this.referenceSeconds,resident:!!status.vae_options_ready,
          sawBusy:!!p.busy,recorded:false};
      }
      const c=this.current;
      if(p.stage==='encoding')c.hadEncoding=true;
      if(p.busy)c.sawBusy=true;
      // A server progress update supplies a new monotonic elapsed-time anchor.
      // Identical polling snapshots must not restart the countdown.
      if(Number.isFinite(p.elapsed_s)&&p.elapsed_s>c.elapsed){
        c.elapsed=p.elapsed_s;c.anchor=now;
      }
      if(p.ready&&!c.recorded&&c.sawBusy&&c.resident&&!c.hadEncoding&&p.elapsed_s>0){
        this.history.delete(p.scene_id);
        this.history.set(p.scene_id,p.elapsed_s);
        if(this.history.size>32)this.history.delete(this.history.keys().next().value);
        c.recorded=true;
      }
      if(!p.busy)return {remaining:0,source:'complete',overrun:false};
      if(Number.isFinite(p.estimated_remaining_s)&&p.estimated_remaining_s>=0){
        return {remaining:p.estimated_remaining_s,source:'server',
          overrun:!!p.estimate_overrun||p.estimated_remaining_s===0};
      }
      // Cold model loading is not comparable to a resident scene switch.
      if(!status.ready||!c.resident||c.hadEncoding)return {remaining:null,source:'startup',overrun:false};
      const elapsed=c.elapsed+Math.max(0,now-c.anchor)/1000;
      const remaining=Math.max(0,c.estimate-elapsed);
      return {remaining,source:this.history.has(p.scene_id)?'observed':'reference',
        overrun:remaining===0};
    }
  }
  if(typeof module==='object'&&module.exports)module.exports={PreparationETA};
  else root.SolarWMPreparationETA=PreparationETA;
})(globalThis);
