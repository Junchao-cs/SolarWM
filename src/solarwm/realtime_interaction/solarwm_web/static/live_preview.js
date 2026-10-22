// Narrow demuxer for this server's single-track, no-B-frame fMP4 stream.
// Encoded H.264 bytes are unchanged. Every decoded frame gets its own RAF draw.
function h3Boxes(bytes){
  const out=[];let p=0;const v=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
  while(p<bytes.length){
    if(p+8>bytes.length)throw Error('Truncated MP4 header');
    const n=v.getUint32(p),type=String.fromCharCode(...bytes.subarray(p+4,p+8));
    if(n<8||p+n>bytes.length)throw Error('Invalid MP4 extent');
    out.push({type,body:bytes.subarray(p+8,p+n)});p+=n;
  }return out;
}
function h3Box(bytes,type){const x=h3Boxes(bytes).filter(x=>x.type===type);if(x.length!==1)throw Error('Expected one '+type);return x[0].body}
function h3U32(b,p){return new DataView(b.buffer,b.byteOffset,b.byteLength).getUint32(p)}
class H3CanvasLive {
  constructor(){
    const old=document.getElementById('live');this.canvas=document.createElement('canvas');
    this.canvas.id='liveCanvas';this.canvas.hidden=true;
    this.canvas.style.cssText='width:100%;height:auto;display:block;object-fit:contain;';
    const style=document.createElement('style');style.textContent='#liveCanvas[hidden]{display:none!important}';document.head.append(style);
    old.after(this.canvas);old.hidden=true;this.ctx=this.canvas.getContext('2d',{alpha:false,desynchronized:true});
    this.callbacks=[];this.epoch=0;this.decoderEpoch=0;this.load();
    this.tick=this.tick.bind(this);requestAnimationFrame(this.tick);
  }
  get hidden(){return this.canvas.hidden}set hidden(v){this.canvas.hidden=v}
  get videoWidth(){return this.canvas.width}get videoHeight(){return this.canvas.height}
  get buffered(){const n=this.received;return {length:n?1:0,start:()=>0,end:()=>n/16}}
  get currentTime(){return Math.max(0,this.drawn-1)/16}
  set currentTime(v){throw Error('Frame-skipping seek is prohibited')}
  getVideoPlaybackQuality(){return {droppedVideoFrames:0,totalVideoFrames:this.drawn,measurement:'ordered_canvas_draws_not_HTMLVideoElement'}}
  removeAttribute(){}
  load(){
    this.epoch++;this.closeDecoder();
    this.queue=[];this.pending=new Map();this.received=0;this.drawn=0;this.decoded=0;
    this.nextAt=0;this.paused=true;this.playbackRate=1;this.done=false;
    this.config=null;this.fault=null;this.terminal=null;this.history=[];this.historyBytes=0;
    this.recoveryTimes=[];this.recovered=0;this.prerequisiteDecodes=0;
    this.events=[];this.lastOutputAt=performance.now();
  }
  closeDecoder(){
    this.decoderEpoch++;
    if(this.decoder&&this.decoder.state!=='closed')this.decoder.close();
    for(const x of this.queue||[])x.frame.close();
    this.queue=[];this.pending=new Map();this.decoder=null;
  }
  pause(){this.paused=true}play(){this.paused=false;return Promise.resolve()}
  requestVideoFrameCallback(cb){this.callbacks.push(cb)}
  note(kind,extra={}){
    this.events.push({kind,at:performance.now(),received:this.received,drawn:this.drawn,...extra});
    if(this.events.length>128)this.events.shift();
  }
  check(epoch,signal){
    if(epoch!==this.epoch||signal?.aborted)throw new DOMException('Session replaced or media cancelled','AbortError');
    if(this.terminal)throw this.terminal;
  }
  fail(e){
    if(!this.fault){this.fault=e;this.note('decoder_fault',{name:e.name||'Error'});}
    // Do not pause the renderer: it must drain restored frames during recovery.
    // The single transport consumer owns recovery, including idle/finish paths.
  }
  async configure(moov,signal){
    const epoch=this.epoch;
    if(this.config)throw Error('Unexpected second decoder configuration');
    const trak=h3Box(moov,'trak'),mdia=h3Box(trak,'mdia'),mdhd=h3Box(mdia,'mdhd');
    this.timescale=h3U32(mdhd,mdhd[0]===1?20:12);
    const stsd=h3Box(h3Box(h3Box(mdia,'minf'),'stbl'),'stsd');
    if(h3U32(stsd,4)!==1)throw Error('Multiple sample entries');
    const avc=h3Box(stsd.subarray(8),'avc1'),desc=h3Box(avc.subarray(78),'avcC');
    const v=new DataView(avc.buffer,avc.byteOffset,avc.length),width=v.getUint16(24),height=v.getUint16(26);
    const codec='avc1.'+Array.from(desc.subarray(1,4),x=>x.toString(16).padStart(2,'0')).join('');
    const config={codec,description:desc.slice(),codedWidth:width,codedHeight:height,optimizeForLatency:true};
    if(!(await VideoDecoder.isConfigSupported(config)).supported)throw Error('H.264 WebCodecs unsupported');
    this.check(epoch,signal);this.config=config;this.canvas.width=width;this.canvas.height=height;
    this.openDecoder();
  }
  openDecoder(){
    const epoch=++this.decoderEpoch;
    this.decoder=new VideoDecoder({output:frame=>{
      if(epoch!==this.decoderEpoch){frame.close();return}
      const item=this.pending.get(frame.timestamp);this.pending.delete(frame.timestamp);
      if(!item||item.index!==this.decoded){frame.close();this.fail(Error('Decoded frame order differs'));return}
      this.decoded++;this.lastOutputAt=performance.now();
      if(item.index<this.drawn){this.prerequisiteDecodes++;frame.close();return}
      item.decoded_ms=performance.now();item.frame=frame;this.queue.push(item);
    },error:e=>{if(epoch===this.decoderEpoch)this.fail(e)}});
    this.decoder.configure(this.config);
  }
  stalled(){
    if(document.hidden||this.hidden||this.queue.length||!this.pending.size)return false;
    const oldest=this.pending.values().next().value.submitted_ms;
    return performance.now()-Math.max(oldest,this.lastOutputAt)>2500;
  }
  pruneHistory(){
    // Keep the latest key at/before the first unseen frame, never a later key.
    let keep=0;
    for(let i=1;i<this.history.length;i++)if(this.history[i].type==='key'&&this.history[i].index<=this.drawn)keep=i;
    if(keep){for(const x of this.history.splice(0,keep))this.historyBytes-=x.data.byteLength;}
  }
  remember(sample){
    this.pruneHistory();
    if(!this.history.length&&sample.type!=='key')throw Error('Missing recovery keyframe');
    if(this.history.length>=1024||this.historyBytes+sample.data.byteLength>64*1024*1024)
      throw Error('Recovery history capacity exceeded; no frames discarded');
    // Own just this sample; do not retain a huge parent MP4/network buffer.
    sample.data=new Uint8Array(sample.data);this.history.push(sample);this.historyBytes+=sample.data.byteLength;
  }
  submit(sample){
    this.pending.set(sample.timestamp,{index:sample.index,submitted_ms:performance.now(),row:sample.row});
    try{this.decoder.decode(new EncodedVideoChunk(sample));}catch(e){this.fail(e)}
  }
  async ensureHealthy(signal){
    const epoch=this.epoch;this.check(epoch,signal);
    if(this.stalled())this.fail(Error('Decoder output stalled'));
    if(!this.fault)return;
    while(this.fault){
      this.check(epoch,signal);
      const now=performance.now();this.recoveryTimes=this.recoveryTimes.filter(t=>now-t<30000);
      if(!this.config||!this.history.length||this.recoveryTimes.length>=3){
        this.terminal=Error('Playback recovery stopped; saved frames retained. No replay from frame zero.');
        this.terminal.cause=this.fault;this.note('decoder_terminal');throw this.terminal;
      }
      this.recoveryTimes.push(now);this.pruneHistory();
      const samples=this.history.slice(),first=samples[0];
      if(first.type!=='key'||first.index>this.drawn)throw Error('Unseen frame missing from recovery history');
      this.note('decoder_recover',{key:first.index,first_unseen:this.drawn});
      this.closeDecoder();this.decoded=first.index;this.fault=null;this.lastOutputAt=now;
      try{this.openDecoder();}catch(e){this.fail(e);continue}
      for(const sample of samples){
        while(this.queue.length+this.pending.size>=3&&!this.fault){
          this.check(epoch,signal);if(this.stalled()){this.fail(Error('Recovery decode stalled'));break}
          await new Promise(r=>setTimeout(r,8));
        }
        this.check(epoch,signal);if(this.fault)break;this.submit(sample);
      }
      if(!this.fault){this.recovered++;this.note('decoder_resumed');}
    }
  }
  async feed(bytes,row,signal){
    const epoch=this.epoch;this.check(epoch,signal);
    const all=h3Boxes(bytes);
    for(const box of all)if(box.type==='moov')await this.configure(box.body,signal);
    if(!row.count)return;
    if(!this.decoder)throw Error('Media before decoder configuration');
    const traf=h3Box(h3Box(bytes,'moof'),'traf'),tfhd=h3Box(traf,'tfhd'),run=h3Box(traf,'trun'),tfdt=h3Box(traf,'tfdt');
    const f=h3U32(tfhd,0)&0xffffff,r=h3U32(run,0)&0xffffff;
    if(!(f&0x20000)||f&1)throw Error('Unsupported MP4 data base');
    let p=8,du=0,sz=0,fl=0;if(f&2)p+=4;if(f&8){du=h3U32(tfhd,p);p+=4}if(f&16){sz=h3U32(tfhd,p);p+=4}if(f&32)fl=h3U32(tfhd,p);
    let time=tfdt[0]===1?Number(new DataView(tfdt.buffer,tfdt.byteOffset).getBigUint64(4)):h3U32(tfdt,4);
    const n=h3U32(run,4);if(n!==row.count||row.first!==this.received)throw Error('Sample count/order differs');
    p=8;if(r&1)p+=4;let firstFlags=null;if(r&4){firstFlags=h3U32(run,p);p+=4}
    const payload=h3Box(bytes,'mdat');let offset=0;
    for(let i=0;i<n;i++){
      let duration=du,size=sz,flags=i===0&&firstFlags!==null?firstFlags:fl,cto=0;
      if(r&0x100){duration=h3U32(run,p);p+=4}if(r&0x200){size=h3U32(run,p);p+=4}if(r&0x400){flags=h3U32(run,p);p+=4}if(r&0x800){cto=h3U32(run,p);p+=4}
      if(cto||!duration||!size||offset+size>payload.length||Math.abs(duration/this.timescale-1/16)>1e-8)throw Error('Unsupported sample timing/extent');
      do{
        this.check(epoch,signal);await this.ensureHealthy(signal);
        if(this.queue.length+this.pending.size<3)break;
        await new Promise(r=>setTimeout(r,8));
      }while(true);
      const index=this.received,timestamp=Math.round(time*1e6/this.timescale);
      if(Math.abs(timestamp-index*62500)>1)throw Error('Unexpected media timestamp');
      const sample={index,type:flags&0x10000?'delta':'key',timestamp,duration:62500,data:payload.subarray(offset,offset+size),row};
      this.remember(sample);this.received++;this.submit(sample);
      offset+=size;time+=duration;
    }
    if(offset!==payload.length||p!==run.length)throw Error('Unconsumed MP4 sample bytes');
    await this.ensureHealthy(signal);
  }
  async finish(signal){
    const epoch=this.epoch;
    while(true){
      this.check(epoch,signal);await this.ensureHealthy(signal);if(!this.decoder)return;
      let timer,abort;
      try{
        await Promise.race([this.decoder.flush(),new Promise((_,reject)=>{
          timer=setTimeout(()=>reject(Error('Decoder flush stalled')),3000);
          abort=()=>reject(new DOMException('Media cancelled','AbortError'));
          signal?.addEventListener('abort',abort,{once:true});if(signal?.aborted)abort();
        })]);
        this.check(epoch,signal);if(!this.fault&&this.decoded===this.received)return;
        this.fail(Error('Incomplete decoder flush'));
      }catch(e){this.check(epoch,signal);this.fail(e)}
      finally{clearTimeout(timer);signal?.removeEventListener('abort',abort)}
    }
  }
  tick(now){
    requestAnimationFrame(this.tick);
    const previous=h3Delivery.frames.at(-1);
    if(previous?.measurement==='canvas_draw_submission'&&previous.paint_opportunity_ms==null){
      previous.paint_opportunity_ms=performance.now();previous.draw_to_next_raf_ms=previous.paint_opportunity_ms-previous.callback_ms;
    }
    if(this.paused||this.hidden||document.hidden||this.fault||this.terminal||!this.queue.length||now<this.nextAt)return;
    const item=this.queue[0];if(item.index!==this.drawn){this.fail(Error('Draw sequence differs'));return}
    const start=performance.now();
    try{this.ctx.drawImage(item.frame,0,0)}catch(e){this.fail(e);return}
    this.queue.shift();item.frame.close();const end=performance.now();
    this.drawn++;this.pruneHistory();this.nextAt=Math.max(this.nextAt+1000/(16*this.playbackRate),now+1);
    const row=item.row;
    h3Delivery.frames.push({frame:item.index,callback_ms:end,presentation_ms:end,expected_display_ms:null,
      measurement:'canvas_draw_submission',decode_ms:item.decoded_ms-item.submitted_ms,
      decode_start_ms:item.submitted_ms,decode_end_ms:item.decoded_ms,render_queue_ms:start-item.decoded_ms,draw_ms:end-start,
      presented:this.drawn,received_ms:row.received_ms,append_start_ms:item.submitted_ms,append_end_ms:item.decoded_ms,
      server_send_ms:row.send_ms,server_fragment_ms:row.fragment_ready_ms,playback_rate:this.playbackRate,hidden:this.hidden});
    if(h3Delivery.frames.length>8192)h3Delivery.frames.shift();
    const callbacks=this.callbacks;this.callbacks=[];
    for(const cb of callbacks)try{cb(end,{mediaTime:item.index/16})}catch(e){this.note('observer_error',{name:e.name||'Error'})}
  }
}

// Server and browser clocks are never assumed synchronized. Preserve an interval.
window.h3Delivery={fragments:[],frames:[],clock:[],map:new Map(),buffers:new WeakMap(),
  lower:-Infinity,upper:Infinity,errors:[],sequence:0};
document.getElementById('replay').preload='none';
const h3FastPacing=new URLSearchParams(location.search).get('pacing')!=='reference';
async function h3DeliveryClock(){
  const d=h3Delivery;
  for(let i=0;i<12;i++){
    const a=performance.now();
    try{
      const r=await fetch('/delivery-clock',{cache:'no-store',signal:AbortSignal.timeout(4000)});
      const s=await r.json(),b=performance.now();
      if(!r.ok)continue;
      const lower=s.sent_ms-b,upper=s.received_ms-a;
      d.clock.push({client_send:a,client_receive:b,...s,lower,upper,rtt:b-a});
      d.lower=Math.max(d.lower,lower);d.upper=Math.min(d.upper,upper);
      if(d.lower>d.upper)d.errors.push('clock interval inconsistent');
    }catch(e){d.errors.push('clock: '+e.message)}
    await new Promise(r=>setTimeout(r,40));
  }
}
function h3DeliveryReader(reader){
  let data=new Uint8Array(0),next=0;
  const d=h3Delivery;
  d.fragments=[];d.frames=[];d.map.clear();d.sequence++;
  return {async read(){
    while(true){
      if(data.length>=8){
        const view=new DataView(data.buffer,data.byteOffset,data.byteLength);
        const n=view.getUint32(0),m=view.getUint32(4);
        if(n>65536||m>32*1024*1024)throw Error('Invalid delivery envelope');
        if(data.length>=8+n+m){
          const row=JSON.parse(new TextDecoder().decode(data.subarray(8,8+n)));
          if(row.serial!==next++ || row.length!==m)throw Error('Delivery sequence differs');
          const bytes=data.slice(8+n,8+n+m);data=data.slice(8+n+m);
          row.received_ms=performance.now();d.fragments.push(row);d.buffers.set(bytes,row);
          for(let f=row.first;f<row.first+row.count;f++)d.map.set(f,row);
          if(d.fragments.length>8192)d.fragments.shift();
          for(const f of d.map.keys())if(f<row.first-8192)d.map.delete(f);else break;
          return {done:false,value:bytes};
        }
      }
      const p=await reader.read();
      if(p.done){if(data.length)throw Error('Incomplete final envelope');return {done:true}}
      const combined=new Uint8Array(data.length+p.value.length);
      combined.set(data);combined.set(p.value,data.length);data=combined;
    }
  }};
}
const h3OriginalAppend=SourceBuffer.prototype.appendBuffer;
SourceBuffer.prototype.appendBuffer=function(bytes){
  const row=h3Delivery.buffers.get(bytes);
  if(row){row.append_start_ms=performance.now();this.addEventListener('updateend',()=>{
    row.append_end_ms=performance.now();
  },{once:true})}
  return h3OriginalAppend.call(this,bytes);
};
function h3DeliveryPresented(now,meta){
  const d=h3Delivery,frame=Math.round(meta.mediaTime*16),row=d.map.get(frame);
  if(row){
    d.frames.push({frame,callback_ms:performance.now(),presentation_ms:meta.presentationTime,
      expected_display_ms:meta.expectedDisplayTime,decode_ms:meta.processingDuration*1000,
      presented:meta.presentedFrames,received_ms:row.received_ms,
      append_start_ms:row.append_start_ms,append_end_ms:row.append_end_ms,
      server_send_ms:row.send_ms,server_fragment_ms:row.fragment_ready_ms,
      playback_rate:document.getElementById('live').playbackRate,hidden:document.getElementById('live').hidden});
    if(d.frames.length>8192)d.frames.shift();
  }
  /* Canvas supplies delivery measurements. */
}
/* Canvas supplies delivery measurements. */
// Trigger manually from the acceptance harness or once after successful load.
// Acceptance explicitly samples this before generation, never competes with media.

// EventSource does not retry every initial HTTP failure. Preserve listeners
// across CLOSED connections, but leave CONNECTING's native retry untouched.
class H3EventStream extends EventTarget {
  constructor(url) {
    super(); this.url=url; this.closed=false; this.types=new Set();
    this.timer=null; this.connectTimer=null; this.source=null; this.connect();
  }
  get readyState(){return this.closed ? 2 : this.source.readyState;}
  forward(type){
    this.source.addEventListener(type,event=>this.dispatchEvent(new MessageEvent(type,{
      data:event.data,lastEventId:event.lastEventId||'',origin:event.origin||''
    })));
  }
  addEventListener(type,listener,options){
    super.addEventListener(type,listener,options);
    if(type!=='error'&&type!=='open'&&!this.types.has(type)){
      this.types.add(type);if(this.source)this.forward(type);
    }
  }
  connect(){
    if(this.closed)return;
    const source=this.source=new EventSource(this.url);
    for(const type of this.types)this.forward(type);
    this.connectTimer=setTimeout(()=>{
      this.connectTimer=null;
      if(!this.closed&&source===this.source&&source.readyState!==1){
        source.close();this.scheduleReconnect(source);
      }
    },8000);
    source.onopen=()=>{clearTimeout(this.connectTimer);this.connectTimer=null;this.dispatchEvent(new Event('open'));this.onopen?.();};
    source.onerror=event=>{
      this.dispatchEvent(new Event('error'));this.onerror?.(event);
      if(source.readyState===2)this.scheduleReconnect(source);
    };
  }
  scheduleReconnect(source){
    if(source!==this.source||this.closed||this.timer!==null)return;
    clearTimeout(this.connectTimer);this.connectTimer=null;source.close();
    this.timer=setTimeout(()=>{this.timer=null;this.connect();},2000);
  }
  close(){this.closed=true;clearTimeout(this.timer);clearTimeout(this.connectTimer);this.timer=this.connectTimer=null;this.source?.close();}
}


"use strict";
// Low-latency playback of every byte in the ordered, binary fMP4 preview.
// Seeking skips stale *display*, not transport or native MP4 frames.
const screen=document.getElementById('screen');
const live=new H3CanvasLive();
const replay=document.getElementById('replay');
const fpsOverlay=document.getElementById('fpsOverlay');
const previewState=document.getElementById('previewState');
const badge=document.getElementById('badge');
const metrics=document.getElementById('metrics');
const start=document.getElementById('start');
const pause=document.getElementById('pause');
const resume=document.getElementById('resume');
const stop=document.getElementById('stop');
const download=document.getElementById('download');
const previewResolution=document.getElementById('previewResolution');
const keys='qweasdijkl',buttons={};
const sourceFps=16;
const mime='video/mp4; codecs="avc1.640020"';
let active=false,lastStatus=null,lastStatusAt=0;
let deploymentId=null,retiredDeployments=new Set(),mediaGeneration=0;
let mediaRetryAt=0,mediaFailures=0;
let paceLeadSamples=[];
let mediaSeq=0,mediaSource=null,sourceBuffer=null,mediaUrl=null,streamAbort=null;
let appendQueue=[],streamDone=false,streamError=null;
let frameMetaLast=null,metaGap=false,videoFrameCallbacks=0,skippedDisplayFrames=0;
let paintedAt=[],arrivalSamples=[],lastBufferedFrames=0;
let clockOffsetMs=null,clockRttMs=null,clockSyncing=false;
let chunkReady=new Map(),chunkDisplayed=new Set(),lastChunkLatencyMs=null;
let previewReady=new Map(),lastPreviewToScreenMs=null;
let measuredChunks=0,unseenChunkEnds=0;
async function syncClock(){
  if(clockSyncing)return;
  clockSyncing=true;
  try{
    for(let i=0;i<5;i++){
      const before=performance.now();
      const response=await fetch('/clock',{cache:'no-store',signal:AbortSignal.timeout(3000)});
      const row=await response.json();
      const after=performance.now(),rtt=after-before;
      if(!response.ok||!Number.isFinite(row.server_monotonic_ms))continue;
      if(clockRttMs===null||rtt<clockRttMs){
        clockRttMs=rtt;
        clockOffsetMs=row.server_monotonic_ms-(before+after)/2;
      }
    }
  }catch(error){/* Retain the last valid clock estimate. */}
  finally{clockSyncing=false}
}

let generatedChunks=[],generationSequence=null,generationPaused=false;
function noteGeneratedChunk(s,now){
  if(s.stream_sequence!==generationSequence || Boolean(s.paused)!==generationPaused){
    generatedChunks=[];generationSequence=s.stream_sequence;generationPaused=Boolean(s.paused);
  }
}
function noteServerChunk(row,now){
  if(row.sequence!==generationSequence || generationPaused || !Number.isFinite(row.ready_ms))return;
  const frames=row.chunk*17;
  const last=generatedChunks[generatedChunks.length-1];
  if(last&&(row.ready_ms<=last.time || frames<=last.frames))return;
  generatedChunks.push({time:row.ready_ms,frames,received:now});
  if(generatedChunks.length>9)generatedChunks.shift();
}
function chunkGeneratedFps(now){
  if(!lastStatus?.active || lastStatus?.paused || generationPaused || generatedChunks.length<3)return 0;
  const first=generatedChunks[0],last=generatedChunks[generatedChunks.length-1];
  if(now-last.received>2500)return 0;
  return (last.frames-first.frames)*1000/(last.time-first.time);
}

function clamp(value,low,high){return Math.max(low,Math.min(high,value))}
function bufferedEnd(){
  if(!live.buffered.length)return 0;
  return live.buffered.end(live.buffered.length-1);
}
function arrivalFps(now){
  while(arrivalSamples.length && arrivalSamples[0].time<now-8000)arrivalSamples.shift();
  if(arrivalSamples.length<2)return 0;
  const first=arrivalSamples[0],last=arrivalSamples[arrivalSamples.length-1];
  return (last.frames-first.frames)*1000/(last.time-first.time);
}
function shownFps(now){
  while(paintedAt.length && paintedAt[0]<now-4000)paintedAt.shift();
  return paintedAt.length>1 ? (paintedAt.length-1)*1000/(paintedAt[paintedAt.length-1]-paintedAt[0]) : 0;
}
function recordPresented(now,metadata){
  if(!live.hidden){
    videoFrameCallbacks++;paintedAt.push(now);
    const frame=Number.isFinite(metadata?.mediaTime)
      ? Math.round(metadata.mediaTime*sourceFps) : -1;
    if(frame>=0 && clockOffsetMs!==null){
      for(const [chunk,readyMs] of chunkReady){
        if(chunkDisplayed.has(chunk))continue;
        const lastFrame=chunk*17-1;
        if(frame<lastFrame-1)continue;
        chunkDisplayed.add(chunk);
        if(Math.abs(frame-lastFrame)>1){unseenChunkEnds++;continue}
        const latency=now-(readyMs-clockOffsetMs);
        if(latency>=-clockRttMs/2 && latency<30000){
          lastChunkLatencyMs=latency;measuredChunks++;
        }
        if(previewReady.has(chunk)){
          const previewLatency=now-(previewReady.get(chunk)-clockOffsetMs);
          if(previewLatency>=-clockRttMs/2 && previewLatency<30000)
            lastPreviewToScreenMs=previewLatency;
        }
      }
      for(const chunk of chunkDisplayed){
        if(chunk*17<frame-170){
          chunkDisplayed.delete(chunk);chunkReady.delete(chunk);previewReady.delete(chunk);
        }
      }
    }
  }
  if('requestVideoFrameCallback' in live)live.requestVideoFrameCallback(recordPresented);
}
if('requestVideoFrameCallback' in live)live.requestVideoFrameCallback(recordPresented);
function resetLive(){
  mediaGeneration++;
  paceLeadSamples=[];live._paceAt=0;live.playbackRate=1;
  if(streamAbort)streamAbort.abort();
  streamAbort=null;mediaSeq=0;mediaSource=null;sourceBuffer=null;
  appendQueue=[];streamDone=false;streamError=null;
  frameMetaLast=null;metaGap=false;videoFrameCallbacks=0;skippedDisplayFrames=0;
  paintedAt=[];arrivalSamples=[];lastBufferedFrames=0;generatedChunks=[];
  chunkReady=new Map();chunkDisplayed=new Set();lastChunkLatencyMs=null;
  previewReady=new Map();lastPreviewToScreenMs=null;
  measuredChunks=0;unseenChunkEnds=0;
  live.pause();live.removeAttribute('src');live.load();
  if(mediaUrl)URL.revokeObjectURL(mediaUrl);
  mediaUrl=null;live.hidden=true;screen.hidden=false;replay.hidden=true;
}
function pumpAppend(){
  if(!sourceBuffer || sourceBuffer.updating)return;
  if(live.buffered.length && live.currentTime>15){
    const cutoff=Math.max(0,live.currentTime-12);
    if(live.buffered.start(0)<cutoff-2){
      try{sourceBuffer.remove(0,cutoff);return}
      catch(error){streamError=error}
    }
  }
  if(appendQueue.length){
    try{sourceBuffer.appendBuffer(appendQueue.shift())}
    catch(error){streamError=error;badge.textContent='实时视频解码错误：'+error.message}
  }else if(streamDone && mediaSource?.readyState==='open'){
    try{mediaSource.endOfStream()}catch(error){streamError=error}
    if(lastStatus)status(lastStatus);
  }
}
async function openLive(sequence){return h3ResumeOpen(sequence)}
function mediaFailed(error,generation){
  if(generation!==mediaGeneration||error.name==='AbortError')return;
  // Recoverable faults resume inside the session; never replay from frame zero.
  streamError=error;h3Transport.terminal=true;mediaRetryAt=Infinity;
  h3Delivery.errors.push(String(error));h3MediaNote('playback_terminal');
  badge.textContent='Playback needs attention · all source frames remain saved';
}
function renderTelemetry(){
  const now=performance.now(),end=bufferedEnd();
  const received=h3Transport.receivedFrames||0;
  if(received>lastBufferedFrames){
    lastBufferedFrames=received;
    arrivalSamples.push({time:now,frames:received});
  }
  const generated=chunkGeneratedFps(now);
  const receivedRate=arrivalFps(now),displayRate=shownFps(now);
  const outputFrames=lastStatus?.stream_frames||0;
  const lag=mediaSeq ? Math.max(0,outputFrames/sourceFps-live.currentTime) : 0;
  const lagText=mediaSeq ? lag.toFixed(1)+' 秒' : '—';
  const chunkText=lastChunkLatencyMs===null?'—':Math.max(0,lastChunkLatencyMs).toFixed(0)+' ms';
  const transportText=lastPreviewToScreenMs===null?'—':Math.max(0,lastPreviewToScreenMs).toFixed(0)+' ms';
  fpsOverlay.textContent='生成 '+generated.toFixed(1)
    +' · 到达 '+receivedRate.toFixed(1)
    +' · 显示 '+displayRate.toFixed(1)+' FPS · 块到屏 '+chunkText
    +' · 预览到屏 '+transportText;
  previewState.textContent='逐帧 H.264：已生成 '+outputFrames+' 帧，可播到 '+received
    +' 帧，实际显示 '+videoFrameCallbacks+' 帧，预览跳过过时画面约 '
    +skippedDisplayFrames+' 帧'
    +'；播放积压 '+lagText
    +'；块到屏样本 '+measuredChunks+'，未显示块尾 '+unseenChunkEnds
    +(clockRttMs===null?'':'；计时校准 RTT '+clockRttMs.toFixed(0)+' ms，误差可能约 ±'+(clockRttMs/2).toFixed(0)+' ms')
    +(metaGap?' · 元数据序号有缺口':'')
    +(streamError?' · 传输/解码失败':'');
  h3UpdatePacing(now);
  if(lastStatus?.video_ready)status(lastStatus);
}
function status(s,fromEvent=false){
  if(deploymentId && !s.deployment_id)return;
  if(s.deployment_id && s.deployment_id!==deploymentId){
    if(retiredDeployments.has(s.deployment_id))return;
    if(deploymentId)retiredDeployments.add(deploymentId);
    deploymentId=s.deployment_id;resetLive();lastStatus=null;
    mediaRetryAt=0;mediaFailures=0;clockOffsetMs=clockRttMs=null;
    generationSequence=null;syncClock();
  }
  if(lastStatus && s.stream_sequence<lastStatus.stream_sequence)return;
  lastStatus=s;lastStatusAt=performance.now();active=s.active;noteGeneratedChunk(s,lastStatusAt);
  if(Array.isArray(s.chunk_ready_ms)){
    for(let i=0;i<s.chunk_ready_ms.length;i++){
      if(Number.isFinite(s.chunk_ready_ms[i]))
        chunkReady.set((s.chunk_ready_base||1)+i,s.chunk_ready_ms[i]);
    }
  }
  if(Array.isArray(s.preview_ready_ms)){
    for(let i=0;i<s.preview_ready_ms.length;i++){
      if(Number.isFinite(s.preview_ready_ms[i]))
        previewReady.set((s.preview_base_chunk||1)+i,s.preview_ready_ms[i]);
    }
  }
  badge.textContent=s.error||(!s.ready&&s.phase==='warming' ? '正在预热 '+(s.warmup_chunk??0)+'/'+(s.warmup_total??12)+'，尚未生成画面' : s.phase);
  metrics.textContent=(!s.ready&&s.phase==='warming')
    ? '后台编译预热 '+(s.warmup_chunk??0)+'/'+(s.warmup_total??58)+' 块 · 开始按钮待验收后开放'
    : '块 '+s.chunk+' · 已生成帧 '+s.frames
      +' · 最近 DiT '+(s.dit_ms??'—')+' ms · 最近 VAE 解码 '+(s.vae_ms??'—')+' ms'
      +(s.last_preview_ms==null?'':' · 预览 '+s.last_preview_ms+' ms')
      +(s.last_full_encode_ms==null?'':' · 存档 '+s.last_full_encode_ms+' ms');
  start.disabled=!s.ready||s.active||s.start_pending||(!!window.gallerySelection&&!s.gallery_generation);
  previewResolution.disabled=!!s.active||!!s.start_pending;
  if(s.active && (s.preview_resolution==='full'||s.preview_resolution==='half'))
    previewResolution.value=s.preview_resolution;
  pause.disabled=!s.active||s.paused||!!s.stop_requested;
  resume.disabled=!s.active||!s.paused||!!s.stop_requested;
  stop.disabled=(!s.active&&!s.start_pending)||!!s.stop_requested;
  download.hidden=!s.video_ready;
  if(s.active && s.stream_ready && s.stream_sequence>0 && mediaSeq!==s.stream_sequence
      && performance.now()>=mediaRetryAt){
    const promise=openLive(s.stream_sequence),generation=mediaGeneration;
    promise.catch(error=>mediaFailed(error,generation));
  }
  const fullyAppended=streamDone && !appendQueue.length && !sourceBuffer?.updating
    && bufferedEnd()>=Math.max(0,(s.frames||0)/sourceFps-.2)
    && (h3Delivery.frames.at(-1)?.frame===(s.frames||0)-1 && Number.isFinite(h3Delivery.frames.at(-1)?.paint_opportunity_ms));
  const showReplay=!!s.video_ready && (!mediaSeq||fullyAppended);
  fpsOverlay.hidden=showReplay;
  screen.hidden=showReplay || (mediaSeq && bufferedEnd()>.15);
  live.hidden=showReplay || !(mediaSeq && bufferedEnd()>.15);
  replay.hidden=!showReplay;
  if(showReplay){
    live.pause();
    if(!replay.hasAttribute('src'))replay.src='/video';
    previewState.textContent='生成已结束：点击视频中央回放完整结果。';
  }else if(replay.hasAttribute('src')){
    replay.pause();replay.removeAttribute('src');replay.load();
  }
  if(!s.active)for(const key of keys)buttons[key].classList.remove('down');
  window.refreshGalleryState?.();
}
async function post(path,body={}){
  const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data=await response.json();
  if(!response.ok)throw new Error(data.error||('HTTP '+response.status));
  return data;
}
function showError(error){badge.textContent='错误：'+error.message}
start.onclick=()=>{
  if(window.gallerySelection&&!lastStatus?.gallery_generation)return;
  window.galleryPreview=false;
  resetLive();mediaRetryAt=0;mediaFailures=0;
  const body={preview_resolution:previewResolution.value};
  if(lastStatus?.gallery_generation)body.scene_id=window.gallerySelection?.id||window.galleryDefaultSceneId;
  post('/start',body).then(status).catch(showError);
};
pause.onclick=()=>post('/pause').then(status).catch(showError);
resume.onclick=()=>post('/resume').then(status).catch(showError);
stop.onclick=()=>post('/stop').then(status).catch(showError);
let controlLane=Promise.resolve();
const heldSources=new Map();
const keyLabels={q:'下降',w:'前进',e:'上升',a:'左移',s:'后退',d:'右移',i:'抬头',j:'左转',k:'低头',l:'右转'};
const keyCodes={KeyQ:'q',KeyW:'w',KeyE:'e',KeyA:'a',KeyS:'s',KeyD:'d',KeyI:'i',KeyJ:'j',KeyK:'k',KeyL:'l',ArrowUp:'w',ArrowDown:'s',ArrowLeft:'a',ArrowRight:'d',Space:'e',ShiftLeft:'q',ShiftRight:'q'};
function orderedControl(path,body){
  const task=controlLane.then(()=>post(path,body));
  controlLane=task.catch(error=>{showError(error);document.getElementById('speedState').textContent='控制发送失败，请重试'});
  return task;
}
function sendKey(key,isDown){
  if(!active||lastStatus?.stop_requested||!lastStatus?.navigation_controls)return;
  orderedControl('/control',{key,down:isDown}).catch(()=>{});
  buttons[key]?.classList.toggle('down',isDown);
}
function hold(source,key,isDown){
  const was=[...heldSources.values()].includes(key);
  if(isDown)heldSources.set(source,key);else heldSources.delete(source);
  const now=[...heldSources.values()].includes(key);
  if(was!==now)sendKey(key,now);
}
function releaseAll(){
  heldSources.clear();
  for(const button of Object.values(buttons))button.classList.remove('down');
  if(lastStatus?.navigation_controls)orderedControl('/navigation',{idle:true}).catch(()=>{});
}
for(const key of keys){
  const button=document.createElement('button');button.textContent=key.toUpperCase();
  button.className='key';button.dataset.key=key;button.dataset.label=keyLabels[key];
  button.setAttribute('aria-label',key.toUpperCase()+' '+keyLabels[key]);
  button.onpointerdown=event=>{
    event.preventDefault();button.setPointerCapture(event.pointerId);hold('pointer-'+event.pointerId,key,true);
  };
  button.onpointerup=event=>{event.preventDefault();hold('pointer-'+event.pointerId,key,false)};
  button.onpointercancel=event=>hold('pointer-'+event.pointerId,key,false);
  button.onlostpointercapture=event=>hold('pointer-'+event.pointerId,key,false);
  document.getElementById('keys').appendChild(button);buttons[key]=button;
}
document.addEventListener('keydown',event=>{
  const key=keyCodes[event.code];
  if(!key||event.repeat||['INPUT','TEXTAREA','SELECT'].includes(event.target.tagName)||event.target.isContentEditable||(event.code==='Space'&&['BUTTON','SUMMARY'].includes(event.target.tagName)))return;
  event.preventDefault();hold(event.code,key,true);
});
document.addEventListener('keyup',event=>{
  const key=keyCodes[event.code];if(!key)return;
  if(heldSources.has(event.code)){event.preventDefault();hold(event.code,key,false)}
});
window.addEventListener('blur',releaseAll);
document.addEventListener('visibilitychange',()=>{if(document.hidden)releaseAll()});
document.getElementById('allIdle').onclick=releaseAll;
pause.onclick=()=>{releaseAll();orderedControl('/pause',{}).then(status).catch(()=>{})};
stop.onclick=()=>{releaseAll();orderedControl('/stop',{}).then(status).catch(()=>{})};
const moveSpeed=document.getElementById('moveSpeed'),turnSpeed=document.getElementById('turnSpeed');
let speedTimer=null,speedVersion=0,speedPending=false,navigationEpoch=null;
function showSpeed(){
  document.getElementById('moveSpeedValue').textContent=Number(moveSpeed.value).toFixed(1)+'×';
  document.getElementById('turnSpeedValue').textContent=Number(turnSpeed.value).toFixed(1)+'×';
}
function sendSpeeds(){
  if(!lastStatus?.navigation_controls)return;
  const version=++speedVersion;speedPending=true;
  document.getElementById('speedState').textContent='正在应用…';
  orderedControl('/navigation',{move_speed:Number(moveSpeed.value),turn_speed:Number(turnSpeed.value)})
    .then(row=>{if(version===speedVersion){
      if(lastStatus){lastStatus.move_speed=row.move_speed;lastStatus.turn_speed=row.turn_speed}
      speedPending=false;document.getElementById('speedState').textContent='下一个生成块生效';
    }})
    .catch(()=>{if(version===speedVersion)speedPending=false});
}
for(const input of [moveSpeed,turnSpeed])input.oninput=()=>{
  speedPending=true;++speedVersion;showSpeed();clearTimeout(speedTimer);
  speedTimer=setTimeout(()=>{speedTimer=null;sendSpeeds()},180);
};
function navigationUI(){
  const enabled=!!lastStatus?.navigation_controls;
  moveSpeed.disabled=turnSpeed.disabled=!enabled;
  for(const button of Object.values(buttons))button.disabled=!enabled||!active||!!lastStatus?.paused||!!lastStatus?.stop_requested;
  if(!enabled)document.getElementById('speedState').textContent='控制功能正在上线';
  if(enabled&&navigationEpoch!==deploymentId){
    navigationEpoch=deploymentId;moveSpeed.value=lastStatus.move_speed;turnSpeed.value=lastStatus.turn_speed;showSpeed();
    document.getElementById('speedState').textContent='下一个生成块生效';
  }
  if(enabled&&!speedPending&&!speedTimer&&Number.isFinite(lastStatus.move_speed)&&Number.isFinite(lastStatus.turn_speed)){
    moveSpeed.value=lastStatus.move_speed;turnSpeed.value=lastStatus.turn_speed;showSpeed();
  }
  if(!active){heldSources.clear()}
  const t=window.galleryPreview?0:!replay.hidden?replay.currentTime:mediaSeq===lastStatus?.stream_sequence?live.currentTime:0;
  const seconds=Math.max(0,Math.floor(Number(t)||0));
  const mm=String(Math.floor(seconds/60)%60).padStart(2,'0'),ss=String(seconds%60).padStart(2,'0');
  const hh=Math.floor(seconds/3600);
  document.getElementById('roamTimer').textContent='漫游 '+(hh?String(hh).padStart(2,'0')+':':'')+mm+':'+ss;
}
setInterval(navigationUI,200);
const stream=new EventTarget();stream.close=()=>{};
window.addEventListener('beforeunload',()=>stream.close());
stream.addEventListener('status',event=>status(JSON.parse(event.data),true));
stream.addEventListener('frame_meta',event=>{
  const row=JSON.parse(event.data);
  if(!Number.isInteger(row.index)||row.index<0)metaGap=true;
  else if(frameMetaLast!==null && row.index!==frameMetaLast+1)metaGap=true;
  frameMetaLast=row.index;
});
stream.addEventListener('chunk_meta',event=>{
  const row=JSON.parse(event.data);noteServerChunk(row,performance.now());
  if(row.sequence===lastStatus?.stream_sequence && Number.isInteger(row.chunk)
      && row.chunk>=1 && Number.isFinite(row.ready_ms))
    chunkReady.set(row.chunk,row.ready_ms);
});
stream.addEventListener('gap',()=>{metaGap=true});
stream.onerror=()=>{if(!lastStatus||lastStatus.ready)badge.textContent='连接中断，正在重连'};
setInterval(renderTelemetry,50);
setInterval(syncClock,30000);
syncClock();

let statusPolling=false;
async function pollStatus(){
  if(statusPolling)return;
  statusPolling=true;
  try{
    const response=await fetch('/status',{cache:'no-store',signal:AbortSignal.timeout(4000)});
    if(response.ok)status(await response.json());
  }catch(error){}finally{statusPolling=false}
}
setInterval(pollStatus,500);
pollStatus();


// Incremental persistent media, independent bounded receive/decode queues.
// No seek, frame discard, codec/quality change, or decoder reset on reconnect.
window.h3Transport={version:4,retries:[],accepted:[],controls:[],mediaCursor:0,networkCursor:0,
  receivedFrames:0,arrivals:[],encodedBytes:0,encodedPackets:0,maxEncodedBytes:0,leases:0};
let h3StatusAt=-Infinity;
function h3PolledGeneratedFps(now){
  const s=lastStatus;
  return s?.active&&!s.paused&&now-h3StatusAt<=2500&&Number.isFinite(s.generated_fps)?Math.max(0,s.generated_fps):0;
}
async function h3BoundedFetch(url,options={},milliseconds=5000,parent=null){
  const controller=new AbortController(),abort=()=>controller.abort();
  if(parent?.aborted)abort();else parent?.addEventListener('abort',abort,{once:true});
  const timer=setTimeout(abort,milliseconds);
  try{const response=await fetch(url,{...options,signal:controller.signal,cache:'no-store'});
    const bytes=await response.arrayBuffer();return {response,bytes,receivedAt:performance.now()};
  }finally{clearTimeout(timer);parent?.removeEventListener('abort',abort)}
}
class H3PacketReader {
  constructor(reader,abort){this.reader=reader;this.abort=abort;this.tail=new Uint8Array(0);}
  async exact(size){
    const result=new Uint8Array(size);let at=0;
    while(at<size){
      if(!this.tail.length){
        const timer=setTimeout(this.abort,10000);let part;
        try{
          part=await this.reader.read();h3Transport.lastByteAt=performance.now();
        }finally{clearTimeout(timer)}
        if(part.done)throw Error('媒体流中断，续传未完成的分片');
        this.tail=part.value;
      }
      const n=Math.min(size-at,this.tail.length);
      result.set(this.tail.subarray(0,n),at);at+=n;this.tail=this.tail.subarray(n);
    }
    return result;
  }
  async packet(){
    const head=await this.exact(8),view=new DataView(head.buffer);
    const n=view.getUint32(0),m=view.getUint32(4);
    if(!n||n>65536||m>32*1024*1024)throw Object.assign(Error('非法媒体分片长度'),{fatal:true});
    const row=JSON.parse(new TextDecoder().decode(await this.exact(n)));
    return {row,payload:await this.exact(m)};
  }
}
async function h3ResumeOpen(sequence){
  if(mediaSeq===sequence)return;
  if(!window.VideoDecoder)throw Error('请使用支持 WebCodecs 的浏览器');
  // Freeze the identity from this session's model status. Never accept a
  // different deployment merely because its session number happens to match.
  const deployment=lastStatus?.deployment_id,scene=lastStatus?.scene_id;
  if(typeof deployment!=='string'||!/^[A-Za-z0-9_-]{1,80}$/.test(deployment)||
     typeof scene!=='string'||lastStatus.stream_sequence!==sequence)
    throw Error('缺少当前视频会话身份');
  resetLive();mediaSeq=sequence;streamAbort=new AbortController();
  const parent=streamAbort.signal,generation=mediaGeneration;
  const current=()=>generation===mediaGeneration&&!parent.aborted;
  const d=h3Delivery;d.fragments=[];d.frames=[];d.map.clear();d.sequence++;
  Object.assign(h3Transport,{accepted:[],retries:[],mediaCursor:0,networkCursor:0,receivedFrames:0,
    arrivals:[],encodedBytes:0,encodedPackets:0,maxEncodedBytes:0,leases:0});
  const queue=[];let producerDone=false,failure=null,connectionAbort=null;
  const consumerAbort=new AbortController();
  const stopConsumer=()=>consumerAbort.abort();parent.addEventListener('abort',stopConsumer,{once:true});
  let waitingForBytes=false;
  h3Transport.retryTotal=0;h3Transport.terminal=false;h3Transport.lastByteAt=performance.now();
  const watchdog=setInterval(()=>{
    // Only infer a stuck media lane if fresh status says encoded frames exist,
    // no bytes arrive, and OUR receive queue is not exerting backpressure.
    const s=lastStatus,now=performance.now();
    if(current()&&waitingForBytes&&!document.hidden&&now-h3StatusAt<2500&&
       s?.deployment_id===deployment&&s.scene_id===scene&&s.stream_sequence===sequence&&
       s.stream_frames>h3Transport.receivedFrames&&now-h3Transport.lastByteAt>2500){
      h3MediaNote('network_no_progress');connectionAbort?.abort();
    }
  },500);
  const nap=()=>new Promise(r=>setTimeout(r,4));
  const producer=(async()=>{
    let failures=0;
    while(current()&&!failure&&!producerDone){
      const controller=new AbortController();connectionAbort=controller;
      const abort=()=>controller.abort();parent.addEventListener('abort',abort,{once:true});
      // Explicit testable connection handle; aborting resumes at verified cursor.
      h3Transport.connectionAbort=abort;
      let reader;h3Transport.lastByteAt=performance.now();h3MediaNote("connection_open");
      try{
        const timer=setTimeout(abort,10000);let response;
        try{response=await fetch(window.SOLARWM_MEDIA_ORIGIN+'/stream?'+new URLSearchParams({
          scene,session:String(sequence),cursor:String(h3Transport.networkCursor)}),{signal:controller.signal,cache:'no-store'})}
        finally{clearTimeout(timer)}
        if(response.status===409)throw Object.assign(Error('视频会话校验失败'),{fatal:true});
        if(!response.ok)throw Error('视频 HTTP '+response.status);
        if(response.headers.get('X-Deployment')!==deployment)throw Object.assign(Error('视频部署不匹配'),{fatal:true});
        reader=response.body.getReader();const parser=new H3PacketReader(reader,abort);
        while(current()&&!failure){
          // Payload queue bounded separately from the 3-frame decoder queue.
          // A single source fragment may exceed 8 MiB, but is capped at 32 MiB.
          while(current()&&!failure&&(queue.length>=64||h3Transport.encodedBytes>=8*1024*1024))await nap();
          if(!current()||failure)break;
          let packet;waitingForBytes=true;
          try{packet=await parser.packet()}finally{waitingForBytes=false}
          const {row,payload}=packet;
          if(row.session!==sequence||row.scene!==scene||row.deployment_id!==deployment)
            throw Object.assign(Error('视频分片上下文不匹配'),{fatal:true});
          if(!payload.length){
            if(row.cursor!==h3Transport.networkCursor)throw Object.assign(Error('媒体终点游标不匹配'),{fatal:true});
            if(row.kind==='heartbeat')continue;
            if(row.kind==='done'){producerDone=true;break}
            if(row.kind==='lease'){h3Transport.leases++;break}
            throw Object.assign(Error(row.error||'非法媒体通知'),{fatal:true});
          }
          if(row.serial!==h3Transport.networkCursor||row.first!==h3Transport.receivedFrames||row.length!==payload.length)
            throw Object.assign(Error('视频分片不连续'),{fatal:true});
          const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',payload)),x=>x.toString(16).padStart(2,'0')).join('');
          if(hash!==row.sha256)throw Error('分片校验失败，重新接收');
          if(!current())break;
          row.received_ms=performance.now();queue.push({row,payload});
          h3Transport.encodedBytes+=payload.length;h3Transport.encodedPackets=queue.length;
          h3Transport.maxEncodedBytes=Math.max(h3Transport.maxEncodedBytes,h3Transport.encodedBytes);
          h3Transport.networkCursor++;h3Transport.receivedFrames=row.first+row.count;
          h3Transport.arrivals.push({at:row.received_ms,frames:h3Transport.receivedFrames});
          if(h3Transport.arrivals.length>1024)h3Transport.arrivals.shift();
          failures=0;
        }
      }catch(error){
        if(!current()||failure)return;
        if(error.fatal)throw error;
        h3Transport.retryTotal++;h3MediaNote("network_retry");
        h3Transport.retries.push({cursor:h3Transport.networkCursor,at:performance.now(),error:String(error)});
        if(h3Transport.retries.length>256)h3Transport.retries.shift();
        badge.textContent='Reconnecting video · keeping all frames';
        await new Promise(r=>setTimeout(r,Math.min(1000,100*++failures)));
      }finally{
        waitingForBytes=false;abort();
        // Reader cancellation itself must not block reconnect forever.
        if(reader){let timer;try{await Promise.race([reader.cancel().catch(()=>{}),new Promise(r=>timer=setTimeout(r,100))])}finally{clearTimeout(timer)}}
        parent.removeEventListener('abort',abort);
      }
    }
  })().catch(error=>{failure=error;consumerAbort.abort()});
  const consumer=(async()=>{
    while(current()&&!failure){
      await live.ensureHealthy(consumerAbort.signal);
      if(!queue.length){if(producerDone)break;await nap();continue}
      const {row,payload}=queue.shift();h3Transport.encodedBytes-=payload.length;h3Transport.encodedPackets=queue.length;
      d.fragments.push(row);d.buffers.set(payload,row);
      for(let f=row.first;f<row.first+row.count;f++)d.map.set(f,row);
      if(d.fragments.length>8192)d.fragments.shift();
      for(const f of d.map.keys())if(f<row.first-8192)d.map.delete(f);else break;
      await live.feed(payload,row,consumerAbort.signal);
      h3Transport.mediaCursor++;
      h3Transport.accepted.push({serial:row.serial,first:row.first,count:row.count,sha256:row.sha256});
      if(h3Transport.accepted.length>8192)h3Transport.accepted.shift();
      renderTelemetry();
    }
  })().catch(error=>{if(!failure)failure=error;connectionAbort?.abort()});
  try{
    await Promise.all([producer,consumer]);
    if(!current())return;
    if(failure)throw failure;
    await live.finish(consumerAbort.signal);
    streamDone=true;live.done=true;if(lastStatus)status(lastStatus);
  }finally{clearInterval(watchdog);parent.removeEventListener('abort',stopConsumer)}
}

// Lifecycle commands are never queued behind movement requests. Old queued
// movement is invalidated on stop; non-idempotent POSTs are never auto-replayed.
let h3ControlEpoch=0,h3StopInFlight=false;
async function h3Post(path,body={}){
  const priority=path==='/stop';
  const begin=performance.now();
  const url=priority?window.SOLARWM_CONTROL_ORIGIN+'/stop':path;
  try{
    const {response,bytes}=await h3BoundedFetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)},priority?5000:4000);
    const data=JSON.parse(new TextDecoder().decode(bytes));
    if(!response.ok)throw Error(data.error||'HTTP '+response.status);
    h3Transport.controls.push({path,begin,end:performance.now(),ok:true});return data;
  }catch(error){h3Transport.controls.push({path,begin,end:performance.now(),ok:false,error:String(error)});throw error}
  finally{if(h3Transport.controls.length>512)h3Transport.controls.shift()}
}
function h3OrderedControl(path,body){
  const epoch=h3ControlEpoch,queued=performance.now();
  const task=controlLane.then(()=>{
    if(epoch!==h3ControlEpoch||h3StopInFlight||lastStatus?.stop_requested)return;
    if(performance.now()-queued>1500&&!(path==='/navigation'&&body.idle)){
      // A delayed release must still clear held keys; a delayed press must not
      // unexpectedly move the camera after a connection recovers.
      return body.down===false?h3Post('/navigation',{idle:true}):undefined;
    }
    return h3Post(path,body);
  });
  controlLane=task.catch(error=>{showError(error);document.getElementById('speedState').textContent='控制超时，请重试'});
  return task;
}
async function h3PriorityStop(){
  if(h3StopInFlight)return;
  h3ControlEpoch++;h3StopInFlight=true;
  heldSources.clear();for(const button of Object.values(buttons))button.classList.remove('down');
  stop.textContent='正在结束…';
  try{status(await h3Post('/stop',{}))}
  catch(error){showError(error);badge.textContent='停止尚未确认，请重试';}
  finally{h3StopInFlight=false;stop.textContent='结束生成'}
}

// Keep every frame. Catch up to the complete source->display backlog, not only
// the downloaded portion. Three-frame decoder depth is a cap, not a wait timer.
class H3BacklogPacer {
  constructor(){this.epoch=null;this.base=24;this.target=24;this.samples=[];this.last=0;}
  update({now,epoch,generated,produced,received,drawn,queued}){
    if(epoch!==this.epoch){this.epoch=epoch;this.base=24;this.target=24;this.samples=[];this.last=now;}
    if(Number.isFinite(generated)&&generated>0)this.base=Math.min(50,generated);
    const downloaded=Math.max(0,received-drawn),backlog=Math.max(downloaded,produced-drawn);
    this.samples.push({at:now,backlog});this.samples=this.samples.filter(x=>x.at>=now-800);
    const reserve=Math.min(...this.samples.map(x=>x.backlog));
    const extra=Math.min(26,Math.max(0,reserve-3)*1.5);
    const desired=Math.min(50,Math.max(1,this.base+extra));
    const dt=Math.min(.25,Math.max(0,(now-this.last)/1000));this.last=now;
    this.target=Math.max(this.target-dt*30,Math.min(this.target+dt*100,desired));
    return {fps:this.target,base:this.base,backlog,downloaded,upstream:Math.max(0,produced-received),
      reserve,decoder_queue:queued,catching_up:extra>0,estimated_queue_s:backlog/Math.max(1,this.base)};
  }
}
const h3BacklogPacer=new H3BacklogPacer();
function h3UpdatePacing(now){
  if(!mediaSeq||live.received<=0)return;
  const state=h3BacklogPacer.update({now,epoch:mediaGeneration,generated:chunkGeneratedFps(now),
    produced:lastStatus?.stream_sequence===mediaSeq?lastStatus.frames||0:0,
    received:h3Transport.receivedFrames||live.received,drawn:live.drawn,queued:live.queue.length+live.pending.size});
  live.playbackRate=state.fps/sourceFps;h3Transport.pacing=state;
  if(live.paused&&(live.queue.length>=2||streamDone))live.play().catch(showError);
  screen.hidden=true;live.hidden=false;
}
function h3NetworkArrivalFps(now){
  const list=h3Transport.arrivals||[];
  while(list.length>2&&list[0].at<now-8000)list.shift();
  if(list.length<2||now-list.at(-1).at>2500)return 0;
  const dt=list.at(-1).at-list[0].at;
  return dt>0?(list.at(-1).frames-list[0].frames)*1000/dt:0;
}

post=h3Post;orderedControl=h3OrderedControl;stop.onclick=h3PriorityStop;
const h3NativeStatus=status;status=function(s,...args){if(s!==lastStatus)h3StatusAt=performance.now();return h3NativeStatus(s,...args)};chunkGeneratedFps=h3PolledGeneratedFps;
arrivalFps=h3NetworkArrivalFps;

// Bounded browser-local diagnostics. No automatic telemetry uploads.
function h3MediaNote(kind){
  const t=window.h3Transport;if(!t)return;
  t.events=t.events||[];
  t.events.push({kind,at:performance.now(),cursor:t.networkCursor,accepted:t.mediaCursor,
    drawn:live.drawn,decoded:live.decoded,pending:live.pending?.size||0,queued:live.queue?.length||0});
  if(t.events.length>128)t.events.shift();
}
function h3MediaSnapshot(){
  const t=window.h3Transport;
  return {version:1,session:mediaSeq,cursor:t.networkCursor,accepted:t.mediaCursor,
    received:t.receivedFrames,submitted:live.received,decoded:live.decoded,drawn:live.drawn,
    pending:live.pending?.size||0,queued:live.queue?.length||0,encoded_packets:t.encodedPackets,
    encoded_bytes:t.encodedBytes,network_retries:t.retryTotal||0,decoder_recoveries:live.recovered||0,
    decoder_fault:!!live.fault,terminal:!!live.terminal||!!t.terminal,hidden:document.hidden,
    status_age_ms:Math.min(86400000,Math.max(0,Math.round(performance.now()-h3StatusAt))),
    upstream_frames:lastStatus?.stream_sequence===mediaSeq?lastStatus.stream_frames||0:0};
}
