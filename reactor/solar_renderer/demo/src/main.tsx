import {useEffect, useMemo, useRef, useState} from "react";
import {createRoot} from "react-dom/client";
import {Reactor, type ReactorStatus} from "@reactor-team/js-sdk";
import "./style.css";
import "./examples.css";
import {ComparisonClock} from "./comparison";
import {Logo} from "@reactor-team/ui";
import {examples,loadExample} from "./examples";

type Status = {world_id:number; chunks:number; frames:number; complete:boolean; paused:boolean; active:boolean; elapsed:number|null};
function useFileUrl(file:File|null) {
  const [url,setUrl] = useState("");
  useEffect(()=>{
    if(!file) {setUrl(""); return;}
    const next=URL.createObjectURL(file); setUrl(next);
    return ()=>URL.revokeObjectURL(next);
  },[file]);
  return url;
}
function Workspace() {
  const [endpoint,setEndpoint] = useState(import.meta.env.VITE_RUNTIME_URL || "http://localhost:8080");
  const [activeEndpoint,setActiveEndpoint] = useState(endpoint);
  const client = useMemo(()=>new Reactor({
    modelName:"solarrenderer", local:true,
    apiUrl:activeEndpoint,
    modelTracks:[{name:"main_video",kind:"video",direction:"recvonly"}],
    // Commands run between model blocks; cold preparation can exceed a minute.
    controlRequestTimeoutMs:180_000,
  }),[activeEndpoint]);
  const actionLock = useRef(false);
  const [displayedTime,setDisplayedTime] = useState<number|null>(null);
  const video = useRef<HTMLVideoElement>(null);
  const sourceVideo = useRef<HTMLVideoElement>(null);
  const clock = useRef(new ComparisonClock());
  const latestStatus = useRef<Status|null>(null);
  const pendingPair = useRef<{structure:File; reference:File|null}|null>(null);
  const pairedWorld = useRef<number|null>(null);
  const sourceTime = useRef(0);
  const [pair,setPair] = useState<{structure:File; reference:File|null}|null>(null);
  const [connection, setConnection] = useState<ReactorStatus>("disconnected");
  const [prompt, setPrompt] = useState("");
  const [seed, setSeed] = useState(0);
  const [structure, setStructure] = useState<File|null>(null);
  const [reference, setReference] = useState<File|null>(null);
  const [status, setStatus] = useState<Status|null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [selectedExample,setSelectedExample] = useState("");
  const [fileRevision,setFileRevision] = useState(0);
  const sourceUrl=useFileUrl(pair?.structure ?? structure);
  const referenceUrl=useFileUrl(pair ? pair.reference : reference);
  function seekSource() {
    const source=sourceVideo.current;
    if(source && source.readyState>=1 && !source.seeking && Math.abs(source.currentTime-sourceTime.current)>0.001)
      source.currentTime=Math.min(sourceTime.current, Math.max(0,source.duration-0.001));
  }
  useEffect(()=>{
    client.on("statusChanged", value=>{
      setConnection(value);
      if(value === "disconnected") {
        pairedWorld.current=null; clock.current.reset(); sourceTime.current=0;
        latestStatus.current=null; setStatus(null); setDisplayedTime(null);
        if(video.current) video.current.srcObject=null;
        seekSource();
      }
    });
    client.on("message", message => {
      if(message.type !== "state_update") return;
      const next=message.data as Status;
      if(next.world_id !== latestStatus.current?.world_id) {
        clock.current.reset(); sourceTime.current=0; setDisplayedTime(null);
        pairedWorld.current=pendingPair.current ? next.world_id : null;
        setPair(pendingPair.current); pendingPair.current=null;
        seekSource();
      }
      latestStatus.current=next; setStatus(next);
    });
    client.on("error", error => setError(String(error)));
    client.on("trackReceived", (name, _track, stream)=>{
      if(name === "main_video" && video.current) video.current.srcObject = stream;
    });
    return ()=>{void client.disconnect();};
  }, [client]);
  useEffect(()=>{
    const output=video.current;
    if(!output || !output.requestVideoFrameCallback) return;
    let callback:number;
    const frame:VideoFrameRequestCallback=(_now,metadata)=>{
      const current=latestStatus.current;
      if(current && current.frames>0 && pairedWorld.current===current.world_id) {
        const time=clock.current.observe(metadata.presentedFrames,metadata.width,metadata.height);
        if(time!==null) {sourceTime.current=time; setDisplayedTime(time); seekSource();}
      }
      callback=output.requestVideoFrameCallback(frame);
    };
    callback=output.requestVideoFrameCallback(frame);
    return ()=>output.cancelVideoFrameCallback(callback);
  },[]);
  async function perform(action:()=>Promise<unknown>) {
    if(actionLock.current) return;
    actionLock.current=true;
    setBusy(true); setError("");
    try {await action();} catch(e) {setError(e instanceof Error ? e.message : String(e));}
    finally {setBusy(false); actionLock.current=false;}
  }
  async function command(name:string, data:Record<string,unknown>) {
    const result = await client.sendCommand(name, data);
    if(result === undefined) throw new Error(`The server refused ${name}. See the command error below.`);
    return result;
  }
  async function start() {
    if(!structure) return;
    await command("set_structure", {video: await client.uploadFile(structure)});
    if(reference) await command("set_reference", {image: await client.uploadFile(reference)});
    else await command("clear_reference", {});
    await command("set_prompt", {prompt});
    await command("set_seed", {seed});
    pendingPair.current={structure,reference};
    try {await command("start", {});} finally {pendingPair.current=null;}
  }
  async function selectExample(id:string) {
    const input=await loadExample(id);
    setStructure(input.structure); setReference(input.reference);
    setPrompt(input.prompt); setSeed(input.seed);
    setSelectedExample(id); setFileRevision(value=>value+1);
  }
  const ready = connection === "ready";
  return <main>
    <header><a className="brand" href="https://reactor.inc" target="_blank" rel="noreferrer"><Logo variant="symbol" width={24} height={24}/><strong>SolarRenderer</strong><span>on Reactor</span></a>
      <div className="connection-bar"><label className="endpoint">Endpoint<input aria-label="Endpoint" type="url" value={endpoint} disabled={busy || connection!=="disconnected"} onChange={e=>{setEndpoint(e.target.value);setActiveEndpoint(e.target.value.trim().replace(/\/$/,""));}}/></label>
      <span className={`connection ${ready ? "live" : ""}`}>{connection}</span>
      <button disabled={busy || !["ready","disconnected"].includes(connection) || activeEndpoint!==endpoint.trim().replace(/\/$/,"")}
        onClick={()=>void perform(async()=>{
          if(ready) {
            await client.disconnect();
            const deadline=Date.now()+30000;
            while(Date.now()<deadline) {
              const response=await fetch(`${activeEndpoint}/health`,{signal:AbortSignal.timeout(3000)});
              if(response.ok && (await response.json()).state==="available") return;
              await new Promise(resolve=>setTimeout(resolve,300));
            }
            throw new Error("Disconnected. The server is still cleaning up; wait before reconnecting.");
          } else await client.connect();
        })}>
        {ready ? "Disconnect" : "Connect"}</button></div></header>
    <div className="workspace"><aside>
      <p className="eyebrow">STRUCTURE IN. VIDEO OUT.</p><h1>Give structure<br/>a new look.</h1>
      <p>Choose a recorded structure video, an optional aligned appearance image, and the six-part rendering prompt.</p>
      <p className="eyebrow">01 / TRY AN EXAMPLE</p>
      <div className="example-grid" role="group" aria-label="Example inputs">{examples.map(example=><button key={example.id} className={`example-card ${selectedExample===example.id ? "selected" : ""}`} aria-label={`Select ${example.title}`} aria-pressed={selectedExample===example.id} disabled={busy || Boolean(status?.active)} onClick={()=>void perform(()=>selectExample(example.id))}>
        <img src={example.thumbnail} alt=""/><strong>{example.title}</strong><small>{example.detail}</small>
      </button>)}</div>
      <p className="example-note">One official scene, two tested input modes. Click to fill the inputs, then press Start. Nothing starts automatically.</p>
      <p className="eyebrow">02 / INPUTS · EDIT OR UPLOAD YOUR OWN</p>
      <label>Structure video<input key={`video-${fileRevision}`} disabled={busy} type="file" accept="video/*" onChange={e=>{setStructure(e.target.files?.[0]??null);setSelectedExample("");}}/></label>
      {structure && <small className="loaded-file" data-testid="structure-name">Loaded: {structure.name}</small>}
      <small>At least 158 frames at 24 FPS. Longer videos use the native opening segment.</small>
      <label>Appearance reference · optional<input key={`image-${fileRevision}`} disabled={busy} type="file" accept="image/png,image/jpeg,image/webp" onChange={e=>{setReference(e.target.files?.[0]??null);setSelectedExample("");}}/></label>
      <small className="loaded-file" data-testid="reference-name">{reference ? `Loaded: ${reference.name}` : "No appearance image selected"}</small>
      {reference && <button disabled={busy} onClick={()=>{setReference(null);setSelectedExample("");setFileRevision(value=>value+1);}}>Clear appearance image</button>}
      <label>Rendering prompt<textarea aria-label="Rendering prompt" disabled={busy} rows={9} maxLength={16000} value={prompt} onChange={e=>{setPrompt(e.target.value);setSelectedExample("");}} placeholder="Select an example above, or paste your six-part rendering prompt…"/></label>
      <label>Seed<input disabled={busy} type="number" min={0} max={4294967295} value={seed} onChange={e=>{setSeed(Number(e.target.value));setSelectedExample("");}}/></label>
      <button className="primary" disabled={!ready || busy || !structure || !prompt.trim()} onClick={()=>void perform(start)}>{busy ? "Preparing…" : "Start new video"}</button>
      <button disabled={!ready || busy || !status?.active} onClick={()=>void perform(()=>command("set_paused",{paused:!status?.paused}))}>{status?.paused ? "Resume" : "Pause"}</button>
      <p>Files wait until Start. Prompt edits apply to the next video; the current render keeps its original prompt.</p>
      <p>Pause and other commands wait for the current block to finish. Preparing each new video can take over a minute.</p>
      <p role="status">{status ? `${status.chunks}/10 blocks · ${status.frames} frames${status.elapsed == null ? "" : ` · ${status.elapsed.toFixed(2)} s/block`}${status.complete ? " · complete" : ""}` : "Connect to begin"}</p>
      {error && <p role="alert">{error}</p>}
    </aside><section className="comparison">
      <div className="comparison-heading"><div><p className="eyebrow">ONE SEQUENCE · TWO VIEWS</p><h2>Structure → Render</h2><p>The source follows displayed output frames, not generation progress.</p></div><output className="playback-clock" aria-label="Displayed video time">{displayedTime===null ? "Waiting for video" : `${displayedTime.toFixed(2)} / ${(157/24).toFixed(2)} s`}</output></div>
      <div className="comparison-grid"><figure><figcaption>Input structure <span>Recorded conditioning video</span></figcaption>
        <div className="stage"><video ref={sourceVideo} src={sourceUrl || undefined} muted playsInline preload="auto" className="video" onLoadedMetadata={seekSource} onSeeked={seekSource} onError={()=>setError("This browser cannot preview the structure video. Try an MP4 with H.264 video.")}/>
        {!sourceUrl && <div className="placeholder">Choose a structure video</div>}</div>
      </figure><figure><figcaption>Live render <span>Reactor output</span></figcaption>
      <div className="stage"><video ref={video} autoPlay playsInline muted className={`video ${displayedTime===null ? "awaiting-output" : ""}`}/>
      {displayedTime===null && <div className="placeholder">{status?.active ? "Preparing your video… Encoding inputs and loading the renderer can take over a minute." : "Connect, choose your inputs, then Start new video"}</div>}
      </div></figure></div>
      <p className="sync-note">Playback-aligned preview · The source waits when output stalls. Buffered output may continue after Pause. Browser-dropped frames can affect alignment; reconnecting requires Start new video to align again.</p>
      {referenceUrl && <div className="appearance"><img src={referenceUrl} alt="Appearance reference for this comparison"/><div><strong>Appearance reference</strong><p>Guides the look; the structure video supplies motion and layout.</p></div></div>}
      {pair && <p>Comparison uses the inputs captured at Start. Newly selected files apply to the next video.</p>}
    </section></div>
    <footer>SolarRenderer by its original authors · Served with <a href="https://reactor.inc">Reactor</a></footer>
  </main>;
}
createRoot(document.getElementById("root")!).render(<Workspace/>);
