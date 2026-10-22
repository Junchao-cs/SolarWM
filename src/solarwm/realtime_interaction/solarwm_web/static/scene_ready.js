/* Selection-driven preparation. Passive tabs must never change server scenes. */
(()=>{
  const $=id=>document.getElementById(id);
  const selected=()=>window.gallerySelection?.id||window.galleryDefaultSceneId||null;
  const eta=new window.SolarWMPreparationETA();
  let sending=false, sentScene=null, localError=null;
  const previousRefresh=window.refreshGalleryState;
  const labels={queued:'等待后台准备',encoding:'正在准备图片与 Prompt',initializing:'正在初始化场景',
    warming:'正在后台预热',capturing:'正在抓取 Graph',steady:'正在确认稳态',failed:'场景准备失败',
    cancelled:'已取消准备'};
  function render(){
    if(typeof lastStatus==='undefined'||!lastStatus)return;
    const s=lastStatus,p=s.scene_preparation,match=p?.scene_id===selected();
    const estimate=eta.observe(s,performance.now());
    const busy=!!p?.busy, ready=!!(s.ready&&!s.error&&match&&p.ready);
    const show=!s.active&&!s.start_pending&&(!ready||busy);
    $('start').disabled=!ready||s.active||s.start_pending||busy||sending;
    $('start').textContent=busy||sending?'准备中…':'开始生成';
    $('prepareOverlay').hidden=!show;
    $('roamTimer').hidden=show||(!s.active&&!s.video_ready);
    $('prepareOverlay').setAttribute('aria-busy',String(busy||sending||!s.ready));
    const failed=match&&['failed','cancelled'].includes(p.stage);
    const mismatch=!!(s.ready&&!s.active&&!s.start_pending&&!busy&&!sending&&!match);
    const needsPrepare=!!(s.ready&&!s.active&&!s.start_pending&&!busy&&!sending&&!ready);
    if(match&&(p.ready||p.busy))localError=null;
    $('prepareTitle').textContent=s.error?'服务异常':localError?'准备请求失败':!s.ready?'模型启动预热中':busy?(match?(labels[p.stage]||'正在准备场景'):'其他场景正在准备'):failed?labels[p.stage]:mismatch?'所选场景尚未就绪':sending?'正在提交场景':'所选场景尚未就绪';
    const startupLabel='首次启动需要加载模型与缓存，请稍候';
    $('prepareEstimate').textContent=s.error||localError||(failed?(p.error||'可以重新准备'):
      !s.ready?startupLabel:
      busy?(estimate.remaining==null?
        '正在估算剩余时间；首次启动可能需要较长时间':
        estimate.overrun?'已超过参考时间，仍在准备':
        '预计还需约 '+Math.max(1,Math.ceil(estimate.remaining))+' 秒'+
          (estimate.source==='reference'?'（参考估计）':'')):
        mismatch?(sentScene===selected()?'服务的当前场景已切换，请重新准备所选首帧':'点击准备所选首帧'):
        sending?'等待服务器确认…':'点击重新准备');
    $('prepareCancel').hidden=!busy||!!s.error;
    $('prepareCancel').disabled=!!p?.cancel_requested;
    $('prepareCancel').textContent=p?.cancel_requested?'正在取消…':'取消准备';
    $('prepareRetry').hidden=!!s.error||!(failed||localError||needsPrepare);
    $('prepareRetry').textContent=mismatch?'准备所选场景':'重新准备';
    $('prepareRetry').disabled=busy||sending;
    document.querySelector('.prepare-spinner').hidden=!!(s.error||failed||localError||needsPrepare);
    if(show){$('compactFps').hidden=true;$('stateBadge').textContent=s.error?'异常':failed||needsPrepare?'未就绪':'准备中';}
    if(match&&ready)$('sceneNote').textContent='场景已就绪';
  }
  window.refreshGalleryState=()=>{previousRefresh?.();render();};
  window.prepareSelectedScene=async()=>{
    if(typeof lastStatus==='undefined'||!lastStatus?.ready||lastStatus.error||lastStatus.active||lastStatus.start_pending||sending||!selected())return;
    const id=selected(),p=lastStatus.scene_preparation;
    if(p?.scene_id===id&&(p.ready||p.busy))return render();
    if(p?.busy)return render();
    sending=true;sentScene=id;localError=null;render();
    try{const s=await post('/prepare',{scene_id:id});status(s);}
    catch(e){localError=e.message;sentScene=null;}
    finally{sending=false;render();}
  };
  $('start').onclick=async()=>{
    const p=lastStatus?.scene_preparation,id=selected();
    if(sending||lastStatus?.error||lastStatus?.active||lastStatus?.start_pending||!p?.ready||p.scene_id!==id)return;
    sending=true;
    $('start').disabled=true;
    try{
      resetLive();mediaRetryAt=0;mediaFailures=0;
      const s=await post('/start',{scene_id:id,preparation_id:p.id,preview_resolution:$('previewResolution').value,vae_backend:document.getElementById('vaeBackend')?.value||'current'});
      window.galleryPreview=false;status(s);
    }catch(e){localError=e.message;showError(e);}
    finally{sending=false;render();}
  };
  $('prepareCancel').onclick=async()=>{try{status(await post('/stop'));}catch(e){localError=e.message;render();}};
  $('prepareRetry').onclick=()=>{sentScene=null;localError=null;window.prepareSelectedScene();};
  // Polling is read-only: another tab or a default garden page must not undo a
  // user's prepared scene. Explicit gallery clicks/retry still prepare it.
  // A ready scene replaced elsewhere is actionable, not an endless spinner.
  setInterval(render,250);
  render();
})();

/* Visible, native-radio VAE choices. The per-session payload stays unchanged. */
(()=>{
  const panel=document.createElement('div');
  panel.id='vaeSelectorPanel';panel.hidden=true;
  panel.innerHTML=`
    <div class="vae-heading"><span id="vaeBackendTitle">VAE 解码器</span><small id="vaeBackendNote"></small></div>
    <input id="vaeBackend" type="hidden" value="current">
    <div class="vae-options" role="radiogroup" aria-labelledby="vaeBackendTitle" aria-describedby="vaeBackendNote">
      <label class="vae-choice">
        <input type="radio" name="h3VaeBackend" value="current" checked aria-label="现用 VAE">
        <span class="vae-card"><span class="vae-copy"><strong>现用 VAE</strong><small>原版解码器</small></span><span class="vae-check" aria-hidden="true"></span></span>
      </label>
      <label class="vae-choice">
        <input type="radio" name="h3VaeBackend" value="streamrender" aria-label="StreamRender VAE">
        <span class="vae-card"><span class="vae-copy"><span class="vae-name"><strong>StreamRender</strong><span class="vae-speed-badge">Faster</span></span><small>新版解码器</small></span><span class="vae-check" aria-hidden="true"></span></span>
      </label>
    </div>`;
  const style=document.createElement('style');
  style.textContent=`
    #vaeSelectorPanel{grid-column:1/-1;margin:0 0 14px;padding:12px;background:#f7f8fa;border:1px solid #e2e7ec;border-radius:14px;min-width:0}
    #vaeSelectorPanel[hidden]{display:none}
    #vaeSelectorPanel .vae-heading{display:flex;align-items:center;justify-content:space-between;gap:8px;margin:0 0 9px}
    #vaeBackendTitle{font:600 12px/1.4 system-ui,sans-serif;color:#46586a}
    #vaeBackendNote{font:400 10px/1.4 system-ui,sans-serif;color:#758394}
    #vaeSelectorPanel .vae-options{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}
    #vaeSelectorPanel .vae-choice{position:relative;display:block;margin:0;min-width:0;cursor:pointer;text-align:left}
    #vaeSelectorPanel input[type=radio]{position:absolute;clip-path:inset(50%);width:1px;height:1px;margin:0;padding:0;overflow:hidden;white-space:nowrap}
    #vaeSelectorPanel .vae-card{box-sizing:border-box;display:flex;align-items:center;justify-content:space-between;gap:4px;min-height:60px;padding:10px;border:1px solid #dce2e8;border-radius:10px;background:#fff;color:#34495d;transition:background .15s,border-color .15s,color .15s}
    #vaeSelectorPanel .vae-copy{display:grid;gap:3px;min-width:0}
    #vaeSelectorPanel .vae-name{display:flex;align-items:center;flex-wrap:wrap;gap:4px 6px}
    #vaeSelectorPanel .vae-speed-badge{display:inline-flex;align-items:center;padding:2px 5px;border-radius:5px;background:#f3ead8;color:#72582d;font:600 9px/1.2 system-ui,sans-serif;white-space:nowrap}
    #vaeSelectorPanel strong{font:600 12px/1.3 system-ui,sans-serif;white-space:nowrap}
    #vaeSelectorPanel .vae-copy small{font:400 10px/1.3 system-ui,sans-serif;color:#81909d}
    #vaeSelectorPanel .vae-check{display:grid;place-items:center;flex:0 0 12px;width:12px;height:12px;box-sizing:border-box;border:1px solid #c7d0d8;border-radius:50%}
    #vaeSelectorPanel input:not(:disabled)+.vae-card:hover{border-color:#b99d67}
    #vaeSelectorPanel input:checked+.vae-card{background:#012140;border-color:#012140;color:#fff}
    #vaeSelectorPanel input:checked+.vae-card small{color:#d5dfeb}
    #vaeSelectorPanel input:checked+.vae-card .vae-speed-badge{background:#ddc795;color:#012140}
    #vaeSelectorPanel input:checked+.vae-card .vae-check{background:#ddc795;border-color:#ddc795;color:#012140}
    #vaeSelectorPanel input:checked+.vae-card .vae-check:after{content:"✓";font:700 10px/1 system-ui,sans-serif}
    #vaeSelectorPanel input:focus-visible+.vae-card{outline:2px solid #b99d67;outline-offset:3px}
    #vaeSelectorPanel input:disabled+.vae-card{cursor:not-allowed;opacity:.65}
    @media(prefers-reduced-motion:reduce){#vaeSelectorPanel .vae-card{transition:none}}
  `;
  document.head.append(style);
  document.getElementById('start').before(panel);
  // Keep the existing start-payload interface; there is no hidden dropdown.
  const selected=document.getElementById('vaeBackend');
  const note=document.getElementById('vaeBackendNote');
  const choices=[...panel.querySelectorAll('input[type=radio]')];
  let initialized=false;
  const sync=()=>choices.forEach(input=>{input.checked=input.value===selected.value;input.disabled=selected.disabled;});
  function render(){
    if(typeof lastStatus==='undefined'||!lastStatus)return;
    const s=lastStatus,supported=Array.isArray(s.vae_options)&&s.vae_options.some(x=>x.id==='streamrender');
    panel.hidden=!supported;
    const locked=!!(s.active||s.start_pending||s.scene_preparation?.busy);
    selected.disabled=locked||!s.vae_options_ready;
    if((!initialized||s.active||s.start_pending)&&['current','streamrender'].includes(s.vae_backend)){
      selected.value=s.vae_backend;initialized=true;
    }
    note.textContent=!s.vae_options_ready?'解码器预热中':locked?'结束生成后可切换':'选择后开始生成';
    sync();
  }
  choices.forEach(input=>input.addEventListener('change',()=>{
    render();
    if(selected.disabled)return;
    selected.value=input.value;sync();
    selected.dispatchEvent(new Event('change',{bubbles:true}));
  }));
  setInterval(render,250);render();
})();

/* Move the existing prompt node; preserve gallery values, listeners and visibility. */
(()=>{
  const viewer=document.querySelector('.viewer');
  const prompt=document.getElementById('scenePromptPanel');
  if(!viewer||!prompt)return;
  viewer.append(prompt);
  const style=document.createElement('style');
  style.textContent=`
    .viewer>#scenePromptPanel{margin:10px 5px 2px;padding:14px 0 0;border:0;border-top:1px solid #eee9df;border-radius:0;background:transparent;min-width:0}
    .viewer>#scenePromptPanel label{font-size:12px;line-height:1.5;overflow-wrap:anywhere}
    .viewer #scenePrompt{display:block;box-sizing:border-box;width:100%;min-height:90px;max-height:220px;line-height:1.6}
    @media(max-width:640px){.viewer>#scenePromptPanel{margin:10px 4px 4px;padding-top:12px}}
  `;
  document.head.append(style);
})();

/* Display-only localization of the resident UI, including late status/queue updates.
 * Never rewrite input values, prompt content, request payloads, or model state. */
(()=>{
  if(window.h3EnglishUI)return;
  const phrases=[
    ['选择后自动准备',''],
    ['3 分钟不操作自动释放并重新排队','Control is released after 3 minutes of inactivity; you rejoin the queue.'],
    ['3 分钟不操作自动释放控制权','Control is released after 3 minutes of inactivity.'],
    ['已因无操作释放，请点击重新接管','Released due to inactivity. Click to take control again.'],
    ['接管确认已超时，可重新排队','Your turn expired. You can rejoin the queue.'],
    ['请在倒计时内点击接管','Take control before the timer expires.'],
    ['正在确认服务状态，暂不交接控制权','Checking service status before handing over control.'],
    ['停止结果待确认，暂不交接控制权','Waiting for stop confirmation before handing over control.'],
    ['请求结果未确认，请先查看状态；未自动重试','Request not confirmed. Check status; it was not retried.'],
    ['请求结果未确认；未自动重试，请先查看状态','Request not confirmed. Check status; it was not retried.'],
    ['排队人数已满，请稍后重试','The queue is full. Please try again later.'],
    ['控制权已释放，请重新排队','Control was released. Please rejoin the queue.'],
    ['已有会话正在使用，等待结束','Waiting for the active session to finish.'],
    ['正在安全交接控制权','Handing over control'],
    ['正在交接控制权','Handing over control'],
    ['正在结束上一段生成','Stopping the previous session'],
    ['等待上一段生成结束','Waiting for the previous session to finish'],
    ['其他人正在控制 · 可观看','Someone else is controlling · You can watch'],
    ['你的排队位置：','Your queue position: '],
    ['已加入排队，当前第 ','In queue · Position '],
    ['当前控制者无操作 ','Control will be released after '],
    [' 后释放',' of inactivity'],
    ['你已回到队尾','You have rejoined at the back of the queue'],
    ['你正在控制','You are in control'],
    ['排队第 ','Queue position '],
    [' 位',''],
    ['轮到你了','Your turn'],
    ['等待交接','Waiting for handover'],
    ['共享入口重连中','Reconnecting to shared access'],
    ['正在连接共享入口','Connecting to shared access'],
    ['共享入口暂为只读','Shared access is currently read-only'],
    ['共享会话未绑定','Shared session is not bound'],
    ['请先取得共享控制权','Take control to continue'],
    ['请排队并取得控制权','Join the queue to take control'],
    ['请先取得控制权','Take control to continue'],
    ['请点击加入排队','Click Join queue to continue'],
    ['暂时无法取得控制权','Control is temporarily unavailable'],
    ['没有控制权','You do not have control'],
    ['请求未完成','Request incomplete'],
    ['共享体验','Shared access'],
    ['共享访问','Shared access'],
    ['释放控制','Release control'],
    ['接管控制','Take control'],
    ['退出排队','Leave queue'],
    ['加入排队','Join queue'],
    ['已显示画面的时间，按原始 16 FPS 时间轴计算','Displayed video time on the original 16 FPS timeline'],
    ['可持续生成，结束后可下载视频。','Generate continuously. Download the video when finished.'],
    ['生成已结束：点击视频中央回放完整结果。','Generation complete. Click the video to replay.'],
    ['首次启动需要加载模型与缓存，请稍候','Loading the model and caches for startup. Please wait.'],
    ['正在估算剩余时间；首次启动可能需要较长时间','Estimating time remaining; first startup can take longer.'],
    ['预计还需约 ','Estimated time remaining: about '],
    ['（参考估计）',' (reference estimate)'],
    ['返回默认场景','Default scene'],
    ['服务的当前场景已切换，请重新准备所选首帧','The server scene changed. Prepare your selected scene again.'],
    ['已超过参考时间，仍在准备','Taking longer than estimated; still preparing'],
    ['比预计稍久，仍在准备…','Taking a little longer; still preparing…'],
    ['停止尚未确认，请重试','Stop not confirmed. Please retry.'],
    ['控制发送失败，请重试','Control request failed. Please retry.'],
    ['控制超时，请重试','Control request timed out. Please retry.'],
    ['请使用支持 WebCodecs 的浏览器','Please use a browser that supports WebCodecs'],
    ['浏览器不支持 H.264 分片实时预览','This browser does not support fragmented H.264 live preview'],
    ['媒体流中断，续传未完成的分片','Media stream interrupted; resuming unfinished fragments'],
    ['视频重连中，保留全部帧','Reconnecting video; all frames are retained'],
    ['视频重连中，保留已接收帧','Reconnecting video; received frames are retained'],
    ['视频连接中断，正在重连','Video disconnected; reconnecting'],
    ['连接中断，正在重连','Disconnected; reconnecting'],
    ['视频会话或分片校验失败','Video session or fragment verification failed'],
    ['视频分片校验失败，重新接收','Fragment verification failed; receiving again'],
    ['分片校验失败，重新接收','Fragment verification failed; receiving again'],
    ['视频会话校验失败','Video session verification failed'],
    ['视频分片上下文不匹配','Video fragment context mismatch'],
    ['媒体终点游标不匹配','Media end cursor mismatch'],
    ['视频分片序号不匹配','Video fragment sequence mismatch'],
    ['视频分片不连续','Non-contiguous video fragments'],
    ['缺少当前视频会话身份','Missing current video session identity'],
    ['非法媒体分片长度','Invalid media fragment length'],
    ['视频部署不匹配','Video deployment mismatch'],
    ['非法媒体通知','Invalid media event'],
    ['视频分片头不完整','Incomplete video fragment header'],
    ['视频分片不完整','Incomplete video fragment'],
    ['视频批次尾不完整','Incomplete video batch tail'],
    ['接收游标不连续','Non-contiguous receive cursor'],
    ['实时视频解码错误：','Live video decode error: '],
    ['视频解码错误：','Video decode error: '],
    ['视频分片解码失败','Video fragment decoding failed'],
    ['视频连接提前结束','Video connection ended early'],
    ['视频连接超时','Video connection timed out'],
    ['视频字节传输停滞','Video transfer stalled'],
    ['MediaSource 打开超时','MediaSource opening timed out'],
    ['MediaSource 打开失败','MediaSource opening failed'],
    ['实时 H.264 传输 HTTP ','Live H.264 transfer HTTP '],
    ['视频 HTTP ','Video HTTP '],
    ['等待后台准备','Waiting for preparation'],
    ['正在准备图片与 Prompt','Preparing image and prompt'],
    ['正在编码首帧和 Prompt','Encoding image and prompt'],
    ['正在初始化场景','Initializing scene'],
    ['正在后台预热','Warming up'],
    ['正在抓取 Graph','Capturing Graph'],
    ['正在确认稳态','Checking steady state'],
    ['场景准备失败','Scene preparation failed'],
    ['已取消准备','Preparation cancelled'],
    ['准备请求失败','Preparation request failed'],
    ['模型启动预热中','Warming up the model'],
    ['所选场景尚未就绪','Selected scene is not ready'],
    ['其他场景正在准备','Another scene is being prepared'],
    ['正在提交场景','Submitting scene'],
    ['正在准备场景','Preparing scene'],
    ['正在估算等待时间…','Estimating wait time…'],
    ['等待服务器确认…','Waiting for server confirmation…'],
    ['点击准备所选首帧','Click to prepare the selected scene'],
    ['点击重新准备','Click to prepare again'],
    ['可以重新准备','You can prepare again'],
    ['准备所选场景','Prepare scene'],
    ['正在取消…','Cancelling…'],
    ['取消准备','Cancel'],
    ['重新准备','Retry'],
    ['场景已就绪','Scene ready'],
    ['后台编译预热','Compilation warmup'],
    ['启动预热','Startup warmup'],
    ['参考还需约 ','Estimated remaining: '],
    ['预计还需约 ','About '],
    [' 分钟',' min'],
    [' 秒',' s'],
    [' 块 · 开始按钮待验收后开放',' chunks · Start is enabled after validation'],
    ['尚未生成画面','no frames generated yet'],
    ['生成中断 · 查看运行详情','Generation interrupted · See details'],
    ['正在生成首块','Generating the first chunk'],
    ['首帧预览 · 原图 ','Image preview · Original '],
    ['所选首帧 · Prompt','Selected image · Prompt'],
    ['所选首帧 · ','Selected image · '],
    ['首帧对应的 Prompt','Prompt for the selected image'],
    ['首帧图片库','Scene gallery'],
    ['图片库暂不可用','Gallery temporarily unavailable'],
    ['生成服务更新中','Generation service is updating'],
    ['选择后开始生成','Select before starting'],
    ['结束生成后可切换','Switch after stopping'],
    ['本次生成已锁定 · 结束后可切换','Locked for this session · Switch after stopping'],
    ['正在后台预热解码器','Warming up the decoder'],
    ['解码器预热中','Warming up the decoder'],
    ['VAE 解码器','VAE decoder'],
    ['视频解码器','Video decoder'],
    ['现用 VAE','Optimized Native VAE'],
    ['原版解码器','Accelerated native decoder'],
    ['新版解码器','New decoder'],
    ['StreamRender VAE · 新','StreamRender VAE · New'],
    ['返回花园','Back to garden'],
    ['花园场景','Garden scene'],
    ['选择首帧','Choose a scene'],
    ['搜索场景','Search scenes'],
    ['上一页','Previous'],
    ['下一页','Next'],
    [' 张',' scenes'],
    ['下一个生成块生效','Applies to the next chunk'],
    ['控制功能正在上线','Controls are being enabled'],
    ['正在应用…','Applying…'],
    ['↑↓←→ 移动 · 空格升 · Shift 降','↑↓←→ Move · Space up · Shift down'],
    ['移动速度','Move speed'],
    ['转向速度','Turn speed'],
    ['镜头控制','Camera controls'],
    ['键盘或点按','Keyboard or touch'],
    ['松开全部','Release keys'],
    ['下降','Down'],['上升','Up'],['前进','Forward'],['后退','Back'],
    ['左移','Left'],['右移','Right'],['抬头','Look up'],['低头','Look down'],
    ['左转','Turn left'],['右转','Turn right'],
    ['开始生成','Start generation'],
    ['结束生成','Stop generation'],
    ['下载视频','Download video'],
    ['操作面板','Controls'],
    ['实时画面','Live view'],
    ['运行详情','Details'],
    ['等待模型就绪','Waiting for the model'],
    ['等待预览','Waiting for preview'],
    ['已就绪 · 可以开始','Ready · You can start'],
    ['未就绪','Not ready'],
    ['已就绪','Ready'],
    ['已完成','Completed'],
    ['已暂停','Paused'],
    ['已中断','Interrupted'],
    ['服务异常','Service error'],
    ['异常','Error'],
    ['错误：','Error: '],
    ['准备首帧','Preparing image'],
    ['准备首块','Preparing first chunk'],
    ['启动中','Starting'],
    ['准备中','Preparing'],
    ['预热中','Warming up'],
    ['正在预热','Warming up'],
    ['正在结束','Stopping'],
    ['生成中','Generating'],
    ['连接中','Connecting'],
    ['加载中','Loading'],
    ['暂停','Pause'],
    ['继续','Resume'],
    ['可回放','Replay available'],
    ['实时预览','Live preview'],
    ['画质','Quality'],
    ['原始 ','Original '],
    ['流畅 ','Reduced '],
    ['漫游 ','Time '],
    ['逐帧 H.264：已生成 ','Frame-by-frame H.264: generated '],
    ['实时 H.264：已生成 ','Live H.264: generated '],
    [' 帧，可播到 ',' frames; available '],
    [' 帧，实际显示 ',' frames; displayed '],
    [' 帧，预览跳过过时画面约 ',' frames; skipped '],
    ['；播放积压 ','; playback backlog '],
    ['；块到屏样本 ','; chunk-to-screen samples '],
    ['，未显示块尾 ','; unseen chunk ends '],
    ['；计时校准 RTT ','; clock calibration RTT '],
    ['，误差可能约 ±','; estimated uncertainty ±'],
    ['元数据序号有缺口','metadata sequence gap'],
    ['传输/解码失败','transfer/decode failure'],
    ['最近 VAE 解码 ','Latest VAE decode '],
    ['最近 DiT ','Latest DiT '],
    ['已生成帧 ','Generated frames '],
    ['预览到屏 ','Preview-to-screen '],
    ['块到屏 ','Chunk-to-screen '],
    ['第 ','Chunk '],
    [' 块 · ',' · '],
    ['块 ','Chunk '],
    [' 帧',' frames'],
    ['生成 ','Generation '],
    ['到达 ','Arrival '],
    ['显示 ','Display '],
    ['预览 ','Preview '],
    ['存档 ','Archive '],
  ].sort((a,b)=>b[0].length-a[0].length);
  const han=/[\u3400-\u9fff]/;
  const cache=new Map();
  function translate(value){
    if(!han.test(value))return value;
    if(cache.has(value))return cache.get(value);
    let result=value;
    for(const [from,to] of phrases)result=result.split(from).join(to);
    result=result.replaceAll('：',': ').replaceAll('，',', ').replaceAll('；','; ').replaceAll('。','.');
    if(cache.size>=512)cache.clear();
    cache.set(value,result);return result;
  }
  const skip='script,style,textarea,[contenteditable],#scenePrompt';
  const attributes=['aria-label','title','placeholder','alt','data-label'];
  function visit(node){
    if(node.nodeType===Node.TEXT_NODE){
      if(node.parentElement?.closest(skip))return;
      const value=translate(node.data);if(value!==node.data)node.data=value;
      return;
    }
    if(node.nodeType!==Node.ELEMENT_NODE)return;
    for(const name of attributes){
      if(!node.hasAttribute(name))continue;
      const before=node.getAttribute(name),after=translate(before);
      if(after!==before)node.setAttribute(name,after);
    }
    if(node.matches(skip))return;
    for(const child of node.childNodes)visit(child);
  }
  document.documentElement.lang='en';
  // Short English labels fit the existing compact control panel.
  const style=document.createElement('style');
  style.textContent=`
    #sceneNote:empty{display:none}
    .scene-note{justify-content:flex-end}.scene-note #sceneNote:not(:empty){margin-right:auto}
    #vaeSelectorPanel .vae-heading{align-items:flex-start}
    #vaeSelectorPanel strong{white-space:normal}
    #vaeSelectorPanel .vae-card{height:100%;min-height:96px}
    #vaeBackendNote{max-width:135px;text-align:right}
    #keys .key{gap:4px}
    #keys .key::after{font-size:10px}
    .control-foot{flex-wrap:wrap}
    .shared-queue-copy{max-width:300px}
    @media(max-width:640px){.prepare-card{flex-wrap:wrap}.prepare-copy{flex:1;min-width:120px}.shared-queue-copy{max-width:230px}}
  `;
  document.head.append(style);
  visit(document.body);
  const observer=new MutationObserver(records=>{
    for(const record of records){
      if(record.type==='childList'){for(const node of record.addedNodes)visit(node);}
      else if(record.type==='attributes'){
        const value=record.target.getAttribute(record.attributeName);
        if(value!==null){const next=translate(value);if(next!==value)record.target.setAttribute(record.attributeName,next);}
      }else visit(record.target);
    }
  });
  observer.observe(document.body,{subtree:true,childList:true,characterData:true,attributes:true,attributeFilter:attributes});
  window.h3EnglishUI=Object.freeze({version:'en-v1'});
})();
