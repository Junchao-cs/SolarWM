/* Scene selection is carried into the owned backend generation request. */
(()=>{
  const $=id=>document.getElementById(id);
  const grid=$('galleryGrid'), search=$('gallerySearch');
  let items=[],filtered=[],page=0;
  let initialSelectionResolved=false;
  const size=10;
  const defaultTitle='snow';
  let defaultItem=null;
  let uploading=false;
  const busy=()=>uploading||(typeof lastStatus!=='undefined'&&(lastStatus?.active||lastStatus?.start_pending||lastStatus?.scene_uploading||lastStatus?.scene_preparation?.busy));
  window.gallerySelection=null;
  window.galleryPreview=false;
  function restoreReadySelection(){
    if(initialSelectionResolved||!items.length||typeof lastStatus==='undefined')return;
    const s=lastStatus,p=s?.scene_preparation;
    if(!s)return;
    // A passive page does not change the server scene. Reflect an in-flight
    // session if visible, otherwise default to Snow and offer explicit Prepare.
    const item=(s?.active||s?.start_pending||p?.busy)?
      items.find(i=>i.id===s.scene_id):defaultItem;
    if(!item)return;
    initialSelectionResolved=true;
    window.gallerySelection=item;
    window.galleryPreview=!s.active&&!s.start_pending;
    $('scenePrompt').value=item.prompt;
    $('sceneLabel').textContent='所选首帧 · '+item.filename;
    $('scenePromptPanel').hidden=false;
    if(window.galleryPreview){
      resetLive();
      $('screen').src=item.preview;$('screen').alt=item.title;
    }
    // Restore only this tab's selection. Never POST /prepare, switch the
    // backend, or override a subsequent explicit choice in this tab.
  }
  window.refreshGalleryState=()=>{
    restoreReadySelection();
    const locked=!!busy();
    for(const button of grid.querySelectorAll('button')){
      button.disabled=locked;
      button.setAttribute('aria-pressed',String(window.gallerySelection?.id===button.dataset.sceneId));
    }
    $('returnDefault').disabled=locked;
    for(const input of $('uploadSceneForm').querySelectorAll('input,textarea,button'))
      input.disabled=locked||!lastStatus?.ready;
    document.querySelector('.resolution').hidden=false;
    const canGenerate=typeof lastStatus!=='undefined'&&lastStatus?.gallery_generation;
    $('sceneNote').textContent=canGenerate?'选择后自动准备':'生成服务更新中';
    if(window.gallerySelection&&window.galleryPreview&&!locked){
      $('screen').hidden=false;$('live').hidden=true;$('replay').hidden=true;
      $('start').disabled=!canGenerate||!lastStatus?.ready;$('compactFps').hidden=true;
    }
  };
  function select(item){
    if(busy())return;
    initialSelectionResolved=true;
    resetLive();
    window.gallerySelection=item;
    window.galleryPreview=true;
    $('screen').src=item.preview;$('screen').alt=item.title;
    $('scenePrompt').value=item.prompt;
    $('sceneLabel').textContent='所选首帧 · '+item.filename;
    $('scenePromptPanel').hidden=false;
    $('compactStatus').textContent='首帧预览 · 原图 '+item.width+' × '+item.height;
    render();window.refreshGalleryState();window.prepareSelectedScene?.();
  }
  function render(){
    grid.replaceChildren();
    for(const item of filtered.slice(page*size,(page+1)*size)){
      const b=document.createElement('button');b.className='scene-card';b.type='button';
      b.dataset.sceneId=item.id;
      b.setAttribute('aria-pressed',String(window.gallerySelection?.id===item.id));
      b.title=item.title+' · '+item.filename;
      const img=document.createElement('img');img.src=item.thumbnail;img.alt=item.title;img.loading='lazy';img.decoding='async';
      const label=document.createElement('span');label.textContent=item.title;
      b.append(img,label);b.onclick=()=>select(item);grid.append(b);
    }
    $('galleryCount').textContent=filtered.length+' 张';
    $('galleryPage').textContent=(filtered.length?page+1:0)+' / '+Math.ceil(filtered.length/size);
    $('galleryPrev').disabled=page===0;
    $('galleryNext').disabled=(page+1)*size>=filtered.length;
    document.querySelector('.gallery-pages').hidden=filtered.length<=size;
    window.refreshGalleryState();
  }
  $('galleryPrev').onclick=()=>{if(page>0){page--;render()}};
  $('galleryNext').onclick=()=>{if((page+1)*size<filtered.length){page++;render()}};
  search.oninput=()=>{const q=search.value.trim().toLowerCase();filtered=items.filter(i=>(i.title+' '+i.scene+' '+i.prompt).toLowerCase().includes(q));page=0;render()};
  $('returnDefault').onclick=()=>{if(defaultItem)select(defaultItem);};
  async function reload(){
      const response=await fetch('/gallery/manifest.json');
      if(!response.ok)throw Error('gallery unavailable');
      const data=await response.json();
      const available=data.items;
      defaultItem=available.find(item=>item.title.toLowerCase()===defaultTitle)||available[0];
      items=defaultItem?[defaultItem,...available.filter(item=>item.id!==defaultItem.id)]:[];
      window.galleryDefaultSceneId=defaultItem?.id||null;
      filtered=items;search.value='';page=0;render();
  }
  const readImage=file=>new Promise((resolve,reject)=>{
    const reader=new FileReader();reader.onerror=()=>reject(Error('Cannot read image'));
    reader.onload=()=>resolve(reader.result.split(',')[1]);reader.readAsDataURL(file);
  });
  let selectedImage=null;
  $('useSelectedImage').onclick=async()=>{
    if(!window.gallerySelection||busy())return;
    try{
      const response=await fetch(window.gallerySelection.preview);
      if(!response.ok)throw Error('Cannot read selected image');
      selectedImage=await response.blob();
      $('uploadImage').value='';$('uploadPrompt').value=window.gallerySelection.prompt;
      $('uploadMessage').textContent='Using '+window.gallerySelection.title+' — edit the prompt below.';
    }catch(error){$('uploadMessage').textContent=error.message;}
  };
  $('uploadImage').onchange=()=>{selectedImage=null;$('uploadMessage').textContent='';};
  $('uploadSceneForm').onsubmit=async event=>{
    event.preventDefault();
    if(busy())return;
    const file=$('uploadImage').files[0]||selectedImage,prompt=$('uploadPrompt').value.trim();
    if(!file||!prompt){$('uploadMessage').textContent='Choose an image and enter a prompt.';return;}
    if(file.size>8*1024*1024){$('uploadMessage').textContent='Image must be at most 8 MiB.';return;}
    uploading=true;window.refreshGalleryState();$('uploadMessage').textContent='Uploading…';
    try{
      const result=await post('/scenes',{image:await readImage(file),prompt});
      await reload();
      // Refresh server state before selecting; passive polls never prepare.
      status(await (await fetch('/status')).json());
      uploading=false;
      const item=items.find(item=>item.id===result.scene_id);
      if(!item)throw Error('Uploaded scene is not available');
      page=Math.floor(items.indexOf(item)/size);select(item);
      $('uploadMessage').textContent='Saved to your scene gallery.';
    }catch(error){$('uploadMessage').textContent=error.message;}
    finally{uploading=false;window.refreshGalleryState();}
  };
  reload()
    .catch(()=>{$('galleryCount').textContent='图片库暂不可用';$('galleryPrev').disabled=true;$('galleryNext').disabled=true});
})();
