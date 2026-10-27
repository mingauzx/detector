(()=>{
'use strict';
const $=id=>document.getElementById(id),core=window.VisionCore;
const labels={capped:'Caneta com tampa',uncapped:'Caneta sem tampa',empty:'Sem caneta',uncertain:'Resultado inconclusivo'};
let data=core.createDataset(),model=null,modelPromise=null,stream=null,facing='environment',busy=false,hasPhoto=false,total=0,history=[],undo=[],cameraGeneration=0;
const video=$('video'),canvas=$('capture'),ctx=canvas.getContext('2d',{willReadFrequently:true});
function notice(text,error=false){$('notice').textContent=text;$('notice').classList.toggle('error',error);}
function sourceReady(){return (!!stream&&video.readyState>=2&&video.videoWidth>0)||hasPhoto;}
function update(){
 let samples=0,classes=0;
 for(const label of core.CLASSES){const count=data[label].length;samples+=count;if(count>=core.MIN_SAMPLES)classes++;const el=$('count-'+label);el.textContent=count+' / '+core.MIN_SAMPLES+' exemplos';el.classList.toggle('ready',count>=core.MIN_SAMPLES);}
 $('trainingStatus').textContent=classes+' / 3 classes prontas'+(classes===3?' · compare com imagens novas':' · mínimo de 8 exemplos por classe');
 $('sampleTotal').textContent=samples+' exemplos no total';
 $('analyze').disabled=busy||!sourceReady()||!core.ready(data);
 $('addSample').disabled=busy||!sourceReady();
 $('undoSample').disabled=busy||!undo.length;
 $('exportModel').disabled=busy||!samples;
 for(const id of ['photo','sampleFiles','importModel','clearModel'])$(id).disabled=busy;
 $('startCamera').disabled=busy||!!stream;
 $('stopCamera').disabled=!stream;
 $('switchCamera').disabled=busy||!stream;
}
async function exclusive(fn){
 if(busy)return;busy=true;update();
 try{await fn();}catch(error){notice(error.message||'Não foi possível concluir. Tente novamente.',true);}
 finally{busy=false;update();}
}
async function loadModel(){
 if(model)return model;
 if(modelPromise)return modelPromise;
 modelPromise=(async()=>{
  $('modelStatus').textContent='Carregando IA…';
  if(!window.tf||!window.mobilenet)throw Error('A biblioteca de IA não carregou. Confira a internet e recarregue a página.');
  try{await tf.setBackend('webgl');await tf.ready();}
  catch{await tf.setBackend('cpu');await tf.ready();}
  model=await mobilenet.load({version:2,alpha:.5});
  $('modelStatus').textContent='IA pronta · '+tf.getBackend().toUpperCase();
  return model;
 })();
 try{return await modelPromise;}catch(error){$('modelStatus').textContent='IA indisponível';modelPromise=null;throw error;}
}
function draw(source,width,height){
 if(!width||!height)throw Error('A imagem ainda não está pronta.');
 ctx.fillStyle='#808080';ctx.fillRect(0,0,canvas.width,canvas.height);
 const ratio=Math.min(canvas.width/width,canvas.height/height),w=width*ratio,h=height*ratio;
 ctx.drawImage(source,(canvas.width-w)/2,(canvas.height-h)/2,w,h);
}
function captureSource(){
 if(stream&&video.readyState>=2){draw(video,video.videoWidth,video.videoHeight);return;}
 if(!hasPhoto)throw Error('Abra a câmera ou envie uma foto primeiro.');
}
async function embedding(){
 await loadModel();
 const tensor=tf.tidy(()=>model.infer(canvas,true));
 try{const result=Array.from(await tensor.data());if(CLASSES_DIM()&&result.length!==CLASSES_DIM())throw Error('As amostras importadas não combinam com esta versão da IA.');return result;}finally{tensor.dispose();}
}
function CLASSES_DIM(){return core.CLASSES.flatMap(c=>data[c])[0]?.length||0;}
function stopCamera({announce=true}={}){
 cameraGeneration++;
 if(stream){for(const track of stream.getTracks())track.stop();stream=null;}
 video.srcObject=null;video.hidden=true;
 canvas.hidden=!hasPhoto;$('placeholder').hidden=hasPhoto;$('sourceStatus').textContent=hasPhoto?'Foto carregada':'Câmera desligada';
 update();if(announce)notice('Câmera desligada. Você pode continuar com uma foto.');
}
function cameraError(error){
 return {NotAllowedError:'A câmera foi bloqueada. Permita o acesso nas configurações do navegador ou use “Enviar foto”.',NotFoundError:'Não encontramos uma câmera disponível. Use “Enviar foto”.',NotReadableError:'A câmera pode estar em uso por outro aplicativo. Feche-o e tente novamente.',OverconstrainedError:'Essa configuração de câmera não está disponível. Use “Enviar foto”.'}[error.name]||'Não foi possível abrir a câmera. Tente Chrome ou Safari atualizado, ou use “Enviar foto”.';
}
async function startCamera(){
 if(busy)return;
 if(!window.isSecureContext||!navigator.mediaDevices?.getUserMedia){notice('Para usar a câmera, abra o site em HTTPS num navegador compatível. Enviar foto continua disponível.',true);return;}
 const generation=++cameraGeneration;
 $('startCamera').disabled=true;notice('Aguardando permissão para usar a câmera…');
 let acquired;
 try{
  acquired=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:facing},width:{ideal:640},height:{ideal:480}},audio:false});
  if(generation!==cameraGeneration){acquired.getTracks().forEach(t=>t.stop());return;}
  stream=acquired;video.srcObject=stream;video.hidden=false;canvas.hidden=true;$('placeholder').hidden=true;
  await video.play();
  if(!video.videoWidth)await new Promise((resolve,reject)=>{const timeout=setTimeout(()=>{video.removeEventListener('loadeddata',ready);reject(Error('Câmera sem imagem.'));},10000);function ready(){clearTimeout(timeout);resolve();}video.addEventListener('loadeddata',ready,{once:true});});
  if(generation!==cameraGeneration)return;
  for(const track of stream.getVideoTracks())track.addEventListener('ended',()=>{if(stream?.getTracks().includes(track))stopCamera();});
  $('sourceStatus').textContent='Câmera ao vivo';notice('Câmera pronta. Selecione a classe e capture exemplos com posições diferentes.');update();
 }catch(error){acquired?.getTracks().forEach(t=>t.stop());if(generation===cameraGeneration){stopCamera({announce:false});notice(cameraError(error),true);}}
 finally{update();}
}
async function imageFromFile(file){
 if(!file||file.size>12*1024*1024)throw Error('Use uma imagem de até 12 MB.');
 if(!file.type.startsWith('image/'))throw Error('Escolha um arquivo de imagem.');
 const url=URL.createObjectURL(file),img=new Image();
 try{await new Promise((resolve,reject)=>{img.onload=resolve;img.onerror=()=>reject(Error('Não consegui abrir a imagem. Tente JPG, PNG ou WebP.'));img.src=url;});if(img.naturalWidth*img.naturalHeight>24000000)throw Error('A imagem é grande demais. Reduza a resolução antes de enviar.');return img;}finally{URL.revokeObjectURL(url);}
}
function showPhoto(img){
 stopCamera({announce:false});draw(img,img.naturalWidth,img.naturalHeight);hasPhoto=true;canvas.hidden=false;video.hidden=true;$('placeholder').hidden=true;$('sourceStatus').textContent='Foto carregada';update();
}
function clearResult(){
 $('resultTitle').textContent='Treinamento atualizado';$('resultText').textContent='Analise uma imagem nova para testar o que foi aprendido.';$('scores').hidden=true;
}
function selected(){return document.querySelector('input[name="class"]:checked').value;}
function renderHistory(){
 const body=$('history');body.replaceChildren();
 if(!history.length){const tr=document.createElement('tr'),td=document.createElement('td');td.colSpan=4;td.textContent='Nenhuma análise realizada.';tr.append(td);body.append(tr);return;}
 for(const entry of history){const tr=document.createElement('tr');for(const value of [entry.time,labels[entry.label],Math.round(entry.agreement*100)+'% dos 5 vizinhos',entry.elapsed+' ms']){const td=document.createElement('td');td.textContent=value;tr.append(td);}body.append(tr);}
}
$('startCamera').addEventListener('click',startCamera);
$('stopCamera').addEventListener('click',()=>stopCamera());
$('switchCamera').addEventListener('click',async()=>{stopCamera({announce:false});facing=facing==='environment'?'user':'environment';await startCamera();});
$('photo').addEventListener('change',event=>void exclusive(async()=>{const file=event.target.files[0];if(!file)return;showPhoto(await imageFromFile(file));notice('Foto pronta. Você pode adicioná-la como exemplo ou analisar após preparar as três classes.');event.target.value='';}));
$('addSample').addEventListener('click',()=>void exclusive(async()=>{
 const label=selected();captureSource();core.addExample(data,label,await embedding());undo.push(label);clearResult();notice('Exemplo adicionado em “'+labels[label]+'”. Varie posição e iluminação para a próxima captura.');
}));
$('sampleFiles').addEventListener('change',event=>void exclusive(async()=>{
 const files=[...event.target.files],label=selected();event.target.value='';if(!files.length)return;
 if(data[label].length+files.length>core.MAX_SAMPLES)throw Error('Cada classe aceita até 60 imagens.');
 await loadModel();let added=0,skipped=0;
 for(const file of files){
  try{const img=await imageFromFile(file);showPhoto(img);core.addExample(data,label,await embedding());undo.push(label);added++;}
  catch(error){skipped++;notice(error.message,true);}
  update();await tf.nextFrame();
 }
 clearResult();notice(added+' exemplo(s) adicionado(s) em “'+labels[label]+'”.'+(skipped?' '+skipped+' arquivo(s) não adicionado(s); confira formato, tamanho ou imagens repetidas.':''));
}));
$('undoSample').addEventListener('click',()=>{if(busy)return;const label=undo.pop();if(label)data[label].pop();clearResult();update();notice('Última amostra removida.');});
$('analyze').addEventListener('click',()=>void exclusive(async()=>{
 await loadModel();captureSource();const started=performance.now(),result=core.classify(data,await embedding()),elapsed=Math.round(performance.now()-started);total++;
 $('analyses').textContent=total;$('elapsed').textContent=elapsed+' ms';$('resultTitle').textContent=labels[result.label];
 $('resultIcon').textContent={capped:'✓',uncapped:'!',empty:'○',uncertain:'?'}[result.label];
 $('resultText').textContent={capped:'A imagem se parece com os exemplos de ponta coberta. Confira visualmente para validar.',uncapped:'A imagem se parece com os exemplos de ponta exposta. Isso é uma hipótese, não uma localização da tampa.',empty:'A imagem se parece com os exemplos sem caneta.',uncertain:'Os exemplos não permitem uma separação clara. Melhore o enquadramento ou acrescente exemplos variados.'}[result.label];
 const scores=$('scores');scores.replaceChildren();scores.hidden=false;
 for(const label of core.CLASSES){const row=document.createElement('div');row.className='score-row';const line=document.createElement('div');line.className='score-label';const name=document.createElement('span'),value=document.createElement('span');name.textContent=labels[label];value.textContent=Math.round(result.scores[label]*100)+'%';line.append(name,value);const bar=document.createElement('div');bar.className='score-bar';const fill=document.createElement('span');fill.style.width=(result.scores[label]*100)+'%';bar.append(fill);row.append(line,bar);scores.append(row);}
 const note=document.createElement('p');note.className='smallnote no-pad';note.textContent='Concordância dos 5 exemplos mais próximos; não é probabilidade de acerto.';scores.append(note);
 history.unshift({time:new Date().toLocaleTimeString('pt-BR'),label:result.label,agreement:result.agreement,elapsed});history=history.slice(0,50);renderHistory();notice('Análise concluída. Valide a hipótese usando uma imagem que não fez parte dos exemplos.');
}));
$('clearHistory').addEventListener('click',()=>{history=[];total=0;$('analyses').textContent='0';$('elapsed').textContent='—';renderHistory();notice('Registros da sessão apagados. O treinamento foi mantido.');});
$('clearModel').addEventListener('click',()=>{if(busy||!confirm('Apagar os exemplos de treinamento desta aba? Exporte antes para conservar uma cópia.'))return;data=core.createDataset();undo=[];clearResult();update();notice('Treinamento apagado. Capture novos exemplos.');});
$('exportModel').addEventListener('click',()=>{if(busy)return;const blob=new Blob([core.serialize(data)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='visioncheck-treinamento.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);notice('Treinamento exportado. Guarde o arquivo para importar depois.');});
$('importModel').addEventListener('change',event=>void exclusive(async()=>{
 const file=event.target.files[0];event.target.value='';if(!file)return;
 if(file.size>8*1024*1024)throw Error('Use um treinamento de até 8 MB.');
 const next=core.deserialize(await file.text());
 await loadModel();const probe=tf.zeros([224,224,3]);let feature;
 try{feature=model.infer(probe,true);const size=feature.size;for(const label of core.CLASSES)if(next[label].some(v=>v.length!==size))throw Error('Treinamento incompatível com este modelo.');}finally{probe.dispose();feature?.dispose();}
 data=next;undo=[];clearResult();notice('Treinamento importado. Teste com uma imagem nova antes da apresentação.');
}));
window.addEventListener('pagehide',()=>stopCamera({announce:false}));
document.addEventListener('visibilitychange',()=>{if(document.hidden&&stream)stopCamera();});
update();
})();