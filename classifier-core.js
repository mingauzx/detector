(function(root){
'use strict';
const CLASSES=['capped','uncapped','empty'];
const MODEL_ID='mobilenet-v2-alpha050-embedding-v1';
const MIN_SAMPLES=8,MAX_SAMPLES=60;
function normalize(vector){
 if(!Array.isArray(vector)&&!(vector instanceof Float32Array))throw Error('Vetor inválido.');
 if(vector.length<2||vector.length>2048||[...vector].some(n=>typeof n!=='number'||!Number.isFinite(n)))throw Error('Características inválidas.');
 const norm=Math.hypot(...vector);if(norm<1e-8)throw Error('Não foi possível extrair características da imagem.');
 return Array.from(vector,n=>n/norm);
}
function dot(a,b){if(a.length!==b.length)throw Error('Dimensões incompatíveis.');return a.reduce((sum,n,i)=>sum+n*b[i],0);}
function createDataset(){return {capped:[],uncapped:[],empty:[]};}
function ready(data){return CLASSES.every(c=>data[c]?.length>=MIN_SAMPLES);}
function addExample(data,label,vector){
 if(!CLASSES.includes(label))throw Error('Classe inválida.');
 if(data[label].length>=MAX_SAMPLES)throw Error('Limite de 60 exemplos por classe. Exporte ou reinicie o treinamento.');
 const value=normalize(vector),existing=CLASSES.flatMap(c=>data[c]);
 if(existing.length&&existing[0].length!==value.length)throw Error('Treinamento incompatível com o modelo carregado.');
 if(existing.some(v=>dot(v,value)>.999999))throw Error('Essa imagem já foi adicionada ou está quase idêntica. Mude um pouco o enquadramento.');
 data[label].push(value);return value;
}
function classify(data,vector){
 if(!ready(data))throw Error('Capture pelo menos 8 exemplos de cada classe antes de analisar.');
 const query=normalize(vector),ordered=CLASSES.flatMap(label=>data[label].map(v=>({label,similarity:dot(v,query)}))).sort((a,b)=>b.similarity-a.similarity),neighbors=ordered.slice(0,5);
 const counts=Object.fromEntries(CLASSES.map(c=>[c,0]));
 for(const item of neighbors)counts[item.label]++;
 const scores=Object.fromEntries(CLASSES.map(c=>[c,counts[c]/neighbors.length]));
 const ranking=CLASSES.map(label=>({label,score:scores[label]})).sort((a,b)=>b.score-a.score);
 const top=ranking[0],boundaryTie=ordered.slice(5).some(n=>n.label!==top.label&&n.similarity>=neighbors[4].similarity-1e-6),inconclusive=boundaryTie||top.score<.8||top.score-ranking[1].score<.3||neighbors[0].similarity<.7;
 return {label:inconclusive?'uncertain':top.label,scores,agreement:top.score,similarity:neighbors[0].similarity};
}
function serialize(data){return JSON.stringify({version:1,model:MODEL_ID,samples:data});}
function deserialize(text){
 const parsed=JSON.parse(text);
 if(parsed?.version!==1||parsed.model!==MODEL_ID||!parsed.samples)throw Error('Arquivo de treinamento incompatível.');
 const next=createDataset();let dimension;
 for(const label of CLASSES){
  const samples=parsed.samples[label];if(!Array.isArray(samples)||samples.length>MAX_SAMPLES)throw Error('Quantidade de exemplos inválida.');
  for(const vector of samples){
   if(!Array.isArray(vector)||vector.length<128)throw Error('Arquivo de características inválido.');
   const v=normalize(vector);
   if(dimension&&v.length!==dimension)throw Error('Exemplos com dimensões diferentes.');
   dimension=v.length;next[label].push(v);
  }
 }
 return next;
}
const api={CLASSES,MODEL_ID,MIN_SAMPLES,MAX_SAMPLES,normalize,createDataset,ready,addExample,classify,serialize,deserialize};
if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.VisionCore=api;
})(typeof window!=='undefined'?window:globalThis);
