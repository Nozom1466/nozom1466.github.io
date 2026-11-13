import {readFile,writeFile} from 'node:fs/promises';
import works from './works.mjs';
import selection from './cover-selection.mjs';
for(const [slug,id] of Object.entries(selection)){
 try{
  const path=new URL(`./cache/${slug}.json`,import.meta.url);
  let data;try{data=JSON.parse(await readFile(path));}catch{data={...works.find(x=>x.slug===slug),fields:{}}}
  if(data.coverBgm===id)continue;
  const r=await fetch(`https://api.bgm.tv/v0/subjects/${id}`,{signal:AbortSignal.timeout(20000)});if(!r.ok)throw Error(r.status);
  const raw=await r.json();data.coverBgm=id;data.coverSource=`https://bgm.tv/subject/${id}`;
  data.image=raw.images.large;delete data.asset;delete data.coverNeedsReview;
  data.coverSubject={name:raw.name,name_cn:raw.name_cn,date:raw.date};
  data.coverFields=Object.fromEntries((raw.infobox||[]).map(x=>[x.key,typeof x.value==='string'?x.value:x.value.map(y=>y.v).join(' / ')]));
  if(!data.heading||data.heading==='Michihito Fujii'){
    data.heading=raw.name;data.source=data.coverSource;data.fields=data.coverFields;data.date=raw.date;
  }
  await writeFile(path,JSON.stringify(data,null,2)+'\n');
  console.log(slug,raw.name_cn||raw.name);
 }catch(e){console.log('ERROR',slug,e.message)}
}
