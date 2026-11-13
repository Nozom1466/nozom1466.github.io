import works from './works.mjs';
import {mkdir,writeFile,readFile} from 'node:fs/promises';
const directory=new URL('./cache/candidates/',import.meta.url);await mkdir(directory,{recursive:true});
for(const w of works.filter(w=>!w.bgm)){
  const file=new URL(w.slug+'.json',directory);try{await readFile(file);continue}catch{}
  const keyword=w.title.replace(/（[^）]*）/g,'').replace(/：/g,' ').replace(/ 第[一二三]季$/,'');
  try{
    const type={anime:2,book:1,game:4,film:6,tv:6,variety:6}[w.category];
    const r=await fetch('https://api.bgm.tv/v0/search/subjects?limit=5',{method:'POST',headers:{'Content-Type':'application/json','User-Agent':'PersonalLibraryCatalog/1.0'},body:JSON.stringify({keyword,sort:'match',filter:{type:[type]}}),signal:AbortSignal.timeout(15000)});
    if(!r.ok) throw Error(r.status);const data=await r.json();
    await writeFile(file,JSON.stringify(data,null,2));
    console.log(w.slug,JSON.stringify(data.data?.slice(0,3).map(x=>[x.id,x.name_cn||x.name,x.date])));
  }catch(e){console.log('ERROR',w.slug,e.message)}
}
