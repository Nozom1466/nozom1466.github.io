import {mkdir,writeFile} from 'node:fs/promises';
const queries = ['小市民','日常','氷菓','かぐや様は告らせたい','葬送のフリーレン','月がきれい','名探偵コナン','三体','徒然チルドレン','吞噬星空','秦时明月','天行九歌','けいおん','ヴァイオレット・エヴァーガーデン','四畳半神話大系','四畳半タイムマシンブルース','CITY THE ANIMATION','侣行'];
await mkdir(new URL('./cache/search/',import.meta.url),{recursive:true});
for(const keyword of (process.argv.length>2 ? process.argv.slice(2) : queries)) {
  try {
    const r=await fetch('https://api.bgm.tv/v0/search/subjects?limit=20',{method:'POST',headers:{'Content-Type':'application/json','User-Agent':'PersonalLibraryCatalog/1.0'},body:JSON.stringify({keyword,filter:{type:[2,6]}}),signal:AbortSignal.timeout(20000)});
    if(!r.ok) throw Error(r.status);
    const data=await r.json();
    await writeFile(new URL(`./cache/search/${encodeURIComponent(keyword)}.json`,import.meta.url),JSON.stringify(data,null,2));
    console.log(keyword,JSON.stringify(data.data?.map(x=>({id:x.id,name:x.name,cn:x.name_cn,date:x.date,platform:x.platform}))));
  }catch(e){console.log('ERROR',keyword,e.message)}
}
