import {load} from 'cheerio';
import {writeFile,mkdir} from 'node:fs/promises';
await mkdir(new URL('./cache/pages/',import.meta.url),{recursive:true});
await Promise.all(process.argv.slice(2).map(async (url,index)=>{
 try{
 const r=await fetch(url,{signal:AbortSignal.timeout(18000)});const html=await r.text();const $=load(html);
 const file=encodeURIComponent(url)+'.html';await writeFile(new URL('./cache/pages/'+encodeURIComponent(file),import.meta.url),html);
 console.log(JSON.stringify({url,status:r.status,title:$('title').text(),text:$('body').text().replace(/\s+/g,' ').slice(0,1800),images:$('img').map((i,x)=>({src:$(x).attr('src'),alt:$(x).attr('alt')})).get().slice(0,30),og:$('meta[property="og:image"]').attr('content')}));
 }catch(e){console.log(url,e.message)}
}));
