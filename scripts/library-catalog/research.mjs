import {load} from 'cheerio';
import {mkdir,readFile,writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import works from './works.mjs';
const cache = fileURLToPath(new URL('./cache/', import.meta.url));
await mkdir(cache,{recursive:true});
export async function get(url) {
  const response = await fetch(url,{headers:{'User-Agent':'PersonalLibraryCatalog/1.0 (source research; personal blog)'},signal:AbortSignal.timeout(25000)});
  if(!response.ok) throw Error(`${response.status}: ${url}`);
  return response;
}
export function parse(html,url) {
  const $=load(html);
  const box=$('table.infobox').first();
  const fields={};
  box.find('sup,.reference,.noprint,style').remove();
  box.find('tr').each((_,tr)=>{
    const key=$(tr).children('th').text().trim();
    const value=$(tr).children('td').clone();
    value.find('br').replaceWith(' / ');
    value.find('li').append(' / ');
    if(key&&value.length) fields[key]=value.text().replace(/\s+/g,' ').replace(/( \/ )+$/,'').trim();
  });
  const img=box.find('img').first();
  const image=img.attr('src') ? new URL(img.attr('src'),url).href : null;
  return {heading:$('#firstHeading').text().trim(),fields,image,imageCaption:box.find('.infobox-caption').text().trim(),imagePage:img.closest('a').attr('href')?new URL(img.closest('a').attr('href'),url).href:null};
}
const selected = process.argv.slice(2);
let cursor=0;
const queue=works.filter(w=>!selected.length||selected.includes(w.slug));
await Promise.all(Array.from({length:4},async()=>{
  while(cursor<queue.length) {
    const w=queue[cursor++];const path=`${cache}/${w.slug}.json`;
    try {
      if(!process.env.RESEARCH_FORCE) {try{await readFile(path);continue;}catch{}}
      if(w.bgm) {
        const raw=await (await get(`https://api.bgm.tv/v0/subjects/${w.bgm}`)).json();
        const fields=Object.fromEntries((raw.infobox||[]).map(x=>[x.key,typeof x.value==='string'?x.value:x.value.map(y=>y.v).join(' / ')]));
        const data={...w,source:`https://bgm.tv/subject/${w.bgm}`,retrieved:'2026-11-13',heading:raw.name,fields,image:raw.images.large,imageCaption:'作品宣传图',date:raw.date};
        await writeFile(path,JSON.stringify(data,null,2)+'\n');
        console.log(w.slug,JSON.stringify(fields));continue;
      }
      const lang=w.wiki.match(/^(ja|zh):/)?.[1]||'en';
      const title=w.wiki.replace(/^(ja|zh):/,'');
      const url=`https://${lang}.wikipedia.org/wiki/${encodeURIComponent(title.replaceAll(' ','_'))}`;
      const r=await get(url);const html=await r.text();
      const data={...w,source:r.url,retrieved:'2026-11-13',...parse(html,r.url)};
      await writeFile(path,JSON.stringify(data,null,2)+'\n');
      console.log(w.slug,JSON.stringify(data.fields),data.image?'COVER':'NO COVER');
    }catch(e){console.log('ERROR',w.slug,e.message);}
  }
}));
