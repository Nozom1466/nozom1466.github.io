import sharp from 'sharp';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import works from './works.mjs';
const cache=new URL('./cache/',import.meta.url);
const destination=new URL('../../static/library-covers/',import.meta.url);
await mkdir(destination,{recursive:true});
let cursor=0;
const selection=process.argv.slice(2);
const queue=works.filter(w=>!selection.length||selection.includes(w.slug));
await Promise.all(Array.from({length:3},async()=>{
  while(cursor<queue.length){
    const w=queue[cursor++];
    try {
      const path=new URL(w.slug+'.json',cache);
      const data=JSON.parse(await readFile(path));
      if(!data.image) {console.log('NO_IMAGE',w.slug);continue;}
      const staffId=w.bgm || (w.category==='anime' ? data.coverBgm : null);
      if(staffId&&!data.staff){
        const r=await fetch(`https://api.bgm.tv/v0/subjects/${staffId}/persons`,{signal:AbortSignal.timeout(20000)});
        if(r.ok){data.staff=(await r.json()).map(x=>({name:x.name,role:x.relation,url:`https://bgm.tv/person/${x.id}`}));}
      }
      if(data.asset){await writeFile(path,JSON.stringify(data,null,2)+'\n');continue;}
      let url=data.image;
      if(url.includes('upload.wikimedia.org')&&url.includes('/thumb/')) url=url.replace('/thumb/','/').replace(/\/[^/]+$/,'');
      const headers={'User-Agent':'PersonalLibraryCatalog/1.0'};
      if(data.coverSource||data.source)headers.Referer=data.coverSource||data.source;
      const r=await fetch(url,{headers,signal:AbortSignal.timeout(25000)});
      if(!r.ok) throw Error(`${r.status} ${url}`);
      const bytes=Buffer.from(await r.arrayBuffer());
      const meta=await sharp(bytes).metadata();
      data.imageOriginal=url;data.imageDimensions={width:meta.width,height:meta.height};
      // Do not crop posters, upscale low resolution, or reshape square logos.
      if(meta.width>=meta.height||meta.width<170||meta.height<240){
        data.coverNeedsReview=true;
        console.log('REVIEW_COVER',w.slug,meta.width,meta.height);
      }else{
        const file=w.slug+'.webp';
        await sharp(bytes).rotate().resize({width:640,height:960,fit:'inside',withoutEnlargement:true}).webp({quality:88}).toFile(new URL(file,destination).pathname.replace(/^\/([A-Za-z]:)/,'$1'));
        data.asset='/library-covers/'+file;
        console.log('SAVED',w.slug,meta.width,meta.height);
      }
      await writeFile(path,JSON.stringify(data,null,2)+'\n');
    }catch(e){console.log('ERROR',w.slug,e.message)}
  }
}));
