// Produce a reviewable, source-linked catalog. No personal ratings or reviews are inferred.
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import works from './works.mjs';
const catalog=[];
function fields(d){return Object.fromEntries(Object.entries(d.fields||{}).map(([k,v])=>[k.replace(/\s+/g,' ').toLowerCase(),v]));}
for(const w of works){
 const d=JSON.parse(await readFile(new URL(`./cache/${w.slug}.json`,import.meta.url)));
 const f=fields(d), cf=d.coverFields||{};
 const get=(...keys)=>keys.map(k=>f[k.toLowerCase()]).find(Boolean)||'';
 const staff=(...roles)=>(d.staff||[]).filter(p=>roles.includes(p.role)).map(p=>p.name).filter((v,i,a)=>a.indexOf(v)===i).join('、');
 const creators={};
 const put=(role,value)=>{if(value)creators[`${Object.keys(creators).length+1}_${role}`]=value.replace(/\s*\/\s*/g,', ').trim().replace(/[,、]+$/,'')};
 if(w.category==='book'){
   put('author',cf['作者']||get('author','作者'));
 }else if(['toy-story','toy-story-2','toy-story-3','zootopia','minions','minions-2'].includes(w.slug)){
   put('director',get('directed by'));
   put('writer',get('screenplay by','written by'));
   put('producer',get('produced by'));
   put('studio',get('productioncompany','productioncompanies'));
   put('music',get('music by'));
 }else if(w.category==='anime'){
   // Japanese Wikipedia series infoboxes may combine multiple adaptations. Prefer exact-subject staff.
   put('director',staff('导演','总导演')||cf['导演']||get('directed by','导演'));
   put('author',staff('原作')||cf['原作']||get('原作'));
   put('writer',staff('系列构成')||staff('脚本')||cf['脚本']||get('screenplay by','written by','脚本'));
   if(w.slug!=='she-and-her-cat')put('studio',staff('动画制作')||cf['动画制作']||get('studio','productioncompany','productioncompanies','production companies','动画制作'));
   if(!w.slug.startsWith('qin-'))put('music',staff('音乐')||cf['音乐']||get('music by','音乐'));
 }else if(w.slug==='three-body-tv'){
   put('director',cf['导演']);
   put('writer',cf['编剧']);
   put('author',cf['原作']);
   put('music',cf['音乐']);
 }else if(w.category==='game'){
   put('developer',get('developer','developers'));
   put('publisher',get('publisher','publishers'));
   put('writer',get('writer','writers'));
   put('music',get('composer','composers'));
 }else if(w.category==='documentary'){
   put('creator',get('导演'));
   put('cast',get('主演'));
   put('broadcaster',get('电视台','在线播放平台'));
 }else if(w.category==='variety'){
   put('creator',get('created by'));
   put('broadcaster',get('network'));
 }else{
   put('director',get('directed by','导演'));
   put('writer',get('screenplay by','written by','编剧'));
   put('showrunner',get('showrunners'));
   put('producer',get('produced by','制片','制片人','制片人/制作人'));
   put('music',get('music by','配乐','音乐'));
   if(Object.keys(creators).length<5){
     const based=get('based on');
     if(/\bby\s+/.test(based))put('author',based.split(/\bby\s+/).at(-1));
     else if(get('原作'))put('author',get('原作').replace(/[『《].*$/,''));
     else if(Object.keys(creators).length<4)put('studio',get('productioncompanies','production companies','制片商'));
   }
   if(w.slug.startsWith('agents-of-shield-'))put('broadcaster',get('original network'));
 }
 if(!Object.keys(creators).length)throw Error('No creators: '+w.slug);
 const sources=[d.source,d.coverSource];
 if(d.staff?.length)sources.push(`https://bgm.tv/subject/${w.bgm||d.coverBgm}`);
 const official=d.fields?.['官方网站']||cf['官方网站']||cf.website;
 if(official&&/^https?:\/\/\S+$/.test(official))sources.push(official);
 const releaseText=d.releaseYear||get('released','release dates','release date','release','publication date','上映日期')||d.date||d.coverSubject?.date;
 const year=Number(String(releaseText||'').match(/\b(19\d{2}|20\d{2})\b/)?.[1])||null;
 catalog.push({slug:w.slug,title:w.title,category:w.category,releaseYear:year,creators,cover:d.asset||null,coverDimensions:d.imageDimensions,coverSource:d.coverSource||d.imagePage||d.source,coverOriginal:d.imageOriginal||d.image,sources:[...new Set(sources.filter(Boolean))],researched:'2026-11-13'});
}
await mkdir(new URL('../../docs/',import.meta.url),{recursive:true});
await writeFile(new URL('../../docs/library-catalog.json',import.meta.url),JSON.stringify(catalog,null,2)+'\n');
for(const c of catalog)console.log(c.slug,JSON.stringify(c.creators),c.cover?'':'MISSING COVER');
console.log('Total',catalog.length);
