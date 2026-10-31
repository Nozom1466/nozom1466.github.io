// Explicit, local-only import of one note. No schedule, credentials or COS writes.
import {writeFile,readFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {COS_ORIGIN,parseShard,publicNote} from '../../static/moments/cos-feed.mjs';
import {convertArticle} from './convert.mjs';
const id=process.argv[2];
if(!/^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/i.test(id||''))throw Error('需要指定笔记 UUID');
async function read(path,max=4*1024*1024){
  const r=await fetch(COS_ORIGIN+path,{cache:'no-store',signal:AbortSignal.timeout(20000)});
  if(!r.ok)throw Error(`COS 读取失败：${r.status}`);
  const text=await r.text();if(Buffer.byteLength(text)>max)throw Error('同步文件过大');return text;
}
let marker='',entry=null;
for(let page=0;page<10;page++){
  const query=new URLSearchParams({prefix:'SnapNote_Sync/index_shards/','max-keys':'1000',marker,t:String(Date.now())});
  const xml=await read('/?'+query);
  for(const [,key] of xml.matchAll(/<Key>(SnapNote_Sync\/index_shards\/[a-f0-9]+\.json)<\/Key>/gi)){
    const text=await read('/'+key+'?t='+Date.now());if(text.startsWith('SNENC1'))continue;
    for(const r of parseShard(JSON.parse(text)))if(r.id===id&&(!entry||r.modified>entry.modified||r.modified===entry.modified&&(r.deleted&&!entry.deleted||r.deleted===entry.deleted&&r.revision>entry.revision)))entry=r;
  }
  if(!/<IsTruncated>true<\/IsTruncated>/i.test(xml))break;
  const next=xml.match(/<NextMarker>([^<]+)<\/NextMarker>/)?.[1];if(!next||next===marker||page===9)throw Error('索引分页不完整');marker=next;
}
if(!entry||entry.deleted)throw Error('没有找到有效笔记');
const body=await read('/SnapNote_Sync/notes/'+id+'__'+entry.revision+'.json',2*1024*1024);
if(!publicNote(body,entry))throw Error('笔记不可展示');
const article=convertArticle(JSON.parse(body));
for(const src of article.images){const r=await fetch(src+'?check='+Date.now(),{headers:{Range:'bytes=0-63'},signal:AbortSignal.timeout(20000)});await r.body?.cancel();if(!r.ok)throw Error('配图未同步完整：'+r.status);}
const target=new URL('../../content/ideas/'+article.filename,import.meta.url);
try{const old=await readFile(target,'utf8');if(!old.includes('"snapnote_uuid": "'+id+'"'))throw Error('目标位置已有非同步文章，停止覆盖');}catch(e){if(e.code!=='ENOENT')throw e;}
await writeFile(target,article.source);
console.log(JSON.stringify({title:article.title,images:article.images.length,draft:true,path:fileURLToPath(target)}));
