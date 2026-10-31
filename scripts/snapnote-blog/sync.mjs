import {createHash} from 'node:crypto';
import {mkdir,readdir,readFile,writeFile,unlink,appendFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {COS_ORIGIN,parseShard,publicNote} from '../../static/moments/cos-feed.mjs';
import {channelOf} from '../../static/moments/channels.mjs';
import {convertArticle} from './convert.mjs';
import {convertReview} from './review.mjs';

const uuid=/^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/i;
export async function collectArticles({fetcher=fetch}={}){
  async function read(url,max=4*1024*1024){
    const r=await fetcher(url,{cache:'no-store',signal:AbortSignal.timeout(20000)});
    if(!r.ok)throw Error(`COS 读取失败（${r.status}），保留线上内容`);
    if(Number(r.headers.get('content-length'))>max)throw Error('同步文件过大');
    const reader=r.body.getReader(),chunks=[];let size=0;
    try{while(true){const {value,done}=await reader.read();if(done)break;size+=value.length;if(size>max){await reader.cancel();throw Error('同步文件过大');}chunks.push(Buffer.from(value));}}finally{reader.releaseLock();}
    return Buffer.concat(chunks).toString('utf8');
  }
  const latest=new Map();let marker='';
  for(let page=0;page<100;page++){
    const query=new URLSearchParams({prefix:'SnapNote_Sync/index_shards/','max-keys':'1000',marker,t:String(Date.now())});
    const xml=await read(COS_ORIGIN+'/?'+query);
    if(!/<ListBucketResult[\s>]/.test(xml)||!/<IsTruncated>(true|false)<\/IsTruncated>/.test(xml))throw Error('COS 索引列表不完整');
    for(const [,key] of xml.matchAll(/<Key>(SnapNote_Sync\/index_shards\/[a-f0-9]+\.json)<\/Key>/gi)){
      const text=await read(COS_ORIGIN+'/'+key+'?t='+Date.now());if(text.startsWith('SNENC1'))continue;
      for(const r of parseShard(JSON.parse(text))){const old=latest.get(r.id);if(!old||r.modified>old.modified||r.modified===old.modified&&(r.deleted&&!old.deleted||r.deleted===old.deleted&&r.revision>old.revision))latest.set(r.id,r);}
    }
    if(/<IsTruncated>false<\/IsTruncated>/.test(xml))break;
    const next=xml.match(/<NextMarker>([^<]+)<\/NextMarker>/)?.[1];if(!next||next===marker||page===99)throw Error('索引分页不完整');marker=next;
  }
  const entries=[...latest.values()].filter(e=>!e.deleted),articles=[];let offset=0;
  await Promise.all(Array.from({length:Math.min(4,entries.length)},async()=>{
    while(offset<entries.length){
      const entry=entries[offset++];
      const text=await read(COS_ORIGIN+'/SnapNote_Sync/notes/'+entry.id+'__'+entry.revision+'.json',2*1024*1024);
      if(text.startsWith('SNENC1'))throw Error('存在加密笔记，无法确认分类，停止同步');
      const raw=JSON.parse(text);
      // Validate even non-blog records so a partial response cannot remove articles.
      const channel=channelOf(raw);
      if(!publicNote(text,entry)||!['blog','review'].includes(channel))continue;
      const article=channel==='review'?convertReview(raw,{draft:false,skipDrafts:true}):convertArticle(raw,{draft:false,skipDrafts:true});
      if(article)articles.push(article);
    }
  }));
  articles.sort((a,b)=>a.filename.localeCompare(b.filename));
  for(const src of new Set(articles.flatMap(a=>a.images))){
    const r=await fetcher(src+'?check='+Date.now(),{headers:{Range:'bytes=0-63'},signal:AbortSignal.timeout(20000)});
    const ok=r.ok&&/^image\//i.test(r.headers.get('content-type')||'');await r.body?.cancel();if(!ok)throw Error('配图尚未就绪，保留线上内容');
  }
  const digest=createHash('sha256').update(JSON.stringify(articles.map(a=>[a.section||'ideas',a.filename,a.source]))).digest('hex');
  return {articles,digest,ids:articles.map(a=>a.filename.slice(9,-3)),indexedIds:[...latest.keys()]};
}

export async function writeArticles(articles,output){
  output=path.resolve(output);await mkdir(output,{recursive:true});
  const current=new Map();
  for(const name of await readdir(output)){
    const id=name.startsWith('snapnote-')&&name.endsWith('.md')?name.slice(9,-3):'';
    if(!uuid.test(id))continue;
    const text=await readFile(path.join(output,name),'utf8');
    if(!text.includes('"snapnote_uuid": "'+id+'"'))throw Error('同名文件不是同步文章，停止覆盖');
    current.set(name,text);
  }
  for(const a of articles){if(!uuid.test(a.filename.slice(9,-3))||a.filename!=='snapnote-'+a.filename.slice(9,-3)+'.md')throw Error('生成文件名无效');}
  // Only generated files in the exact content directory may be replaced/removed.
  for(const a of articles)await writeFile(path.join(output,a.filename),a.source);
  const keep=new Set(articles.map(a=>a.filename));for(const name of current.keys())if(!keep.has(name))await unlink(path.join(output,name));
}

export function needsDeployment(snapshot,published,commit){return !published||published.digest!==snapshot.digest||published.commit!==commit;}

async function main(){
  const root=fileURLToPath(new URL('../../',import.meta.url));
  const snapshot=await collectArticles();
  let published=null;
  if(process.env.GITHUB_ACTIONS==='true'){
    const r=await fetch('https://nozom1466.github.io/snapnote-version.json?check='+Date.now(),{cache:'no-store',signal:AbortSignal.timeout(20000)});
    if(r.ok){published=await r.json();if(published.schema!==1||!Array.isArray(published.ids))throw Error('线上同步标记格式无效');}
    else if(r.status!==404)throw Error('无法检查线上版本，停止部署');
    if(published?.ids.some(id=>!snapshot.indexedIds.includes(id)))throw Error('已发布文章缺少索引记录，停止自动撤稿，请检查云端同步');
  }
  await writeArticles(snapshot.articles.filter(a=>a.section!=='library'),path.join(root,'content/ideas'));
  await writeArticles(snapshot.articles.filter(a=>a.section==='library'),path.join(root,'content/library'));
  const commit=process.env.GITHUB_SHA||'local';
  const changed=needsDeployment(snapshot,published,commit);
  await writeFile(path.join(root,'static/snapnote-version.json'),JSON.stringify({schema:1,digest:snapshot.digest,commit,ids:snapshot.ids}));
  if(process.env.GITHUB_OUTPUT)await appendFile(process.env.GITHUB_OUTPUT,`changed=${changed}\n`);
  console.log(JSON.stringify({articles:snapshot.articles.length,changed,titles:snapshot.articles.map(a=>a.title)}));
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url))await main();
