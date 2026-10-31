import {parseLines} from './render.mjs';
import {channelOf} from './channels.mjs';
export const COS_ORIGIN='https://snapnote-test-1316371247.cos.ap-shanghai.myqcloud.com';
const PREFIX='SnapNote_Sync/', SHARDS=PREFIX+'index_shards/';
const uuid=/^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/i;
const revision=/^[a-f0-9]{32,64}$/i;
const imageName=x=>typeof x==='string'&&/^[\w.-]+\.(jpe?g|png|webp|gif)$/i.test(x)&&!x.includes('..');

export function parseListing(text){
  const doc=new DOMParser().parseFromString(text,'application/xml');
  if(doc.querySelector('parsererror')||doc.documentElement.localName!=='ListBucketResult')throw Error('COS 文件列表格式异常。');
  const value=(el,name)=>el.getElementsByTagName(name)[0]?.textContent||'';
  return {files:[...doc.getElementsByTagName('Contents')].map(el=>({key:value(el,'Key'),etag:value(el,'ETag'),size:Number(value(el,'Size'))})),truncated:value(doc,'IsTruncated')==='true',marker:value(doc,'NextMarker')};
}

// SnapNote index_shards rows: [uuid, deleted, lastModifiedMs, vectorClock, revisionId].
export function parseShard(rows){
  if(!Array.isArray(rows))throw Error('闪念索引格式不支持。');
  return rows.map(r=>{
    if(!Array.isArray(r)||!uuid.test(r[0])||![0,1,false,true].includes(r[1])||!Number.isFinite(r[2])||!revision.test(r[4]))throw Error('闪念索引字段不完整。');
    return {id:r[0],deleted:!!r[1],modified:r[2],revision:r[4]};
  });
}

export function publicNote(text,entry,channel=null){
  if(text.startsWith('SNENC1'))return null;
  let raw;try{raw=JSON.parse(text);}catch{throw Error('笔记尚未同步完整，请稍后刷新。');}
  const n=raw.note;
  if(!n||n.uuid!==entry.id||raw.revisionId&&raw.revisionId!==entry.revision)throw Error('笔记版本不一致，请稍后刷新。');
  if(n.hasMarkDeleted||n.isLocked||n.isArchived||raw.stateVector?.deleted)return null;
  if(channel && channelOf(raw)!==channel)return null;
  if(!Number.isFinite(n.createdAtMs)||!Number.isFinite(new Date(n.createdAtMs).getTime()))throw Error('笔记日期无效。');
  const ops=typeof n.content==='string'?JSON.parse(n.content):n.content;parseLines(ops);
  const allowed=['code','bold','italic','strike','underline','link','table','table-row','list','callout','callout-emoji','blockquote','header'];
  const content=ops.filter(op=>op&&Object.hasOwn(op,'insert')).map(op=>{
    let insert=op.insert;
    if(typeof insert!=='string')insert=imageName(insert?.image)?{image:insert.image}:typeof insert?.formula==='string'?{formula:insert.formula}:{unsupported:true};
    const attributes={};for(const key of allowed)if(['string','number','boolean'].includes(typeof op.attributes?.[key]))attributes[key]=op.attributes[key];
    return Object.keys(attributes).length?{insert,attributes}:{insert};
  });
  return {note:{uuid:n.uuid,createdAtMs:n.createdAtMs,content}};
}

export function createCosFeed({fetcher=fetch,parseXML=parseListing,now=Date.now}={}){
  let shards=new Map(),records=[],generation='',refreshedAt=0,pending=null,lastCheck=-Infinity;
  const bodies=new Map();let bodyBytes=0;
  async function read(url,max){
    let response;try{response=await fetcher(url,{mode:'cors',credentials:'omit',cache:'no-store',referrerPolicy:'no-referrer',signal:AbortSignal.timeout(20000)});}catch{throw Error('无法连接 COS，请检查网络和跨域设置。');}
    if(!response.ok)throw Error(response.status===403?'COS 尚未开放此目录的读取权限。':response.status===404?'闪念文件尚未同步完整，请稍后刷新。':'COS 读取失败，请稍后刷新。');
    if(Number(response.headers.get('content-length'))>max)throw Error('同步文件过大，停止读取。');
    if(!response.body){const text=await response.text();if(new TextEncoder().encode(text).length>max)throw Error('同步文件过大。');return text;}
    const stream=response.body.getReader(),chunks=[];let size=0;
    try{while(true){const {done,value}=await stream.read();if(done)break;size+=value.length;if(size>max){await stream.cancel();throw Error('同步文件过大，停止读取。');}chunks.push(value);}}finally{stream.releaseLock();}
    const all=new Uint8Array(size);let p=0;for(const c of chunks){all.set(c,p);p+=c.length;}return new TextDecoder().decode(all);
  }
  async function list(){
    const files=[];let marker='';
    for(let i=0;i<10;i++){
      const url=new URL(COS_ORIGIN+'/');url.searchParams.set('prefix',SHARDS);url.searchParams.set('max-keys','1000');url.searchParams.set('marker',marker);url.searchParams.set('_',String(now()));
      const page=parseXML(await read(url.href,4*1024*1024));files.push(...page.files);
      if(!page.truncated)return files;
      if(!page.marker||page.marker===marker)throw Error('COS 目录分页异常。');marker=page.marker;
    }
    throw Error('索引文件过多。');
  }
  async function refresh(){
    if(pending)return pending;
    if(generation&&now()-lastCheck<3000)return;
    pending=(async()=>{
      const files=(await list()).filter(f=>/^SnapNote_Sync\/index_shards\/[a-f0-9]+\.json$/i.test(f.key));
      const next=new Map(),latest=new Map();let i=0;
      await Promise.all(Array.from({length:Math.min(files.length,4)},async()=>{while(i<files.length){
        const file=files[i++],old=shards.get(file.key);if(file.size>4*1024*1024)throw Error('闪念索引分片过大。');
        let rows=old&&old.etag===file.etag?old.rows:null;
        if(!rows){const text=await read(COS_ORIGIN+'/'+file.key+'?v='+encodeURIComponent(file.etag),4*1024*1024);rows=text.startsWith('SNENC1')?[]:parseShard(JSON.parse(text));}
        next.set(file.key,{etag:file.etag,rows});
      }}));
      for(const {rows} of next.values())for(const r of rows){const p=latest.get(r.id);if(!p||r.modified>p.modified||r.modified===p.modified&&(r.deleted&&!p.deleted||r.deleted===p.deleted&&r.revision>p.revision))latest.set(r.id,r);}
      const nextRecords=[...latest.values()].filter(r=>!r.deleted).sort((a,b)=>b.modified-a.modified||a.id.localeCompare(b.id));
      const bytes=new TextEncoder().encode(JSON.stringify(nextRecords));const hash=await crypto.subtle.digest('SHA-256',bytes);
      generation=[...new Uint8Array(hash)].map(b=>b.toString(16).padStart(2,'0')).join('');
      const keep=new Set(nextRecords.map(r=>r.id+'__'+r.revision));for(const [key,v] of bodies)if(!keep.has(key)){bodyBytes-=v.size;bodies.delete(key);}
      shards=next;records=nextRecords;refreshedAt=now();lastCheck=now();
    })().finally(()=>{pending=null;});return pending;
  }
  async function note(entry){
    const key=entry.id+'__'+entry.revision;
    if(bodies.has(key)){const value=bodies.get(key);bodies.delete(key);bodies.set(key,value);return value.record;}
    const text=await read(COS_ORIGIN+'/'+PREFIX+'notes/'+key+'.json',2*1024*1024),record=publicNote(text,entry,'threads');
    const size=record?new TextEncoder().encode(JSON.stringify(record)).length:16;
    bodies.set(key,{record,size});bodyBytes+=size;
    while(bodies.size>100||bodyBytes>16*1024*1024){const first=bodies.keys().next().value;bodyBytes-=bodies.get(first).size;bodies.delete(first);}return record;
  }
  return async({reset=false,cursor=null}={})=>{
    if(reset||!generation)await refresh();let offset=0;
    if(!reset&&cursor){const match=/^([a-f0-9]{64}):(\d{1,8})$/.exec(cursor);if(!match||match[1]!==generation)throw Error('记录已更新，请刷新。');offset=Number(match[2]);if(offset>records.length)throw Error('分页位置无效。');}
    const start=offset,notes=[];
    while(offset<records.length&&notes.length<20&&offset-start<200){
      const batch=records.slice(offset,offset+Math.min(4,20-notes.length));const loaded=await Promise.all(batch.map(note));offset+=batch.length;notes.push(...loaded.filter(Boolean));
    }
    const media={};for(const r of notes)for(const op of r.note.content)if(op.insert?.image){const url=COS_ORIGIN+'/'+PREFIX+'resources/'+encodeURIComponent(op.insert.image);media[op.insert.image]={original:url,thumbnail:url};}
    return {notes,media,cursor:generation+':'+start,nextCursor:offset<records.length?generation+':'+offset:null,total:records.length,refreshedAt:new Date(refreshedAt).toISOString()};
  };
}
