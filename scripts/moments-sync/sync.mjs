import {createHash} from 'node:crypto';
import {mkdir,readFile,writeFile,copyFile,readdir,unlink,stat} from 'node:fs/promises';
import path from 'node:path';
import sharp from 'sharp';
import {parseLines} from '../../static/moments/render.mjs';

export const BUCKET='snapnote-test-1316371247', REGION='ap-shanghai';
const NOTES='SnapNote_Sync/notes/', RESOURCES='SnapNote_Sync/resources/';
const MAX_NOTE=2*1024*1024, MAX_IMAGE=25*1024*1024, MAX_EXPORT=650*1024*1024;
const imageName=name=>typeof name==='string'&&/^[\w.-]+\.(jpe?g|png|webp|gif)$/i.test(name)&&!name.includes('..');
const hash=value=>createHash('sha256').update(value).digest('hex');
const signature=file=>hash(file.Key+'|'+file.ETag);
const fail=message=>Object.assign(new Error(message),{safeMessage:message});
const call=(client,method,params)=>new Promise((resolve,reject)=>client[method]({Bucket:BUCKET,Region:REGION,...params},(error,data)=>error?reject(error):resolve(data)));
async function list(client,prefix){
  const files=[];let marker='';
  for(let i=0;i<200;i++){
    const data=await call(client,'getBucket',{Prefix:prefix,Marker:marker,MaxKeys:1000});files.push(...(data.Contents||[]));
    if(String(data.IsTruncated)!=='true')return files;
    if(!data.NextMarker||data.NextMarker===marker)throw fail('COS 目录分页异常，停止发布。');
    marker=data.NextMarker;
  }
  throw fail('同步空间超过 20 万个文件，停止发布。');
}
async function download(client,file,limit){
  if(Number(file.Size)>limit)throw fail('文件超出同步大小限制，停止发布。');
  const data=await call(client,'getObject',{Key:file.Key,IfMatch:file.ETag,Range:`bytes=0-${limit}`});
  const body=Buffer.from(data.Body);
  if(body.length>limit)throw fail('下载文件超出同步大小限制，停止发布。');
  return body;
}
// Only rendered fields are retained. Device metadata, OCR and private hidden
// note contents are never written into the build cache or published JSON.
export function normalize(body){
  if(body.subarray(0,6).toString()==='SNENC1')return null;
  let raw;try{raw=JSON.parse(body.toString('utf8'));}catch{throw fail('笔记 JSON 不完整，停止发布，等待下次同步。');}
  const n=raw.note;
  if(!n||typeof n.uuid!=='string'||!Number.isFinite(n.createdAtMs)||!Number.isFinite(new Date(n.createdAtMs).getTime()))return null;
  if(n.hasMarkDeleted||n.isLocked||n.isArchived||raw.stateVector?.deleted)return null;
  let ops;try{ops=typeof n.content==='string'?JSON.parse(n.content):n.content;parseLines(ops);}catch{throw fail('笔记正文格式异常，停止发布。');}
  const allowed=['code','bold','italic','strike','underline','link','table','table-row','list','callout','callout-emoji','blockquote','header'];
  const content=[];
  for(const op of ops){
    if(!op||!Object.hasOwn(op,'insert'))continue;
    let insert=op.insert;
    if(typeof insert!=='string'){
      if(imageName(insert?.image))insert={image:insert.image};
      else if(typeof insert?.formula==='string')insert={formula:insert.formula};
      else insert={unsupported:true};
    }
    const attributes={};for(const key of allowed){const value=op.attributes?.[key];if(['string','boolean','number'].includes(typeof value))attributes[key]=value;}
    content.push(Object.keys(attributes).length?{insert,attributes}:{insert});
  }
  return {note:{uuid:n.uuid,createdAtMs:n.createdAtMs,content}};
}
async function parallel(items,fn){let i=0;await Promise.all(Array.from({length:Math.min(4,items.length)},async()=>{while(i<items.length){const index=i++;await fn(items[index],index);}}));}
async function exists(file){try{return (await stat(file)).isFile();}catch{return false;}}

export async function syncFeed({client,cacheDir,outputDir,publishedVersion=null,now=new Date(),force=false}){
  cacheDir=path.resolve(cacheDir);outputDir=path.resolve(outputDir);
  if(cacheDir===outputDir||!path.basename(cacheDir)||!path.basename(outputDir))throw fail('缓存与输出目录配置无效。');
  // The output is a new build staging directory, never a live site directory.
  if(await exists(path.join(outputDir,'manifest.json')))throw fail('输出目录已有发布数据，请使用新的构建目录。');
  await mkdir(path.join(cacheDir,'media'),{recursive:true});
  let previous={version:1,notes:[]};
  try{const saved=JSON.parse(await readFile(path.join(cacheDir,'state.json'),'utf8'));if(saved.version===1&&Array.isArray(saved.notes))previous=saved;}catch{}
  const oldNotes=new Map(previous.notes.map(x=>[x.signature,x.record]));
  const files=await list(client,NOTES),latest=new Map();
  for(const file of files){
    if(!/^SnapNote_Sync\/notes\/[^/]+\.json$/.test(file.Key))continue;
    const id=file.Key.slice(NOTES.length).split('__')[0].replace(/\.json$/,'');const prior=latest.get(id);
    if(!prior||Date.parse(file.LastModified)>Date.parse(prior.LastModified)||(file.LastModified===prior.LastModified&&file.Key>prior.Key))latest.set(id,file);
  }
  let noteDownloads=0,mediaDownloads=0,skipped=0;
  const selected=[...latest.values()].sort((a,b)=>a.Key.localeCompare(b.Key)),entries=new Array(selected.length);
  await parallel(selected,async(file,i)=>{
    const key=signature(file);let record;
    if(oldNotes.has(key))record=oldNotes.get(key);
    else {record=normalize(await download(client,file,MAX_NOTE));noteDownloads++;}
    if(!record)skipped++;
    entries[i]={signature:key,record};
  });
  const records=entries.map(x=>x.record).filter(Boolean).sort((a,b)=>b.note.createdAtMs-a.note.createdAtMs||a.note.uuid.localeCompare(b.note.uuid));
  const names=[...new Set(records.flatMap(r=>r.note.content.filter(op=>op.insert?.image).map(op=>op.insert.image)))].sort();
  const resourceFiles=names.length?await list(client,RESOURCES):[];
  const resources=new Map(resourceFiles.map(file=>[file.Key.slice(RESOURCES.length),file]));
  const assets={},keep=new Set();let exportBytes=Buffer.byteLength(JSON.stringify(records));
  await parallel(names,async name=>{
    const file=resources.get(name);if(!file)throw fail('正文引用的图片尚未同步完成，停止发布。');
    if(Number(file.Size)>MAX_IMAGE)throw fail('图片超过 25MB，请压缩后重试。');
    const key=signature(file),original=key+path.extname(name).toLowerCase(),thumbnail=key+'.thumb.webp';
    const originalPath=path.join(cacheDir,'media',original),thumbPath=path.join(cacheDir,'media',thumbnail);
    if(!await exists(originalPath)){await writeFile(originalPath,await download(client,file,MAX_IMAGE));mediaDownloads++;}
    if(!await exists(thumbPath)){
      const bytes=await sharp(await readFile(originalPath),{limitInputPixels:80000000}).rotate().resize({width:384,height:384,fit:'inside',withoutEnlargement:true}).webp({quality:78}).toBuffer();
      await writeFile(thumbPath,bytes);
    }
    exportBytes+=(await stat(originalPath)).size+(await stat(thumbPath)).size;
    if(exportBytes>MAX_EXPORT)throw fail('此刻公开副本超过 650MB，请调整附件托管方案后再发布。');
    keep.add(original);keep.add(thumbnail);assets[name]={original:'media/'+original,thumbnail:'media/'+thumbnail};
  });
  // Include renderer schema/version in content identity. No clock value enters it.
  const version=hash(JSON.stringify({schema:1,records,assets:Object.fromEntries(Object.entries(assets).sort())})).slice(0,32);
  const shouldDeploy=force||publishedVersion!==version;
  if(shouldDeploy){
    await mkdir(path.join(outputDir,version),{recursive:true});await mkdir(path.join(outputDir,'media'),{recursive:true});
    await parallel([...keep],name=>copyFile(path.join(cacheDir,'media',name),path.join(outputDir,'media',name)));
    const pages=Math.max(1,Math.ceil(records.length/20)),refreshedAt=now.toISOString();
    for(let page=0;page<pages;page++){
      const notes=records.slice(page*20,(page+1)*20),media={};
      for(const r of notes)for(const op of r.note.content)if(op.insert?.image)media[op.insert.image]=assets[op.insert.image];
      await writeFile(path.join(outputDir,version,`page-${page}.json`),JSON.stringify({version,page,notes,media,refreshedAt}));
    }
    await writeFile(path.join(outputDir,'manifest.json'),JSON.stringify({schema:1,version,total:records.length,pages,pageSize:20,refreshedAt}));
  }
  // Cache only material already eligible for public display. Remove obsolete
  // media by exact filename; never recursively delete user folders.
  for(const name of await readdir(path.join(cacheDir,'media'))){
    if(/^[a-f0-9]{64}(\.thumb)?\.(jpe?g|png|webp|gif)$/.test(name)&&!keep.has(name))await unlink(path.join(cacheDir,'media',name));
  }
  const state={version:1,notes:entries};const cacheChanged=JSON.stringify(state)!==JSON.stringify(previous)||mediaDownloads>0;
  await writeFile(path.join(cacheDir,'state.json'),JSON.stringify(state));
  return {version,shouldDeploy,cacheChanged,total:records.length,skipped,noteDownloads,mediaDownloads,exportBytes};
}
