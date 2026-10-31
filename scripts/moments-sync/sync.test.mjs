import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,readFile,writeFile,readdir,rm} from 'node:fs/promises';
import path from 'node:path';
import {tmpdir} from 'node:os';
import sharp from 'sharp';
import {syncFeed,normalize,BUCKET,REGION} from './sync.mjs';
import {createFeed} from '../../static/moments/feed.mjs';
import {renderContent} from '../../static/moments/render.mjs';

function fakeCOS(){
  const objects=new Map();let gets=0,lists=0;
  const client={
    getBucket(p,cb){
      assert.equal(p.Bucket,BUCKET);assert.equal(p.Region,REGION);lists++;
      const all=[...objects.entries()].filter(([key])=>key.startsWith(p.Prefix)).map(([Key,o])=>({Key,...o.meta}));
      const start=Number(p.Marker||0),end=Math.min(start+p.MaxKeys,all.length);
      cb(null,{Contents:all.slice(start,end),IsTruncated:end<all.length,NextMarker:String(end)});
    },
    getObject(p,cb){gets++;const o=objects.get(p.Key);if(!o)return cb({statusCode:404});if(o.meta.ETag!==p.IfMatch)return cb({statusCode:412});cb(null,{Body:o.body});}
  };
  return {client,objects,get gets(){return gets;},get lists(){return lists;},reset(){gets=lists=0;},
    put(key,body,revision='1'){body=Buffer.isBuffer(body)?body:Buffer.from(typeof body==='string'?body:JSON.stringify(body));objects.set(key,{body,meta:{ETag:revision,Size:body.length,LastModified:'2026-10-30T20:00:00Z'}});},
    note(i,extra={},revision='1'){this.put(`SnapNote_Sync/notes/${i}__one.json`,{note:{uuid:String(i),createdAtMs:100000+i,createInfoJson:'PRIVATE DEVICE',ocr:'PRIVATE OCR',content:JSON.stringify([{insert:'测试正文 '+i+'\n'}]),...extra}},revision);}
  };
}
async function context(t){const root=await mkdtemp(path.join(tmpdir(),'moments-sync-test-'));t.after(()=>rm(root,{recursive:true,force:true}));let run=0;return{root,cacheDir:path.join(root,'cache'),output:()=>path.join(root,'export-'+run++)};}

test('20,000 records paginate, unchanged checks download no bodies, failure retries compare live generation',async t=>{
  const c=await context(t),f=fakeCOS();for(let i=0;i<20000;i++)f.note(i);
  const outputDir=c.output();const first=await syncFeed({client:f.client,cacheDir:c.cacheDir,outputDir});
  assert.equal(first.total,20000);assert.equal(first.noteDownloads,20000);assert.equal(f.lists,20);
  const manifest=JSON.parse(await readFile(path.join(outputDir,'manifest.json'),'utf8'));assert.equal(manifest.pages,1000);
  const page=JSON.parse(await readFile(path.join(outputDir,manifest.version,'page-0.json'),'utf8'));assert.equal(page.notes.length,20);assert.equal(page.notes[0].note.uuid,'19999');assert.ok(!JSON.stringify(page).includes('PRIVATE'));
  f.reset();const second=await syncFeed({client:f.client,cacheDir:c.cacheDir,outputDir:c.output(),publishedVersion:first.version});
  assert.equal(second.shouldDeploy,false);assert.equal(second.cacheChanged,false);assert.equal(f.gets,0);
  f.note(20000);f.reset();const changed=await syncFeed({client:f.client,cacheDir:c.cacheDir,outputDir:c.output(),publishedVersion:first.version});assert.equal(changed.shouldDeploy,true);assert.equal(f.gets,1);
  f.reset();const retry=await syncFeed({client:f.client,cacheDir:c.cacheDir,outputDir:c.output(),publishedVersion:first.version});assert.equal(retry.shouldDeploy,true,'cached export is not proof of deployment');assert.equal(f.gets,0);
  const forced=await syncFeed({client:f.client,cacheDir:c.cacheDir,outputDir:c.output(),publishedVersion:changed.version,force:true});assert.equal(forced.shouldDeploy,true);
});

test('public fields only; locked, archived, deleted, encrypted excluded; deletion removes cached media',async t=>{
  const c=await context(t),f=fakeCOS();
  const image=await sharp({create:{width:1200,height:800,channels:3,background:'#8a705a'}}).jpeg().toBuffer();
  f.put('SnapNote_Sync/resources/photo.jpg',image);
  f.note(1,{content:JSON.stringify([{insert:'公开内容\n',attributes:{bold:true,privateInfo:'NO'}},{insert:{image:'photo.jpg',device:'NO'}}])});
  f.note(2,{isLocked:true});f.note(3,{isArchived:true});f.note(4,{hasMarkDeleted:true});f.put('SnapNote_Sync/notes/5__one.json','SNENC1private');
  const out=c.output();const first=await syncFeed({client:f.client,cacheDir:c.cacheDir,outputDir:out});assert.equal(first.total,1);assert.equal(first.skipped,4);
  const cache=await readFile(path.join(c.cacheDir,'state.json'),'utf8');assert.ok(!cache.includes('PRIVATE'));assert.ok(!cache.includes('privateInfo'));assert.ok(!cache.includes('测试正文 2'));
  const page=JSON.parse(await readFile(path.join(out,first.version,'page-0.json'),'utf8')),asset=page.media['photo.jpg'];
  assert.deepEqual(await readFile(path.join(out,asset.original)),image);const info=await sharp(await readFile(path.join(out,asset.thumbnail))).metadata();assert.equal(info.width,384);assert.equal(info.format,'webp');
  f.reset();await syncFeed({client:f.client,cacheDir:c.cacheDir,outputDir:c.output(),publishedVersion:first.version});assert.equal(f.gets,0);
  f.note(1,{hasMarkDeleted:true},'2');const removed=await syncFeed({client:f.client,cacheDir:c.cacheDir,outputDir:c.output(),publishedVersion:first.version});assert.equal(removed.total,0);assert.equal(removed.shouldDeploy,true);assert.deepEqual(await readdir(path.join(c.cacheDir,'media')),[]);
  assert.throws(()=>normalize(Buffer.from('{broken')),/JSON/);
});

test('missing or invalid image aborts; partial sync cannot overwrite cache state or prior output',async t=>{
  const c=await context(t),f=fakeCOS();f.note(1);const out=c.output();await syncFeed({client:f.client,cacheDir:c.cacheDir,outputDir:out});
  const previous=await readFile(path.join(c.cacheDir,'state.json'),'utf8'),published=await readFile(path.join(out,'manifest.json'),'utf8');
  f.note(2,{content:JSON.stringify([{insert:{image:'missing.jpg'}}])});
  await assert.rejects(syncFeed({client:f.client,cacheDir:c.cacheDir,outputDir:c.output()}),/尚未同步/);
  assert.equal(await readFile(path.join(c.cacheDir,'state.json'),'utf8'),previous);assert.equal(await readFile(path.join(out,'manifest.json'),'utf8'),published);
  await assert.rejects(syncFeed({client:f.client,cacheDir:c.cacheDir,outputDir:out}),/已有发布/);
});

test('static feed fetches one page, maps media safely, refresh accepts a newer generation',async()=>{
  let version='a'.repeat(32),calls=[];const asset='b'.repeat(64);
  const fetcher=async url=>{
    calls.push(url);const data=url.includes('manifest.json')?{schema:1,version,pages:2,pageSize:20,total:21,refreshedAt:new Date().toISOString()}:{version,page:url.endsWith('page-1.json')?1:0,notes:[{note:{uuid:'1',createdAtMs:1,content:[{insert:{image:'photo.jpg'}}]}}],media:{'photo.jpg':{original:`media/${asset}.jpg`,thumbnail:`media/${asset}.thumb.webp`},bad:{original:'https://evil.test/x',thumbnail:'../../secret'}}};
    return new Response(JSON.stringify(data),{headers:{'content-type':'application/json'}});
  };
  const read=createFeed({fetcher});const first=await read({reset:true});assert.equal(calls.length,2);assert.ok(!first.media.bad);
  const html=renderContent(first.notes[0].note.content,first.media);assert.ok(html.includes('/moments-feed/media/'));assert.ok(!html.includes('/api/'));assert.ok(html.includes('loading="lazy"'));
  await read({index:1,cursor:first.nextCursor});assert.equal(calls.length,3);
  version='c'.repeat(32);const fresh=await read({reset:true,cursor:first.cursor});assert.ok(fresh.cursor.startsWith(version));
  await assert.rejects(read({index:1,cursor:first.nextCursor}),/已更新/);
  const blocked=renderContent([{insert:{image:'photo.jpg'}}],{'photo.jpg':{original:'//evil.test/a.jpg',thumbnail:'javascript:alert(1)'}});assert.ok(!blocked.includes('<img'));
});

test('live preview retains its existing local read endpoint',async()=>{
  let called;const read=createFeed({mode:'live',fetcher:async url=>{called=url;return new Response(JSON.stringify({notes:[]}),{headers:{'content-type':'application/json'}});}});
  await read({reset:true});assert.equal(called,'/api/moments/notes?limit=20&refresh=1');
});
