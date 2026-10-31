import test from 'node:test';
import assert from 'node:assert/strict';
import {createCosFeed,parseShard,publicNote,COS_ORIGIN} from '../static/moments/cos-feed.mjs';
import {renderContent} from '../static/moments/render.mjs';
import {BLOG_TAG,THREADS_TAG} from '../static/moments/channels.mjs';

const id=i=>`00000000-0000-4000-8000-${String(i).padStart(12,'0')}`;
const rev='a'.repeat(32),rev2='b'.repeat(32);
const row=(i,deleted=0,version=rev)=>[id(i),deleted,1700000000000+i,{},version];
const record=(i,version=rev)=>({revisionId:version,tagUuids:[THREADS_TAG],note:{uuid:id(i),createdAtMs:1700000000000+i,content:[{insert:'正文\n'},{insert:{image:'photo.jpg'}}],deviceId:'not-for-rendering'}});

test('only threads are shown; blog wins if both channel labels are set',()=>{
  const entry=parseShard([row(1)])[0],raw=record(1);
  assert.ok(publicNote(JSON.stringify(raw),entry,'threads'));
  for(const tags of [[],[BLOG_TAG],[BLOG_TAG,THREADS_TAG]]){
    raw.tagUuids=tags;assert.equal(publicNote(JSON.stringify(raw),entry,'threads'),null);
  }
});
function fixture(count=45){
  let rows=Array.from({length:count},(_,i)=>row(i+1)),etag='v1',clock=10000,fail=false;
  const calls=[];
  const fetcher=async url=>{
    const u=new URL(url);calls.push(u);
    if(fail)return new Response('',{status:403});
    if(u.pathname==='/')return Response.json({files:[{key:'SnapNote_Sync/index_shards/01.json',etag,size:1000}],truncated:false});
    if(u.pathname.includes('/index_shards/'))return Response.json(rows);
    const match=/([0-9]{12})__([a-f0-9]+)\.json$/.exec(u.pathname);
    assert.ok(match);return Response.json(record(Number(match[1]),match[2]));
  };
  return {reader:createCosFeed({fetcher,parseXML:JSON.parse,now:()=>clock}),calls,change(r){rows=r;etag+='x';clock+=4000;},fail(){fail=true;clock+=4000;},tick(){clock+=4000;}};
}
test('first page requests only 20 bodies, following pages use cursor; repeated refresh reuses unchanged bodies',async()=>{
  const f=fixture(),first=await f.reader({reset:true});
  assert.equal(first.notes.length,20);assert.equal(first.notes[0].note.uuid,id(45));
  assert.equal(f.calls.filter(u=>u.pathname.includes('/notes/')).length,20);
  const second=await f.reader({cursor:first.nextCursor});assert.equal(second.notes.length,20);assert.equal(second.notes[0].note.uuid,id(25));
  const third=await f.reader({cursor:second.nextCursor});assert.equal(third.notes.length,5);assert.equal(third.nextCursor,null);
  f.tick();const before=f.calls.length;await f.reader({reset:true});assert.equal(f.calls.length-before,1);
  const back=await f.reader({cursor:first.cursor});assert.equal(back.notes[0].note.uuid,id(45));
  assert.equal(first.media['photo.jpg'].original,COS_ORIGIN+'/SnapNote_Sync/resources/photo.jpg');
});
test('refresh applies new, edited and deleted records and rejects old cursors',async()=>{
  const f=fixture(3),old=await f.reader({reset:true});
  f.change([row(1),row(2,1),row(3,0,rev2),row(4)]);
  const updated=await f.reader({reset:true});
  assert.deepEqual(updated.notes.map(r=>r.note.uuid),[id(4),id(3),id(1)]);
  await assert.rejects(()=>f.reader({cursor:old.cursor}),/记录已更新/);
  assert.equal(f.calls.filter(u=>u.pathname.includes('/notes/')).length,5);
  f.change([]);assert.equal((await f.reader({reset:true})).notes.length,0);
});
test('refresh failure is reported, without silently pretending to have new data',async()=>{
  const f=fixture(1);await f.reader({reset:true});f.fail();await assert.rejects(()=>f.reader({reset:true}),/读取权限/);
});
test('a legacy encrypted shard does not block other public records',async()=>{
  const fetcher=async url=>{
    const path=new URL(url).pathname;
    if(path==='/')return Response.json({files:['01','02'].map(key=>({key:`SnapNote_Sync/index_shards/${key}.json`,etag:key,size:100})),truncated:false});
    if(path.endsWith('/01.json'))return new Response('SNENC1legacy');
    if(path.endsWith('/02.json'))return Response.json([row(1)]);
    return Response.json(record(1));
  };
  const read=createCosFeed({fetcher,parseXML:JSON.parse});
  assert.equal((await read({reset:true})).notes.length,1);
});
test('schema validation, hidden records and render field filtering',()=>{
  assert.throws(()=>parseShard([['bad',0,1,{},rev]]));
  const entry=parseShard([row(1)])[0];
  assert.equal(publicNote('SNENC1test',entry),null);
  const raw=record(1);raw.note.isLocked=true;assert.equal(publicNote(JSON.stringify(raw),entry),null);
  raw.note.isLocked=false;raw.note.hasMarkDeleted=true;assert.equal(publicNote(JSON.stringify(raw),entry),null);
  raw.note.hasMarkDeleted=false;const clean=publicNote(JSON.stringify(raw),entry);assert.equal(clean.note.deviceId,undefined);
  assert.throws(()=>publicNote(JSON.stringify(record(2)),entry));
});
test('COS images are limited to this resource directory and HTML remains escaped',()=>{
  const content=[{insert:'<script>alert(1)</script>\n'},{insert:{image:'photo.jpg'}}];
  const asset=COS_ORIGIN+'/SnapNote_Sync/resources/photo.jpg';
  const html=renderContent(content,{'photo.jpg':{original:asset,thumbnail:asset}});
  assert.ok(html.includes('data-image="'+asset));assert.ok(html.includes('&lt;script&gt;'));
  for(const bad of ['https://evil.example/photo.jpg',COS_ORIGIN+'/profile.json',asset+'?unsafe=1']){
    assert.ok(!renderContent(content,{'photo.jpg':{original:bad,thumbnail:bad}}).includes('data-image='));
  }
});
