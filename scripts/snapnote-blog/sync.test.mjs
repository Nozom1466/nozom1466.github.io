import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,writeFile,readFile,readdir,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {collectArticles,writeArticles,needsDeployment} from './sync.mjs';
import {BLOG_TAG,THREADS_TAG} from '../../static/moments/channels.mjs';
const id=i=>`00000000-0000-4000-8000-${String(i).padStart(12,'0')}`,rev='a'.repeat(32);
function fixture(){
  const rows=[1,2,3].map(i=>[id(i),0,1700000000000+i,{},rev]);
  const records=Object.fromEntries([1,2,3].map(i=>[id(i),{revisionId:rev,tagUuids:[i===1?BLOG_TAG:THREADS_TAG],note:{uuid:id(i),createdAtMs:1700000000000,content:[{insert:'标题\n正文\n'}]}}]));
  const fetcher=async url=>{
    const p=new URL(url).pathname;
    if(p==='/')return new Response('<ListBucketResult><Contents><Key>SnapNote_Sync/index_shards/01.json</Key></Contents><IsTruncated>false</IsTruncated></ListBucketResult>');
    if(p.includes('/index_shards/'))return Response.json(rows);
    const key=p.split('/').pop().split('__')[0];return Response.json(records[key]);
  };
  return {rows,records,fetcher};
}
test('only blog is converted; edit changes digest, threads edits do not',async()=>{
  const f=fixture(),a=await collectArticles(f);assert.equal(a.articles.length,1);
  f.records[id(2)].note.content=[{insert:'threads 修改\n'}];assert.equal((await collectArticles(f)).digest,a.digest);
  f.records[id(1)].note.content=[{insert:'改标题\n改正文\n'}];const b=await collectArticles(f);assert.notEqual(b.digest,a.digest);
  assert.equal(b.articles[0].filename,a.articles[0].filename);
  assert.equal(needsDeployment(b,{digest:b.digest,commit:'c'},'c'),false);
  assert.equal(needsDeployment(b,{digest:a.digest,commit:'c'},'c'),true);
  assert.equal(needsDeployment(b,{digest:b.digest,commit:'old'},'c'),true);
});
test('delete, archive or remove blog label retracts the generated article',async()=>{
  for(const action of ['delete','archive','untag']){
    const f=fixture();if(action==='delete')f.rows[0][1]=1;
    if(action==='archive')f.records[id(1)].note.isArchived=true;
    if(action==='untag')f.records[id(1)].tagUuids=[];
    assert.equal((await collectArticles(f)).articles.length,0);
  }
});
test('read failure, mismatched revisions and incomplete listing abort',async()=>{
  await assert.rejects(()=>collectArticles({fetcher:async()=>new Response('',{status:403})}));
  await assert.rejects(()=>collectArticles({fetcher:async()=>new Response('<ListBucketResult>')}));
  const f=fixture();f.records[id(1)].revisionId='b'.repeat(32);await assert.rejects(()=>collectArticles(f));
});
test('missing image aborts instead of publishing a broken article',async()=>{
  const f=fixture();f.records[id(1)].note.content.push({insert:{image:'pic.jpg'}},{insert:'\n'});
  const original=f.fetcher;f.fetcher=url=>new URL(url).pathname.includes('/resources/')?Promise.resolve(new Response('',{status:404})):original(url);
  await assert.rejects(()=>collectArticles(f),/配图/);
});
test('writes/removes only owned generated files and never overwrites a hand-written collision',async()=>{
  const dir=await mkdtemp(path.join(tmpdir(),'snapnote-sync-test-'));
  try{
    const a=await collectArticles(fixture());await writeFile(path.join(dir,'handwritten.md'),'keep');
    await writeArticles(a.articles,dir);assert.ok((await readdir(dir)).includes(a.articles[0].filename));
    await writeArticles([],dir);assert.equal(await readFile(path.join(dir,'handwritten.md'),'utf8'),'keep');
    await writeFile(path.join(dir,a.articles[0].filename),'user-owned');
    await assert.rejects(()=>writeArticles(a.articles,dir),/不是同步文章/);
    assert.equal(await readFile(path.join(dir,a.articles[0].filename),'utf8'),'user-owned');
  }finally{await rm(dir,{recursive:true});}
});
