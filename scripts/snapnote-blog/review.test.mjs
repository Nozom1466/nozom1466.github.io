import test from 'node:test';
import assert from 'node:assert/strict';
import {convertReview} from './review.mjs';
import {REVIEW_TAG,BLOG_TAG,channelOf} from '../../static/moments/channels.mjs';
const record=()=>({revisionId:'a'.repeat(32),tagUuids:[REVIEW_TAG],note:{uuid:'26546fe9-8f91-4260-8ba1-5e3e9c9d1b95',createdAtMs:1700000000000,content:[{insert:'经营未来\ndraft: false\nrating: 4.6\nauthor: 李明博\ncover: \n'},{insert:{image:'cover.png'}},{insert:'\nsummary: 从穷小子到总统\n\n评价正文。\n'}]}});
const front=a=>JSON.parse(a.source.split('\n\n')[0]);
test('native review channel and cover marker with pasted image map to library fields',()=>{
  const r=record(),a=convertReview(r,{draft:false,skipDrafts:true}),f=front(a);
  assert.equal(channelOf(r),'review');assert.equal(a.section,'library');assert.equal(f.title,'신화는 없다');assert.equal(a.title,f.title);
  assert.equal(f.rating,4.6);assert.equal(f.creators['1_author'],'이명박');assert.equal(f.author,'이명박');assert.equal(f.summary,'从穷小子到总统');
  assert.deepEqual(f.library_tags,['book']);assert.equal(f.draft,false);assert.ok(f.cover.endsWith('/cover.png'));
  assert.ok(a.source.includes('<p>评价正文。</p>'));assert.ok(!a.source.includes('<figure>'));assert.equal(a.images.length,1);
  assert.ok(!a.source.includes('rating:'));assert.ok(!a.source.includes('summary:'));
});
test('unknown or repurposed review titles are not forced to the old work identity',()=>{
  const r=record();r.note.uuid='11111111-1111-4111-8111-111111111111';
  assert.equal(front(convertReview(r)).title,'经营未来');
  r.note.uuid=record().note.uuid;r.note.content[0].insert=r.note.content[0].insert.replace('经营未来','另一本书');
  assert.equal(front(convertReview(r)).title,'另一本书');
  assert.equal(front(convertReview(r)).creators['1_author'],'李明博');
});
test('film creators, labels, date, status and body captions are retained',()=>{
  const r=record();r.note.content[0].insert='电影名\nlabels: film\nrating: 0\ndirector: 导演\nscreenplay: 编剧\ncomposer: 配乐\ndate: 2026-10-01\nstatus: Completed\ncover:\n';
  r.note.content.push({insert:{image:'scene.png'}},{insert:'\ncaption: 剧照\n'});
  const a=convertReview(r,{draft:false}),f=front(a);
  assert.deepEqual(f.library_tags,['film']);assert.equal(f.creators['1_director'],'导演');assert.equal(f.creators['3_music'],'配乐');
  assert.equal(f.rating,0);assert.equal(f.date,'2026-10-01');assert.equal(f.status,'Completed');assert.ok(a.source.includes('<figcaption'));assert.equal(a.images.length,2);
});
test('full-width colon in labels is accepted',()=>{
  const r=record();r.note.content[0].insert=r.note.content[0].insert.replace('draft: false','labels： book, history\ndraft: false');
  assert.deepEqual(front(convertReview(r)).library_tags,['book','history']);
});
test('review drafts skip incomplete content; archived notes cannot publish',()=>{
  const r=record();r.note.content=[{insert:'待完成\ndraft: true\n'}];assert.equal(convertReview(r,{draft:false,skipDrafts:true}),null);
  r.note.isArchived=true;assert.throws(()=>convertReview(r));
});
test('metadata cannot be mistaken for title, and invalid fields fail safely',()=>{
  const r=record();r.note.content[0].insert=r.note.content[0].insert.replace('经营未来\n','');assert.throws(()=>convertReview(r),/作品名称/);
  for(const [key,value] of [['rating','5.1'],['rating','oops'],['date','2026-02-30'],['status','unknown'],['draft','yes']]){
    const r=record();r.note.content[0].insert=`经营未来\n${key}: ${value}\ncover:\n`;assert.throws(()=>convertReview(r));
  }
  const blog=record();blog.tagUuids=[BLOG_TAG];assert.throws(()=>convertReview(blog));
});
test('cover links are limited to known COS origins; fallback first image is removed from body',()=>{
  const r=record();r.note.content[0].insert='经营未来\ncover: https://127.0.0.1/x.png\n';assert.throws(()=>convertReview(r),/COS/);
  r.note.content[0].insert='经营未来\n';const a=convertReview(r);assert.ok(front(a).cover.endsWith('/cover.png'));assert.ok(!a.source.includes('<figure>'));
});
