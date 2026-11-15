import test from 'node:test';
import assert from 'node:assert/strict';
import {supplementReviewMetadata} from './review-metadata.mjs';
import {convertReview} from './review.mjs';
import {REVIEW_TAG} from '../../static/moments/channels.mjs';
import {access} from 'node:fs/promises';
const slug='snapnote-a0f16e5f-d49f-4ae8-9ef9-9be58b620dba';

test('an independent poster keeps the first body image and caption at their original position',()=>{
  const image='b0ad01c284b7a1f5cc2374e54a7e4da3.png';
  const raw={revisionId:'a'.repeat(32),tagUuids:[REVIEW_TAG],note:{uuid:slug.slice(9),createdAtMs:1700000000000,content:[
    {insert:'Dead Poets Society\ndraft: false\nlabels: film\nrating: 4.8\ncover:\nBefore image.\n'},
    {insert:{image}},{insert:'\ncaption: Original caption\nAfter image.\n'},
    {insert:{image:'second.png'}},{insert:'\nLast paragraph.\n'}
  ]}};
  const original=JSON.stringify(raw);
  const article=convertReview(raw,{draft:false});
  const front=JSON.parse(article.source.split('\n\n')[0]);
  assert.equal(front.cover,'/library-covers/dead-poets-society.jpg');
  assert.equal(front.rating,4.8);
  assert.equal(article.images.length,2);
  assert.ok(article.images[0].endsWith('/'+image));
  assert.equal((article.source.match(/<figure>/g)||[]).length,2);
  const sequence=['<p>Before image.</p>','<figure>','Original caption','<p>After image.</p>','second.png','<p>Last paragraph.</p>'];
  let position=-1;
  for(const part of sequence){const next=article.source.indexOf(part,position+1);assert.ok(next>position,part);position=next;}
  assert.equal(JSON.stringify(raw),original,'source note must not be mutated');
  assert.equal(convertReview(raw,{draft:false}).source,article.source,'subsequent sync stays identical');
  raw.note.uuid='11111111-1111-4111-8111-111111111111';
  const normal=convertReview(raw,{draft:false});
  assert.ok(JSON.parse(normal.source.split('\n\n')[0]).cover.endsWith('/'+image));
  assert.equal((normal.source.match(/<figure>/g)||[]).length,1,'other reviews keep the existing fallback cover behavior');
});

test('official poster persists through sync without overriding a future custom cover',async()=>{
  const previous='https://snapnote-test-1316371247.cos.ap-shanghai.myqcloud.com/SnapNote_Sync/resources/b0ad01c284b7a1f5cc2374e54a7e4da3.png';
  const result=supplementReviewMetadata({slug,title:'Dead Poets Society',cover:previous});
  assert.equal(result.cover,'/library-covers/dead-poets-society.jpg');
  assert.ok(result.cover_source.includes('impawards.com/1989/'));
  await access(new URL('../../static'+result.cover,import.meta.url));
  assert.deepEqual(supplementReviewMetadata(result),result);
  assert.equal(supplementReviewMetadata({slug,title:'Another film',cover:previous}).cover,previous);
  const custom=previous.replace('b0ad01c284b7a1f5cc2374e54a7e4da3','future-cover');
  assert.equal(supplementReviewMetadata({slug,title:'Dead Poets Society',cover:custom}).cover,custom);
  const raw={revisionId:'a'.repeat(32),tagUuids:[REVIEW_TAG],note:{uuid:slug.slice(9),createdAtMs:1700000000000,content:[
    {insert:'Dead Poets Society\ndraft: false\nlabels: film\nrating: 4.8\ncover: '+previous+'\n\nMy review.\n'}
  ]}};
  const article=convertReview(raw,{draft:false});
  assert.equal(JSON.parse(article.source.split('\n\n')[0]).cover,result.cover);
  assert.deepEqual(article.images,[],'local cover must not be fetched as a remote COS image');
  assert.ok(article.source.endsWith('<p>My review.</p>\n'));
});

test('verified metadata is additive, repeatable and scoped to the same note and work',()=>{
  const input={slug,title:'Dead Poets Society',rating:4.8,summary:'My summary',cover:'cover.png',creators:{}};
  const result=supplementReviewMetadata(input);
  assert.equal(result.release_year,1989);
  assert.equal(result.creators['1_director'],'Peter Weir');
  assert.equal(result.creators['4_music'],'Maurice Jarre');
  assert.equal(Object.keys(result.creators).length,5);
  for(const key of ['rating','summary','cover'])assert.equal(result[key],input[key]);
  assert.deepEqual(supplementReviewMetadata(result),result);
  assert.deepEqual(input.creators,{});
  for(const other of [{...input,title:'Another film'},{...input,slug:'another-note'}])assert.equal(supplementReviewMetadata(other),other);
});

test('explicit SnapNote credits take precedence, including equivalent role names',()=>{
  const input={slug,title:'死亡诗社',release_year:2000,creators:{'1_writer':'Custom writer','2_composer':'Custom music'}};
  const result=supplementReviewMetadata(input);
  assert.equal(result.title,'Dead Poets Society');
  assert.equal(result.release_year,2000);
  assert.equal(result.creators['1_writer'],'Custom writer');
  assert.equal(result.creators['2_composer'],'Custom music');
  assert.equal(Object.keys(result.creators).length,5);
});

test('sync keeps cover, rating, summary and review body while adding credits',()=>{
  const note={uuid:slug.slice(9),createdAtMs:1700000000000,content:[
    {insert:'Dead Poets Society\ndraft: false\nlabels: film\nrating: 4.8\nsummary: My summary\ncover:\n'},
    {insert:{image:'cover.png'}},{insert:'\nMy own review.\n'}
  ]};
  const raw={revisionId:'a'.repeat(32),tagUuids:[REVIEW_TAG],note};
  const result=convertReview(raw,{draft:false,skipDrafts:true});
  const front=JSON.parse(result.source.split('\n\n')[0]);
  assert.equal(front.rating,4.8);assert.equal(front.summary,'My summary');
  assert.equal(front.creators['5_cinematographer'],'John Seale');
  assert.deepEqual(front.library_tags,['film']);
  assert.ok(front.cover.endsWith('/cover.png'));
  assert.equal(result.source.slice(result.source.indexOf('\n\n')),'\n\n<p>My own review.</p>\n');
  note.content=[{insert:'Dead Poets Society\ndraft: true\n'}];
  assert.equal(convertReview(raw,{draft:false,skipDrafts:true}),null);
});
