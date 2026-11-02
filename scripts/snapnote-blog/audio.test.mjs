import test from 'node:test';
import assert from 'node:assert/strict';
import {convertReview} from './review.mjs';
import {convertArticle} from './convert.mjs';
import {readFileSync} from 'node:fs';
import {REVIEW_TAG,BLOG_TAG} from '../../static/moments/channels.mjs';
import {resourceURL} from './audio.mjs';

const music=(source={audio:'song.mp3'}, name='曲名')=>[
  {insert:`audio: ${name}\nartist: 作曲者\n`}, {insert:source},
  {insert:'\naudio_cover:\n'}, {insert:{image:'album.jpg'}}, {insert:'\n'}
];
const record=(body=music())=>({tagUuids:[REVIEW_TAG],revisionId:'a'.repeat(32),note:{
  uuid:'26546fe9-8f91-4260-8ba1-5e3e9c9d1b95',createdAtMs:1700000000000,
  content:[{insert:'电影\ndraft: false\nlabels: film\ncover:\n'}, {insert:{image:'film.jpg'}},
    {insert:'\n正文\n'},...body,{insert:'后续正文\n'}]
}});

test('review music renders existing player and consumes its cover without changing review cover',()=>{
  const a=convertReview(record(),{draft:false});
  const front=JSON.parse(a.source.split('\n\n')[0]);
  assert.ok(front.cover.endsWith('/film.jpg'));
  assert.ok(a.source.includes('<music-placeholder'));
  assert.ok(a.source.includes('data-name="曲名"'));
  assert.ok(a.source.includes('data-artist="作曲者"'));
  assert.ok(a.source.includes('data-cover="'+resourceURL('album.jpg','image')+'"'));
  assert.deepEqual(a.audios,[resourceURL('song.mp3','audio')]);
  assert.equal(a.images.length,2);assert.ok(!a.source.includes('<figure>'));
  assert.ok(!a.source.includes('audio_cover:'));assert.ok(!front.summary.includes('曲名'));
  assert.ok(a.source.includes('<p>后续正文</p>'));
});

test('multiple tracks, file resource objects, blank lines and optional artist/cover',()=>{
  const body=[...music({file:{path:'song.m4a'}},'第一首'),
    {insert:'AUDIO：第二首\n\n'}, {insert:{audio:'second.ogg'}}, {insert:'\n'}];
  const a=convertReview(record(body));
  assert.equal(a.audios.length,2);assert.equal((a.source.match(/<music-placeholder/g)||[]).length,2);
  assert.equal(a.images.length,2);
});

test('known COS direct audio and cover links work; note markup stays escaped',()=>{
  const link=resourceURL('song.mp3','audio'),cover=resourceURL('album.jpg','image');
  const a=convertReview(record([{insert:`audio: <b>曲名</b> {{< bad >}}\nartist: " onmouseover="x\n${link}\naudio_cover: ${cover}\n`}]));
  assert.ok(a.source.includes('&lt;b&gt;'));assert.ok(!a.source.includes('{{<'));
  assert.ok(a.source.includes('data-artist="&quot; onmouseover=&quot;x"'));
  assert.deepEqual(a.audios,[link]);
});

test('reject incomplete blocks, unsafe sources, wrong types and ambiguous attachments',()=>{
  for(const value of ['../song.mp3','javascript:alert(1)','https://example.com/song.mp3','song.mp4','song.html']){
    assert.throws(()=>convertReview(record(music({audio:value}))));
  }
  for(const body of [[{insert:'audio:\n'}],[{insert:'audio: 曲名\nartist: 作者\n'}],
    [{insert:'audio: 曲名\nsong.mp3\naudio_cover:\n'}],music({file:{name:'song.mp3'}})]){
    assert.throws(()=>convertReview(record(body)));
  }
});

test('album artwork cannot accidentally become the review cover',()=>{
  const r=record();r.note.content=[{insert:'电影\n'},...music()];
  assert.throws(()=>convertReview(r),/缺少封面/);
  r.note.content.push({insert:{image:'film.jpg'}},{insert:'\n'});
  assert.ok(JSON.parse(convertReview(r).source.split('\n\n')[0]).cover.endsWith('/film.jpg'));
});

test('draft skips unfinished music and music creator field remains separate',()=>{
  const r=record([{insert:'audio: 未完成\n'}]);
  r.note.content[0].insert='电影\ndraft: true\n';
  assert.equal(convertReview(r,{draft:false,skipDrafts:true}),null);
  const ready=record();ready.note.content[0].insert='电影\nmusic: 配乐作者\ncover:\n';
  assert.equal(JSON.parse(convertReview(ready).source.split('\n\n')[0]).creators['1_music'],'配乐作者');
});

test('blog renders multiple tracks using the same blocks as review and retains captions',()=>{
  const r=record();r.tagUuids=[BLOG_TAG];
  r.note.content=[{insert:'音乐文章\nlabels: 电影, 音乐\nsummary: 原声推荐\ndraft: false\n正文\n'},
    ...music(),...music({file:{path:'second.m4a'}},'第二首'),
    {insert:{image:'scene.jpg'}},{insert:'\ncaption: 剧照\n'}];
  const a=convertArticle(r,{draft:false});
  assert.equal(a.audios.length,2);assert.equal(a.images.length,3);
  assert.equal((a.source.match(/<music-placeholder/g)||[]).length,2);
  assert.ok(a.source.includes('<figcaption'));assert.ok(a.source.includes('剧照'));
  const f=JSON.parse(a.source.split('\n\n')[0]);assert.equal(f.summary,'原声推荐');assert.equal(f.draft,false);
  assert.deepEqual(f.tags,['电影','音乐']);assert.equal(f.cover,undefined);
  r.note.content=[{insert:'未完成\ndraft: true\naudio: 待填写\n'}];
  assert.equal(convertArticle(r,{draft:false,skipDrafts:true}),null);
});

test('existing Interstellar audio and artwork convert unchanged in a blog',()=>{
  const text=readFileSync(new URL('../../content/library/Interstellar.md',import.meta.url),'utf8');
  const shortcode=text.match(/\{\{< music ([\s\S]*?) >\}\}/)[1];
  const d=Object.fromEntries([...shortcode.matchAll(/(\w+)="([^"]*)"/g)].map(m=>[m[1],m[2]]));
  const r=record();r.tagUuids=[BLOG_TAG];
  r.note.content=[{insert:`原声测试\naudio: ${d.name}\nartist: ${d.artist}\n${d.url}\naudio_cover: ${d.cover}\n`}];
  const a=convertArticle(r);assert.deepEqual(a.audios,[d.url]);assert.deepEqual(a.images,[d.cover]);
  assert.ok(a.source.includes('data-name="Cornfield Chase"'));assert.ok(a.source.includes('data-artist="Hans Zimmer"'));
});
