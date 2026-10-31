import test from 'node:test';
import assert from 'node:assert/strict';
import {convertArticle} from './convert.mjs';
import {BLOG_TAG,THREADS_TAG} from '../../static/moments/channels.mjs';
const record=()=>({revisionId:'a'.repeat(32),tagUuids:[BLOG_TAG],note:{uuid:'5c035e0b-8a05-4f9c-939b-ce617f1904a5',createdAtMs:1700000000000,content:[{insert:'文章标题'},{insert:'\n',attributes:{header:1}},{insert:'第一段\n第二段\n'},{insert:'重点',attributes:{bold:true}},{insert:'\n'},{insert:{image:'photo.png'}},{insert:'\n'}]}});
test('converts title, paragraphs, emphasis and original image into a local draft',()=>{
  const r=record(),a=convertArticle(r);assert.equal(a.title,'文章标题');
  assert.ok(a.source.includes('"draft": true'));assert.ok(a.source.includes('<p>第一段</p>'));
  assert.ok(a.source.includes('<strong>重点</strong>'));assert.equal(a.images.length,1);
  assert.ok(!a.source.includes('<h1>文章标题</h1>'));
  r.note.content[0].insert='改标题';assert.equal(convertArticle(r).filename,a.filename);
});
test('rejects other channels, hidden notes and unsupported embeds',()=>{
  const r=record();r.tagUuids=[THREADS_TAG];assert.throws(()=>convertArticle(r));
  r.tagUuids=[BLOG_TAG];r.note.isLocked=true;assert.throws(()=>convertArticle(r));
  r.note.isLocked=false;r.note.content.push({insert:{video:'movie.mp4'}},{insert:'\n'});assert.throws(()=>convertArticle(r));
});
test('does not execute HTML or Hugo shortcodes from note text',()=>{
  const r=record();r.note.content.push({insert:'<script>alert(1)</script> {{< unsafe >}}\n'});
  const a=convertArticle(r);assert.ok(!a.source.includes('<script>'));assert.ok(!a.source.includes('{{<'));
  assert.ok(a.source.includes('&lt;script&gt;'));
});

for(const [labels,caption] of [['labels:','caption:'],['Labels:','CAPTION:'],['标签：','图注：']])
test(`publishes tags and captions with ${labels} / ${caption}`,()=>{
  const r=record();r.note.content.splice(2,0,{insert:`${labels} Blender, AI，随笔, AI\n\n`});
  r.note.content.push({insert:`${caption} 教师办公室 <参考图>\n后续正文\n`});
  const a=convertArticle(r,{draft:false});
  const front=JSON.parse(a.source.split('\n\n')[0]);
  assert.equal(front.draft,false);assert.deepEqual(front.tags,['Blender','AI','随笔']);
  assert.ok(a.source.includes('<figcaption'));assert.ok(a.source.includes('教师办公室 &lt;参考图&gt;'));
  assert.ok(!a.source.includes(labels));assert.ok(!a.source.includes(caption));
  assert.ok(a.source.includes('<p>后续正文</p>'));assert.ok(!front.summary.includes('教师办公室'));
});
test('a caption not immediately following an image stays ordinary text',()=>{
  const r=record();r.note.content.push({insert:'普通段落\ncaption: 这不是配图说明\n'});
  const a=convertArticle(r);assert.ok(!a.source.includes('<figcaption'));assert.ok(a.source.includes('<p>caption: 这不是配图说明</p>'));
});

for(const header of ['summary: 自定义摘要\nlabels: AI, 随笔\n','labels: AI, 随笔\n\nSUMMARY：自定义摘要\n'])
test(`custom summary and labels work in either order: ${header.split('\n')[0]}`,()=>{
  const r=record();r.note.content.splice(2,0,{insert:header});
  const a=convertArticle(r),[meta,body]=a.source.split(/\n\n(.*)/s),front=JSON.parse(meta);
  assert.equal(front.summary,'自定义摘要');assert.deepEqual(front.tags,['AI','随笔']);
  assert.ok(!body.includes('自定义摘要'));assert.ok(!body.includes('labels:'));assert.ok(body.includes('<p>第一段</p>'));
});
test('empty summary falls back; summary in normal body is not consumed',()=>{
  const r=record();r.note.content.splice(2,0,{insert:'summary: \n'});r.note.content.push({insert:'summary: 正文文字\n'});
  const a=convertArticle(r),front=JSON.parse(a.source.split('\n\n')[0]);
  assert.ok(front.summary.startsWith('第一段'));assert.ok(a.source.includes('<p>summary: 正文文字</p>'));
});
test('custom summary is escaped, not truncated, and duplicate metadata fails',()=>{
  const r=record();const summary='<script> {{< unsafe >}} '+ '长'.repeat(120);
  r.note.content.splice(2,0,{insert:'summary: '+summary+'\n'});
  const a=convertArticle(r),front=JSON.parse(a.source.split('\n\n')[0]);
  assert.ok(front.summary.includes('&lt;script&gt;'));assert.ok(!a.source.includes('{{<'));assert.ok(front.summary.endsWith('长'.repeat(120)));
  r.note.content.splice(3,0,{insert:'summary: 重复\n'});assert.throws(()=>convertArticle(r),/重复/);
});

test('draft metadata hides blog posts, false publishes and preview stays a draft',()=>{
  for(const value of ['true','TRUE','false']){
    const r=record();r.note.content.splice(2,0,{insert:`labels: AI\nDRAFT: ${value}\nsummary: 摘要\n`});
    const expected=value.toLowerCase()==='true';
    const a=convertArticle(r,{draft:false});
    assert.equal(JSON.parse(a.source.split('\n\n')[0]).draft,expected);
    assert.ok(!a.source.includes('DRAFT:'));
    assert.equal(convertArticle(r,{draft:false,skipDrafts:true})===null,expected);
    assert.equal(JSON.parse(convertArticle(r).source.split('\n\n')[0]).draft,true);
  }
});
test('invalid or repeated draft fails instead of publishing; body text is not metadata',()=>{
  for(const value of ['', 'yes','flase','true\ndraft: false']){
    const r=record();r.note.content.splice(2,0,{insert:`draft: ${value}\n`});
    assert.throws(()=>convertArticle(r,{draft:false,skipDrafts:true}),/draft/);
  }
  const r=record();r.note.content.push({insert:'draft: true\n'});
  const a=convertArticle(r,{draft:false,skipDrafts:true});assert.ok(a.source.includes('<p>draft: true</p>'));
});
