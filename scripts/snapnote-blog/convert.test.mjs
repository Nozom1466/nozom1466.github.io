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

test('publishes blog tags and pairs captions with the preceding image',()=>{
  const r=record();r.note.content.splice(2,0,{insert:'标签：Blender, AI，随笔, AI\n\n'});
  r.note.content.push({insert:'图注：教师办公室 <参考图>\n后续正文\n'});
  const a=convertArticle(r,{draft:false});
  const front=JSON.parse(a.source.split('\n\n')[0]);
  assert.equal(front.draft,false);assert.deepEqual(front.tags,['Blender','AI','随笔']);
  assert.ok(a.source.includes('<figcaption'));assert.ok(a.source.includes('教师办公室 &lt;参考图&gt;'));
  assert.ok(!a.source.includes('标签：'));assert.ok(!a.source.includes('图注：'));
  assert.ok(a.source.includes('<p>后续正文</p>'));assert.ok(!front.summary.includes('教师办公室'));
});
test('a caption not immediately following an image stays ordinary text',()=>{
  const r=record();r.note.content.push({insert:'普通段落\n图注：这不是配图说明\n'});
  const a=convertArticle(r);assert.ok(!a.source.includes('<figcaption'));assert.ok(a.source.includes('<p>图注：这不是配图说明</p>'));
});
