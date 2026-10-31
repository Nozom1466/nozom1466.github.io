import assert from 'node:assert/strict';
import {renderContent,excerpt} from '../static/moments/render.mjs';
const picture=i=>({insert:{image:`photo-${i}.jpg`}});
for(const count of [1,2,3,4,5,6,7,9]){
  const html=renderContent(Array.from({length:count},(_,i)=>[picture(i),{insert:'\n\n'}]).flat());
  assert.equal((html.match(/class="photo-grid"/g)||[]).length,1);
  assert.equal((html.match(/class="photo"/g)||[]).length,count);
  assert.equal((html.match(/class="photo-zoom"/g)||[]).length,count);
  assert.ok(!html.includes('查看大图'));
  assert.ok(html.includes(`data-columns="${Math.min(count,3)}"`));
  for(let i=1;i<count;i++)assert.ok(html.indexOf(`photo-${i-1}.jpg`)<html.indexOf(`photo-${i}.jpg`));
}
const caption=renderContent([picture(1),{insert:'\n图片说明\n'},picture(2)]);
assert.equal((caption.match(/class="photo-grid"/g)||[]).length,2);
assert.ok(caption.indexOf('图片说明')>caption.indexOf('photo-1.jpg'));
assert.ok(caption.indexOf('图片说明')<caption.indexOf('photo-2.jpg'));
const mixed=renderContent([{insert:'之前'},picture(1),picture(2),{insert:'之后\n'}]);
assert.equal((mixed.match(/class="photo-grid"/g)||[]).length,1);assert.ok(mixed.indexOf('之前')<mixed.indexOf('photo-1'));assert.ok(mixed.indexOf('之后')>mixed.indexOf('photo-2'));
const long=renderContent(excerpt([{insert:'中'.repeat(201)+'\n'},picture(1),{insert:'\n'},picture(2)]).content);
assert.ok(long.includes('data-columns="2"'));
assert.ok(!renderContent([{insert:{image:'../secret.jpg'}}]).includes('data-image'));
assert.ok(renderContent([{insert:'<script>\n'}]).includes('&lt;script&gt;'));
console.log('PASS: 1–9 images grouped, order and captions preserved, mixed lines, excerpt images, path safety and escaping.');
