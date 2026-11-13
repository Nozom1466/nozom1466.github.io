import assert from 'node:assert/strict';
import {readFile,stat} from 'node:fs/promises';
import {resolve} from 'node:path';
import {load} from 'cheerio';
import sharp from 'sharp';
import vm from 'node:vm';
const catalog=JSON.parse(await readFile(new URL('../../docs/library-catalog.json',import.meta.url)));
assert.equal(new Set(catalog.map(x=>x.slug)).size,catalog.length);
assert.equal(new Set(catalog.map(x=>x.title)).size,catalog.length);
assert.equal(catalog.filter(x=>/^conan-m/.test(x.slug)).length,29);
assert.equal(catalog.filter(x=>/^expendables/.test(x.slug)).length,2);
assert.equal(catalog.filter(x=>/^star-wars-/.test(x.slug)).length,3);
assert(catalog.some(x=>x.slug==='blitz-brigade'));
const removed=['kaguya-1-cn','kaguya-2-cn','kaguya-3-cn','qin-2','qin-3','qin-4','qin-5','qin-6','lvxing-2','lvxing-3','our-lvxing-1','earth-lvxing-3','earth-lvxing-4','earth-lvxing-5','earth-lvxing-6'];
for(const slug of removed){
 assert(!catalog.some(x=>x.slug===slug),slug+': removed from catalog');
 await assert.rejects(stat(new URL(`../../content/library/${slug}.md`,import.meta.url)),{code:'ENOENT'});
}
assert.equal(catalog.find(x=>x.slug==='qin-1')?.title,'秦时明月（系列）');
assert.equal(catalog.find(x=>x.slug==='lvxing-1')?.title,'侣行（系列）');
for(const slug of ['kaguya-1','kaguya-2','kaguya-3','kaguya-first-kiss','nine-songs'])assert(catalog.some(x=>x.slug===slug));
assert(!catalog.some(x=>/game-of-war/i.test(x.slug)||/Game of War/i.test(x.title)));
for(const slug of ['mr-six','zhao-wen-dao','ball-lightning','shouwang-de-juli'])assert.equal(catalog.filter(x=>x.slug===slug).length,1);
let bytes=0;
for(const c of catalog){
 assert(c.sources.length&&c.coverOriginal&&c.coverSource,c.slug+': sources');
 assert(c.cover.startsWith('/library-covers/'),c.slug);
 assert(Object.keys(c.creators).length>=1&&Object.keys(c.creators).length<=5,c.slug+': credits');
 for(const v of Object.values(c.creators))assert(v.trim()&&!/undefined|null/.test(v),c.slug);
 const source=await readFile(new URL(`../../content/library/${c.slug}.md`,import.meta.url),'utf8');
 assert(source.includes(`catalog_id: "${c.slug}"`));
 assert(!/^rating:|^summary:/m.test(source),c.slug+': no invented opinions');
 assert.equal(source.split('---')[2].trim(),'');
 const cover=new URL('../../static'+c.cover,import.meta.url);
 const meta=await sharp(await readFile(cover)).metadata();
 assert.equal(meta.format,'webp');assert(meta.width<meta.height,c.slug+': portrait cover');
 assert(meta.width<=640&&meta.height<=960,c.slug+': bounded size');
 const size=(await stat(cover)).size;assert(size<400000,c.slug+': image bytes');bytes+=size;
}
const base=await readFile(new URL('../../layouts/library/baseof.html',import.meta.url),'utf8');
const functionText=base.slice(base.indexOf('function renderStarRating'),base.indexOf('function openLibraryPopup'));
const context=vm.createContext({});vm.runInContext(functionText,context);
for(const v of [null,undefined,'',' ','invalid'])assert.equal(context.renderStarRating(v),'');
assert(context.renderStarRating('0').includes('0.0/5.0'));
assert(context.renderStarRating('4.9').includes('4.9/5.0'));
if(process.argv[2]){
 const destination=resolve(process.argv[2]);const $=load(await readFile(resolve(destination,'library/index.html'),'utf8'));
 for(const slug of removed)assert.equal($(`[data-review-id="${slug}"]`).length,0,slug+': not rendered');
 for(const c of catalog){
  const card=$(`[data-review-id="${c.slug}"]`);
  assert.equal(card.length,1,c.slug+': rendered once');
  assert.equal(card.attr('data-rating'),'');assert.equal(card.find('.library-card-rating').length,0);
  assert.equal(card.find('img').attr('src'),c.cover);
  assert.equal(card.attr('data-tags'),c.category);
  assert.deepEqual(JSON.parse(card.attr('data-creators')),c.creators);
  assert.equal($(`#review-${c.slug}`).text().trim(),'');
  await stat(resolve(destination,'.'+c.cover));
 }
 const old=$('[data-review-id="Interstellar"]');
 assert.equal(old.attr('data-rating'),'4.9');assert(old.find('.card-rating-number').text().includes('4.9'));
 assert($('#review-Interstellar').text().includes('Cornfield Chase'));
 for(const category of new Set(catalog.map(x=>x.category)))assert.equal($(`.tag-link[data-tag="${category}"]`).length,1);
}
console.log(`PASS: ${catalog.length} entries, covers ${(bytes/1024/1024).toFixed(2)} MiB, credits, sources, versions, no fabricated reviews, unrated display${process.argv[2]?', built HTML and existing rated review':''}.`);
