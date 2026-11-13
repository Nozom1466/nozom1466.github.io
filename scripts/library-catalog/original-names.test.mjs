import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {load} from 'cheerio';
import {originalNames,normalizeOriginalNames} from './original-names.mjs';
test('curated originals cover catalog titles and keep matching native credits unchanged',async()=>{
 const catalog=JSON.parse(await readFile(new URL('../../docs/library-catalog.json',import.meta.url)));
 for(const c of catalog){
  assert.equal(c.title,originalNames[c.slug].title,c.slug);
  assert.deepEqual(normalizeOriginalNames(c),c,c.slug+': idempotent');
  for(const value of Object.values(c.creators))assert(!originalNames[c.slug].creatorNames[value],c.slug+': untranslated credit');
 }
 assert.equal(originalNames['hyouka'].title,'氷菓');
 assert.equal(originalNames['running-man'].title,'런닝맨');
 assert.equal(originalNames['gta-vice-city'].title,'Grand Theft Auto: Vice City');
 assert.equal(originalNames['liaoshen-campaign'].title,'大决战之辽沈战役');
 assert(originalNames['conan-m01'].title.startsWith('M01 · 名探偵コナン '));
});
test('normalization preserves user prose, ratings, links and additional creators',()=>{
 const front={title:'秒速5厘米（真人版）',summary:'新海诚的故事',rating:4.7,cover:'/cover.jpg',slug:'unchanged',creators:{'1_screenplay':'铃木史子','2_cast':'森七菜'},body:'新海诚《秒速5厘米》'};
 const result=normalizeOriginalNames(front,'5-centimeters-per-second-live-action');
 assert.equal(result.title,'秒速5センチメートル');assert.equal(result.creators['1_screenplay'],'鈴木史子');
 assert.equal(result.creators['2_cast'],'森七菜');
 for(const key of ['summary','rating','cover','slug','body'])assert.equal(result[key],front[key]);
 assert.equal(front.title,'秒速5厘米（真人版）');
 assert.equal(normalizeOriginalNames(front,'unknown'),front);
});
if(process.env.LIBRARY_VERIFY_DIR){
 test('existing review titles and credits render from the same registry',async()=>{
  const html=await readFile(new URL('../../'+process.env.LIBRARY_VERIFY_DIR+'/library/index.html',import.meta.url));
  const $=load(html);
  for(const [id,entry]of Object.entries(originalNames)){
   const card=$(`[data-review-id="${id}"]`);assert.equal(card.length,1,id);
   assert.equal(card.attr('data-title'),entry.title,id);
   const creators=JSON.parse(card.attr('data-creators'));
   for(const value of Object.values(creators))assert(!entry.creatorNames[value],id+': native credit');
  }
 });
}
