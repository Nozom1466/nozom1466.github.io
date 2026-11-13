import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
const catalog=JSON.parse(await readFile(new URL('../../docs/library-catalog.json',import.meta.url)));
const quote=JSON.stringify;
let created=0,existing=0;
for(const c of catalog){
 assert(c.cover,`Missing cover: ${c.slug}`);
 assert(Object.keys(c.creators).length>0&&Object.keys(c.creators).length<=5,c.slug);
 const file=new URL(`../../content/library/${c.slug}.md`,import.meta.url);
 const lines=['---',`title: ${quote(c.title)}`,`cover: ${quote(c.cover)}`,'library_tags:',`  - ${c.category}`,'date: 2026-11-13',`catalog_id: ${quote(c.slug)}`];
 if(c.releaseYear)lines.push(`release_year: ${c.releaseYear}`);
 lines.push('creators:',...Object.entries(c.creators).map(([k,v])=>`  ${k}: ${quote(v)}`),'---','');
 const content=lines.join('\n');
 // Import is create-only. Future handwritten/SnapNote reviews must not be replaced.
 try{await writeFile(file,content,{flag:'wx'});created++;}
 catch(e){if(e.code==='EEXIST'){existing++;console.log('Preserved',fileURLToPath(file));}else throw e;}
}
console.log(`Created ${created}, preserved ${existing}. No ratings, summaries or review text generated.`);
