// Mechanical front-matter-only migration. No body, URL, rating or date rewrites.
import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
import {originalNames,normalizeOriginalNames} from './original-names.mjs';
let changed=0;
for(const id of Object.keys(originalNames)){
 const file=new URL(`../../content/library/${id}.md`,import.meta.url);
 const source=await readFile(file,'utf8');let next;
 if(source.startsWith('{')){
  const end=source.indexOf('\n\n'),front=JSON.parse(source.slice(0,end)),body=source.slice(end);
  next=JSON.stringify(normalizeOriginalNames(front,id),null,2)+body;
  assert.equal(next.slice(next.indexOf('\n\n')),body);
 }else{
  const match=source.match(/^(---\r?\n)([\s\S]*?)(\r?\n---)([\s\S]*)$/);
  assert(match,`Unexpected front matter: ${id}`);
  const scalar=v=>{v=v.trim();return v.startsWith('"')?JSON.parse(v):v};
  const title=scalar(match[2].match(/^title: (.*)$/m)[1]);
  const entry=originalNames[id];if(!entry.aliases.includes(title))continue;
  let inCreators=false;
  const header=match[2].split(/(?<=\n)/).map(line=>{
   if(/^title: /.test(line))return line.replace(/^title: [^\r\n]*/,`title: ${JSON.stringify(entry.title)}`);
   if(/^creators:/.test(line)){inCreators=true;return line;}
   if(/^\S/.test(line))inCreators=false;
   const person=inCreators&&line.match(/^(\s+\S+: )([^\r\n]*)(\r?\n)?$/);
   if(person){const value=scalar(person[2]),native=entry.creatorNames[value];if(native)return person[1]+JSON.stringify(native)+(person[3]||'');}
   return line;
  }).join('');
  next=match[1]+header+match[3]+match[4];
  assert.equal(next.slice(next.indexOf(match[3])+match[3].length),match[4]);
 }
 if(next!==source){await writeFile(file,next);changed++;}
}
console.log(`Normalized ${changed} front matters; review bodies preserved.`);
