import {readFileSync} from 'node:fs';
export const originalNames=JSON.parse(readFileSync(new URL('./original-names.json',import.meta.url),'utf8')).works;

// Match the stable work ID AND a verified title; never translate arbitrary user prose
// or apply the old identity after a note is repurposed for a different work.
export function normalizeOriginalNames(front,id=front.catalog_id||front.slug){
 const entry=originalNames[id];
 if(!entry||!entry.aliases.includes(front.title))return front;
 const name=value=>entry.creatorNames[value]||value;
 return {...front,title:entry.title,
  ...(front.creators?{creators:Object.fromEntries(Object.entries(front.creators).map(([role,value])=>[role,name(value)]))}:{}),
  ...(front.author?{author:name(front.author)}:{})};
}
