// QA thumbnails only: original publication images are left unchanged.
import {readFile,mkdir} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import sharp from 'sharp';
const catalog=JSON.parse(await readFile(new URL('../../docs/library-catalog.json',import.meta.url)));
const dir=new URL('./cache/contact-sheets/',import.meta.url);await mkdir(dir,{recursive:true});
for(let start=0;start<catalog.length;start+=30){
 const group=catalog.slice(start,start+30),layers=[];
 for(let i=0;i<group.length;i++){
  const c=group[i],left=i%5*200,top=Math.floor(i/5)*245;
  const pic=await sharp(await readFile(new URL('../../static'+c.cover,import.meta.url))).resize(180,210,{fit:'contain',background:'#eeeeee'}).png().toBuffer();
  layers.push({input:pic,left:left+10,top});
  const label=Buffer.from(`<svg width="200" height="30"><text x="5" y="18" font-size="11" font-family="Arial">${start+i+1}. ${c.slug}</text></svg>`);
  layers.push({input:label,left,top:top+211});
 }
 const out=fileURLToPath(new URL(`${Math.floor(start/30)+1}.jpg`,dir));
 await sharp({create:{width:1000,height:Math.ceil(group.length/5)*245,channels:3,background:'#ffffff'}}).composite(layers).jpeg({quality:88}).toFile(out);
 console.log(out);
}
