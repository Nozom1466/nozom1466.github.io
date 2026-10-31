import {parseLines,safeLink} from '../../static/moments/render.mjs';
import {BLOG_TAG,channelOf} from '../../static/moments/channels.mjs';
import {COS_ORIGIN} from '../../static/moments/cos-feed.mjs';
import {convertArticle} from './convert.mjs';

const roles=['author','director','screenplay','writer','original','translator','developer','publisher','producer','music','composer','artwork','cinematographer','character'];
const keys=new Set(['draft','labels','summary','rating','cover','date','status',...roles]);
const plain=line=>line.parts.map(p=>p.text||'').join('').trim();
function imageURL(name){
  if(!/^[\w.-]+\.(jpe?g|png|webp|gif)$/i.test(name)||name.includes('..'))throw Error('书馆封面图片路径无效');
  return COS_ORIGIN+'/SnapNote_Sync/resources/'+encodeURIComponent(name);
}
function coverURL(value){
  const link=safeLink(value);if(!link)throw Error('cover 需要图片或有效的 COS HTTPS 链接');
  const url=new URL(link);
  if(url.protocol!=='https:'||url.search||url.hash||!((url.origin===COS_ORIGIN&&url.pathname.startsWith('/SnapNote_Sync/resources/'))||(url.hostname==='myblog-1316371247.cos.ap-shanghai.myqcloud.com'&&url.pathname.startsWith('/myblog/'))))throw Error('cover 仅支持现有两个 COS 空间的图片');
  return link;
}
export function convertReview(raw,{draft=true,skipDrafts=false}={}){
  const n=raw.note;
  if(channelOf(raw)!=='review'||!n||n.hasMarkDeleted||n.isArchived||n.isLocked||raw.stateVector?.deleted)throw Error('不是可展示的 review 笔记');
  const lines=parseLines(n.content);
  while(lines.length&&!plain(lines[0])&&!lines[0].parts.some(p=>p.embed))lines.shift();
  const first=lines.shift(),title=first&&plain(first);
  if(!title||first.parts.some(p=>p.embed)||keys.has(title.split(/[：:]/)[0].trim().toLowerCase()))throw Error('书馆评价第一行缺少作品名称，请在 draft 等字段前填写书名或片名');
  const metadata=new Map();let cover='';
  while(lines.length){
    const line=lines[0];if(line.parts.some(p=>p.embed)||Object.keys(line.attrs).length)break;
    const text=plain(line);if(!text){lines.shift();continue;}
    const m=text.match(/^([a-z]+)\s*[：:]\s*(.*)$/i);if(!m||!keys.has(m[1].toLowerCase()))break;
    const key=m[1].toLowerCase();if(metadata.has(key))throw Error(`书馆字段 ${key} 重复`);
    metadata.set(key,m[2].trim());lines.shift();
    if(key==='cover'&&!m[2].trim()){
      while(lines.length&&!plain(lines[0])&&!lines[0].parts.some(p=>p.embed))lines.shift();
      const next=lines[0];
      if(next&&next.parts.length===1&&next.parts[0].embed?.image){cover=imageURL(next.parts[0].embed.image);lines.shift();}
    }
  }
  const draftValue=metadata.get('draft');
  if(draftValue!==undefined&&!/^(true|false)$/i.test(draftValue))throw Error('draft 必须填写 true 或 false');
  if(skipDrafts&&draftValue?.toLowerCase()==='true')return null;
  let rating;
  if(metadata.has('rating')){
    const value=metadata.get('rating');
    if(!/^(?:\d+(?:\.\d+)?|\.\d+)$/.test(value)||Number(value)<0||Number(value)>5)throw Error('rating 必须是 0 到 5 的数字');
    rating=Number(value);
  }
  const coverValue=metadata.get('cover');
  if(coverValue&&coverValue.toLowerCase()!=='first')cover=coverURL(coverValue);
  if(!cover){
    // Without a URL, consume the first image as the cover, not a duplicate body image.
    for(let i=0;i<lines.length&&!cover;i++){
      const index=lines[i].parts.findIndex(p=>p.embed?.image);
      if(index>=0){cover=imageURL(lines[i].parts[index].embed.image);lines[i].parts.splice(index,1);}
    }
  }
  if(!cover)throw Error('书馆评价缺少封面，请在 cover: 下一行插入图片');
  const ops=[{insert:title+'\n'}];
  for(const key of ['draft','labels','summary'])if(metadata.has(key))ops.push({insert:key+': '+metadata.get(key)+'\n'});
  for(const line of lines){for(const p of line.parts)ops.push({insert:p.embed||p.text,attributes:p.attrs});ops.push({insert:'\n',attributes:line.attrs});}
  const article=convertArticle({...raw,tagUuids:[BLOG_TAG],note:{...n,content:ops}},{draft,skipDrafts});
  if(!article)return null;
  const end=article.source.indexOf('\n\n'),front=JSON.parse(article.source.slice(0,end)),body=article.source.slice(end);
  front.cover=cover;front.library_tags=front.tags.length?front.tags:['book'];delete front.tags;
  if(rating!==undefined)front.rating=rating;
  front.creators={};let order=1;
  for(const [key,value] of metadata)if(roles.includes(key)&&value){front.creators[`${order++}_${key==='composer'?'music':key}`]=value;}
  if(metadata.has('author'))front.author=metadata.get('author');
  if(metadata.has('date')){
    const value=metadata.get('date');
    if(!/^\d{4}-\d{2}-\d{2}$/.test(value)||!Number.isFinite(Date.parse(value))||new Date(value).toISOString().slice(0,10)!==value)throw Error('date 必须是有效的 YYYY-MM-DD 日期');
    front.date=value;
  }
  if(metadata.has('status')){
    const value=metadata.get('status'),states={reading:'Reading',completed:'Completed','not started':'Not Started'};
    if(!states[value.toLowerCase()])throw Error('status 支持 Reading、Completed、Not Started');front.status=states[value.toLowerCase()];
  }
  return {...article,section:'library',images:[cover,...article.images],source:JSON.stringify(front,null,2)+body};
}
