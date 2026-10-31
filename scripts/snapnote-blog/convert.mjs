import {parseLines,escapeHTML,safeLink} from '../../static/moments/render.mjs';
import {channelOf} from '../../static/moments/channels.mjs';
import {COS_ORIGIN} from '../../static/moments/cos-feed.mjs';

const escape = value => escapeHTML(value).replaceAll('{','&#123;').replaceAll('}','&#125;');
export function convertArticle(raw,{draft=true}={}) {
  const n=raw.note;
  if(!n || !/^[a-f0-9-]{36}$/i.test(n.uuid))throw Error('笔记 ID 无效');
  if(channelOf(raw)!=='blog'||n.hasMarkDeleted||n.isLocked||n.isArchived||raw.stateVector?.deleted)throw Error('不是可预览的 blog 笔记');
  const lines=parseLines(n.content);
  while(lines.length&&!lines[0].parts.length)lines.shift();
  const first=lines.shift();
  if(!first||first.parts.some(p=>p.embed))throw Error('文章第一行需要纯文字标题');
  const title=first.parts.map(p=>p.text||'').join('').trim();
  if(!title)throw Error('文章标题为空');
  const plain=line=>line.parts.map(p=>p.text||'').join('').trim();
  let tags=[],customSummary='';const metadata=new Set();
  while(lines.length){
    const line=lines[0];
    if(line.parts.some(p=>p.embed)||Object.keys(line.attrs).length)break;
    const value=plain(line);
    if(!value){lines.shift();continue;}
    const match=value.match(/^(labels|标签|summary)\s*[：:]\s*(.*)$/i);
    if(!match)break;
    const key=match[1].toLowerCase()==='summary'?'summary':'labels';
    if(metadata.has(key))throw Error(`文章 ${key} 重复，请只填写一行`);
    metadata.add(key);
    if(key==='summary')customSummary=match[2].trim();
    else{
      tags=[...new Set(match[2].split(/[,，、]/).map(t=>t.trim()).filter(Boolean))];
      if(tags.length>20||tags.some(t=>t.length>60||/[<>{}\r\n]/.test(t)))throw Error('文章标签格式无效');
    }
    lines.shift();
  }
  const images=[];
  function inline(part){
    if(part.embed)throw Error('正文包含尚未适配的附件或公式，停止转换');
    const a=part.attrs;let text=escape(part.text);
    for(const [attr,tag] of [['bold','strong'],['italic','em'],['strike','s'],['underline','u'],['code','code']])if(a[attr])text=`<${tag}>${text}</${tag}>`;
    if(a.link){const href=safeLink(a.link);if(!href)throw Error('文章链接无效');text=`<a href="${escape(href)}" rel="noopener noreferrer">${text}</a>`;}
    return text;
  }
  function image(part,caption=''){
    const name=part.embed.image;
    if(!/^[\w.-]+\.(jpe?g|png|webp|gif)$/i.test(name)||name.includes('..'))throw Error('图片路径无效');
    const src=COS_ORIGIN+'/SnapNote_Sync/resources/'+encodeURIComponent(name);images.push(src);
    return `<figure><a href="${src}" target="_blank" rel="noopener noreferrer"><img src="${src}" alt="${escape(caption||'文章配图')}" loading="lazy" style="display:block;max-width:100%;height:auto;margin:1.5em auto"></a>${caption?`<figcaption style="text-align:center;font-size:.9em;opacity:.75;margin-top:.5em">${escape(caption)}</figcaption>`:''}</figure>`;
  }
  const blocks=[];let list=null;
  const close=()=>{if(list){blocks.push(`</${list}>`);list=null;}};
  const summaryParts=[];
  for(let i=0;i<lines.length;i++){
    const {parts,attrs:a}=lines[i];
    if(a.table||a.callout||a['code-block']||a.indent)throw Error('此富文本格式尚未适配，停止转换');
    if(parts.some(p=>p.embed)){
      close();let text=[];
      const flush=()=>{if(text.length)blocks.push(`<p>${text.map(inline).join('')}</p>`);text=[];};
      for(let j=0;j<parts.length;j++){
        const p=parts[j];
        if(p.embed?.image){
          flush();let caption='';
          if(j===parts.length-1){
            let next=i+1;while(next<lines.length&&!plain(lines[next])&&!lines[next].parts.some(p=>p.embed))next++;
            const match=lines[next]&&!lines[next].parts.some(p=>p.embed)&&plain(lines[next]).match(/^(?:caption|图注)\s*[：:]\s*(.+)$/i);
            if(match){caption=match[1];i=next;}
          }
          blocks.push(image(p,caption));
        }else if(p.embed){throw Error('暂不支持此附件类型');}else {text.push(p);summaryParts.push(p.text||'');}
      }flush();continue;
    }
    const value=parts.map(inline).join('');
    if(!value.trim()){close();continue;}
    summaryParts.push(...parts.map(p=>p.text||''));
    if(a.list==='ordered'||a.list==='bullet'){
      const kind=a.list==='ordered'?'ol':'ul';if(list!==kind){close();list=kind;blocks.push(`<${kind}>`);}blocks.push(`<li>${value}</li>`);continue;
    }
    close();
    if(a.list)throw Error('待办列表尚未适配');
    if(Number.isInteger(a.header)&&a.header>=1&&a.header<=6){const level=Math.max(2,a.header);blocks.push(`<h${level}>${value}</h${level}>`);}
    else if(a.blockquote)blocks.push(`<blockquote><p>${value}</p></blockquote>`);
    else blocks.push(`<p>${value}</p>`);
  }
  close();
  const text=summaryParts.join(' ').replace(/\s+/g,' ').trim();
  const summary=customSummary||(Array.from(text).slice(0,100).join('')+(Array.from(text).length>100?'…':''));
  const front={title,date:new Date(n.createdAtMs).toISOString(),lastmod:new Date(n.updatedAtMs??n.createdAtMs).toISOString(),slug:'snapnote-'+n.uuid,summary:escape(summary),draft,showtoc:false,tags,snapnote_uuid:n.uuid,snapnote_revision:raw.revisionId};
  return {title,images,source:JSON.stringify(front,null,2)+'\n\n'+blocks.join('\n\n')+'\n',filename:'snapnote-'+n.uuid+'.md'};
}
