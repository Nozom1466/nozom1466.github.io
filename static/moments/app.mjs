import {visibleNotes} from './render.mjs';
import {mountContent} from './content-view.mjs';
import {createFeed} from './feed.mjs';
const timeline=document.querySelector('#moments-timeline'),status=document.querySelector('#moments-status'),refresh=document.querySelector('#moments-refresh'),modal=document.querySelector('#moments-lightbox');
const zone='America/New_York';
const more=document.querySelector('#moments-more'),newer=document.querySelector('#moments-newer'),end=document.querySelector('#moments-end');
const MAX_PAGES=5;
const feedSettings=document.querySelector('#moments').dataset;
const readPage=createFeed({mode:feedSettings.feedMode||'live',base:feedSettings.feedBase||'/moments-feed/'});
let pages=[],cursors=[null],busy=false,hasMore=false,autoLoad=true;
document.querySelector('#moments-close').onclick=()=>modal.close();
modal.addEventListener('click',e=>{if(e.target===modal)modal.close();});
timeline.addEventListener('click',e=>{const button=e.target.closest('[data-image]');if(button){modal.querySelector('img').src=button.dataset.image;modal.showModal();modal.focus({preventScroll:true});}});
function renderPage(records,index,media=null){
  const group=document.createElement('div');group.className='moment-page';group.dataset.page=index;
  for(const {note} of records){
    const date=new Date(note.createdAtMs);
    const parts=Object.fromEntries(new Intl.DateTimeFormat('en-US',{timeZone:zone,year:'numeric',month:'short',day:'2-digit'}).formatToParts(date).map(x=>[x.type,x.value]));
    const time=new Intl.DateTimeFormat('en-GB',{timeZone:zone,hour:'2-digit',minute:'2-digit',hour12:false}).format(date);
    const article=document.createElement('article');article.className='moment-entry';
    article.innerHTML=`<div class="moment-date"><span class="moment-day">${parts.day}</span><span>${parts.month}</span><span>${parts.year}</span></div><div class="moment-card"><time class="moment-time" datetime="${date.toISOString()}" title="纽约时间">${time}</time><div class="moment-body"></div></div>`;
    mountContent(article.querySelector('.moment-body'),note.content,200,media);group.append(article);
  }
  return {index,node:group};
}
function controls(){
  refresh.disabled=busy;more.disabled=busy;newer.disabled=busy;
  more.hidden=!hasMore;more.textContent=busy?'正在加载…':'加载更多';
  newer.hidden=!pages.length||pages[0].index===0;
  end.hidden=hasMore||!pages.length;
}
async function load(mode='reset'){
  if(busy)return;
  const index=mode==='reset'?0:mode==='newer'?pages[0].index-1:(pages.at(-1)?.index??-1)+1;
  if(index<0||(index>0&&!cursors[index]))return;
  busy=true;status.textContent='';controls();
  try{
    const data=await readPage({index,reset:mode==='reset',cursor:cursors[index]});
    const records=visibleNotes(data.notes),page=renderPage(records,index,data.media??null);
    if(mode==='reset'){const wasBrowsing=pages.length>0;timeline.replaceChildren();pages=[];cursors=[null];autoLoad=true;if(wasBrowsing)window.scrollTo(0,0);}
    cursors[index]=data.cursor;cursors[index+1]=data.nextCursor;
    if(mode==='newer'){
      const anchor=pages[0]?.node,before=anchor?.getBoundingClientRect().top;
      timeline.prepend(page.node);pages.unshift(page);
      if(pages.length>MAX_PAGES)pages.pop().node.remove();
      if(anchor)window.scrollBy(0,anchor.getBoundingClientRect().top-before);
    }else{
      timeline.append(page.node);pages.push(page);
      if(pages.length>MAX_PAGES){
        const anchor=pages[1].node,before=anchor.getBoundingClientRect().top;
        pages.shift().node.remove();window.scrollBy(0,anchor.getBoundingClientRect().top-before);
      }
    }
    hasMore=!!cursors[pages.at(-1).index+1];
    if(!records.length&&mode==='reset')status.textContent='暂无记录。';
    refresh.title=`刷新 · 上次读取 ${new Date(data.refreshedAt).toLocaleTimeString('zh-CN')}`;
  }catch(error){status.textContent=error.message;autoLoad=false;}
  finally{busy=false;controls();}
}
refresh.onclick=()=>load('reset');more.onclick=()=>{autoLoad=true;load('older');};newer.onclick=()=>load('newer');
if('IntersectionObserver' in window){
  const observer=new IntersectionObserver(entries=>{if(entries.some(e=>e.isIntersecting)&&hasMore&&!busy&&autoLoad)load('older');},{rootMargin:'240px'});
  observer.observe(document.querySelector('#moments-sentinel'));
}
load();
