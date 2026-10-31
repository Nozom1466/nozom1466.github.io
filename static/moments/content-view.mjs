import {excerpt,renderContent} from './render.mjs';
export function mountContent(host, content, limit=200, media=null) {
  const preview=excerpt(content,limit);
  const body=document.createElement('div');body.className='content';host.append(body);
  let expanded=false;
  const render=()=>{
    body.innerHTML=renderContent(expanded?content:preview.content,media);
    body.querySelectorAll('[data-formula]').forEach(el=>{
      if(window.katex) try{window.katex.render(el.dataset.formula,el,{throwOnError:false,trust:false});}catch{el.textContent=el.dataset.formula;}
    });
    body.querySelectorAll('.photo img').forEach(img=>img.addEventListener('error',()=>{
      const error=document.createElement('span');error.className='image-error';error.textContent='图片未加载';img.closest('button').replaceWith(error);
    },{once:true}));
  };
  render();
  if(preview.truncated){
    const button=document.createElement('button');button.className='expand-text';button.textContent='显示全文';button.setAttribute('aria-expanded','false');
    button.onclick=()=>{expanded=!expanded;render();button.textContent=expanded?'收起':'显示全文';button.setAttribute('aria-expanded',String(expanded));};host.append(button);
  }
}
