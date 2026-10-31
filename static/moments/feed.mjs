// Static publication and the local live COS preview share the same UI.
export function createFeed({mode='static',base='/moments-feed/',fetcher=fetch}={}){
  let manifest=null;
  async function read(url){
    const response=await fetcher(url,{cache:'no-store'});
    if(!response.ok){
      if(mode==='live'){const data=await response.json().catch(()=>({}));throw new Error(data.error||'读取失败。');}
      throw new Error(response.status===404?'公开记录尚未生成，或已更新；请稍后刷新。':'读取记录失败，请稍后刷新。');
    }
    if(!response.headers.get('content-type')?.includes('application/json'))throw new Error('记录接口返回格式异常。');
    return response.json();
  }
  return async({index=0,reset=false,cursor=null}={})=>{
    if(mode==='live'){
      const query=new URLSearchParams({limit:'20'});if(reset)query.set('refresh','1');else if(cursor)query.set('cursor',cursor);
      return read('/api/moments/notes?'+query);
    }
    if(reset||!manifest){
      manifest=await read(base+'manifest.json?check='+Date.now());
      if(manifest.schema!==1||!/^[a-f0-9]{32}$/.test(manifest.version)||!Number.isInteger(manifest.pages)||manifest.pages<1||manifest.pageSize!==20)throw new Error('公开记录清单格式异常。');
    }
    if(!Number.isInteger(index)||index<0||index>=manifest.pages)throw new Error('分页位置无效。');
    if(!reset&&cursor&&cursor!==manifest.version+':'+index)throw new Error('记录已更新，请刷新。');
    const data=await read(`${base}${manifest.version}/page-${index}.json`);
    if(data.version!==manifest.version||data.page!==index||!Array.isArray(data.notes)||data.notes.length>20)throw new Error('分页数据不完整，请刷新。');
    const media={};
    for(const [name,item] of Object.entries(data.media||{})){
      if(/^media\/[a-f0-9]{64}\.(jpe?g|png|webp|gif)$/.test(item?.original)&&/^media\/[a-f0-9]{64}\.thumb\.webp$/.test(item?.thumbnail))media[name]={original:base+item.original,thumbnail:base+item.thumbnail};
    }
    return {...data,media,total:manifest.total,cursor:manifest.version+':'+index,nextCursor:index+1<manifest.pages?manifest.version+':'+(index+1):null,refreshedAt:manifest.refreshedAt};
  };
}
