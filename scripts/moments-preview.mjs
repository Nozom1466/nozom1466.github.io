// Local-only read-only bridge: Hugo layout + existing in-memory COS session.
// No credentials or private note snapshots belong in this repository.
import http from 'node:http';
import {Readable} from 'node:stream';
const allowedHosts=new Set(['127.0.0.1:4179','localhost:4179']);
http.createServer(async(req,res)=>{
  const json=(code,error)=>{res.writeHead(code,{'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-store'});res.end(JSON.stringify({error}));};
  if(!allowedHosts.has(req.headers.host)||req.headers['sec-fetch-site']==='cross-site'||(req.headers.origin&&req.headers.origin!==`http://${req.headers.host}`))return json(403,'仅允许本机同源访问。');
  if(!['GET','HEAD'].includes(req.method))return json(405,'只读预览。');
  try{
    const url=new URL(req.url,'http://127.0.0.1:4179');let target;
    if(url.pathname==='/api/moments/notes'){
      const query=new URLSearchParams();for(const key of ['cursor','limit','refresh'])if(url.searchParams.has(key))query.set(key,url.searchParams.get(key));
      target='http://127.0.0.1:4178/api/notes?'+query;
    }
    else if(/^\/api\/moments\/media\/[\w.-]+$/.test(url.pathname)&&!url.pathname.includes('..'))target='http://127.0.0.1:4178/media/'+url.pathname.split('/').pop()+(url.searchParams.get('thumbnail')==='1'?'?thumbnail=1':'');
    else if(url.pathname.startsWith('/api/'))return json(404,'Not found');
    else target='http://127.0.0.1:1314'+url.pathname+url.search;
    const headers={};if(req.headers['if-none-match'])headers['If-None-Match']=req.headers['if-none-match'];
    const upstream=await fetch(target,{method:req.method,headers,redirect:'manual',signal:AbortSignal.timeout(60000)});
    res.statusCode=upstream.status;
    for(const name of ['content-type','content-length','location','etag'])if(upstream.headers.has(name))res.setHeader(name,upstream.headers.get(name));
    res.setHeader('Cache-Control',url.pathname.startsWith('/api/moments/media/')?(upstream.headers.get('cache-control')||'no-store'):'no-store');res.setHeader('X-Content-Type-Options','nosniff');res.setHeader('Referrer-Policy','no-referrer');res.setHeader('X-Frame-Options','SAMEORIGIN');
    if(req.method==='HEAD'||!upstream.body)return res.end();
    Readable.fromWeb(upstream.body).on('error',()=>res.destroy()).pipe(res);
  }catch{if(!res.headersSent)json(502,'本地预览服务未就绪，请确认 Hugo 和 COS 服务均在运行。');else res.end();}
}).listen(4179,'127.0.0.1',()=>console.log('Local: http://127.0.0.1:4179/moments/'));
