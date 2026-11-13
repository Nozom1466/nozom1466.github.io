import {createServer} from 'node:http';
import {readFile,stat} from 'node:fs/promises';
import {resolve,join,extname,relative,isAbsolute} from 'node:path';
const root=resolve(process.argv[2]||'scripts/library-catalog/cache/site');
const types={'.html':'text/html; charset=utf-8','.css':'text/css','.js':'text/javascript','.mjs':'text/javascript','.json':'application/json','.webp':'image/webp','.png':'image/png','.jpg':'image/jpeg','.svg':'image/svg+xml','.woff2':'font/woff2'};
createServer(async(req,res)=>{
 try{
  let file=resolve(root,'.'+decodeURIComponent(new URL(req.url,'http://localhost').pathname));
  const rel=relative(root,file);if(rel.startsWith('..')||isAbsolute(rel)){res.writeHead(403).end();return;}
  if((await stat(file)).isDirectory())file=join(file,'index.html');
  const body=await readFile(file);res.writeHead(200,{'Content-Type':types[extname(file)]||'application/octet-stream','Cache-Control':'no-store'}).end(body);
 }catch{res.writeHead(404).end('Not found');}
}).listen(4183,'127.0.0.1',()=>console.log('Library preview: http://127.0.0.1:4183/library/'));
