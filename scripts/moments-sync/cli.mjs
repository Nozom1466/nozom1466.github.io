import COS from 'cos-nodejs-sdk-v5';
import {appendFile,mkdtemp} from 'node:fs/promises';
import path from 'node:path';
import {tmpdir} from 'node:os';
import {syncFeed} from './sync.mjs';

try{
  const secretId=process.env.COS_SECRET_ID?.trim(),secretKey=process.env.COS_SECRET_KEY?.trim();
  if(!secretId||!secretKey||/\s/.test(secretId)||/\s/.test(secretKey))throw {safeMessage:'请配置 COS_SECRET_ID 和 COS_SECRET_KEY 两个 Repository secrets。'};
  const base=new URL(process.env.MOMENTS_SITE_URL);
  if(base.protocol!=='https:'||base.username||base.password)throw {safeMessage:'线上站点地址配置无效。'};
  if(!base.pathname.endsWith('/'))base.pathname+='/';
  const manifestURL=new URL('moments-feed/manifest.json',base);manifestURL.searchParams.set('check',String(Date.now()));
  const response=await fetch(manifestURL,{signal:AbortSignal.timeout(20000),cache:'no-store'});
  let publishedVersion=null;
  if(response.ok){const data=await response.json();if(!/^[a-f0-9]{32}$/.test(data.version))throw {safeMessage:'线上清单格式异常，停止发布。'};publishedVersion=data.version;}
  else if(response.status!==404)throw {safeMessage:'无法核对线上版本，停止发布，等待重试。'};
  const outputDir=await mkdtemp(path.join(process.env.RUNNER_TEMP||tmpdir(),'moments-export-'));
  const client=new COS({SecretId:secretId,SecretKey:secretKey,Timeout:20000,FollowRedirect:false});
  const result=await syncFeed({client,cacheDir:process.env.MOMENTS_CACHE_DIR||'.moments-cache',outputDir,publishedVersion,force:process.env.GITHUB_EVENT_NAME!=='schedule'});
  if(process.env.GITHUB_OUTPUT)await appendFile(process.env.GITHUB_OUTPUT,`deploy=${result.shouldDeploy}\ncache_changed=${result.cacheChanged}\nexport_dir=${outputDir}\n`);
  console.log(JSON.stringify(result));
}catch(error){
  // SDK error objects may contain signed URLs or request headers. Never log them.
  console.error(error.safeMessage||'COS 同步或网络检查失败，本次不发布；请核查密钥权限、文件格式和网络。');process.exitCode=1;
}
