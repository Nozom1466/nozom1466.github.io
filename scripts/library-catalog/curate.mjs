// Explicit corrections after reading the linked work pages; never infer credits from an image.
import {readFile,writeFile} from 'node:fs/promises';
import {load} from 'cheerio';
import works from './works.mjs';
const updates={
 'huaihai-campaign':{source:'https://m.1905.com/m/film/1929545.shtml',heading:'大决战之淮海战役',date:'1991-08-01',fields:{导演:'李俊、蔡继渭、韦廉、杨光远',编剧:'李平分、史超、王军'},image:'https://image14.m1905.cn/uploadfile/2012/0605/20120605040151243.jpg'},
 'pingjin-campaign':{source:'https://m.1905.com/m/film/1929546.shtml',heading:'大决战之平津战役',date:'1992-01-01',fields:{导演:'李俊、蔡继渭、杨光远、韦廉',编剧:'李平分、史超、王军'},image:'https://image13.m1905.cn/mdb/uploadfile/2024/0515/20240515094758441822.jpg'},
 'crossing-chishui':{source:'https://m.1905.com/m/film/61026.shtml',heading:'四渡赤水',releaseYear:1983,fields:{导演:'蔡继渭、谷德显',编剧:'黎明、王昊、王愿坚、李传弟'},image:'https://image13.m1905.cn/uploadfile/2012/0601/20120601110137598.jpg'},
 'gta-chinatown':{coverSource:'https://www.ionstore.nl/nl/grand-theft-auto-chinatown-wars-playstation-psp.html',image:'https://cdn.webshopapp.com/shops/272655/files/488456918/image.jpg'},
 'blitz-brigade':{coverSource:'https://www.carlvaudrin.com/music.php',image:'https://www.carlvaudrin.com/img/music/blitz_brigade.jpg'},
 'brawl-stars':{coverSource:'https://www.imdb.com/title/tt9480088/',image:'https://m.media-amazon.com/images/M/MV5BMzVkNDFkNjEtZDE1OC00MTJmLWIxNjQtYWM1MWQxNThiODI0XkEyXkFqcGc%40._V1_.jpg'},
 'running-man':{coverSource:'https://sports.khan.co.kr/article/202404150000013',image:'https://images.khan.co.kr/article/2024/04/15/news-p.v1.20240414.db4acbb57fb64421bb63e36131051214.png'},
 'passengers':{coverSource:'https://www.impawards.com/2016/passengers.html',image:'https://www.impawards.com/2016/posters/passengers.jpg'},
 '13-hours':{coverSource:'https://www.impawards.com/2016/thirteen_hours_the_secret_soldiers_of_benghazi.html',image:'https://www.impawards.com/2016/posters/thirteen_hours_the_secret_soldiers_of_benghazi.jpg'},
};
for(const [slug,id] of [['dying-to-survive',2240923],['article-20',2257241],['taking-tiger-mountain',2214203]]){
 const url=`https://m.1905.com/m/film/${id}.shtml`;
 const r=await fetch(url,{signal:AbortSignal.timeout(20000)});if(!r.ok)throw Error(r.status);
 const $=load(await r.text());const image=$('meta[property="og:image"]').attr('content');
 if(!image)throw Error('Missing cover '+slug);
 updates[slug]={coverSource:url,image};
}
for(const [slug,update] of Object.entries(updates)){
 const path=new URL(`./cache/${slug}.json`,import.meta.url);
 let old;try{old=JSON.parse(await readFile(path));}catch{old={...works.find(w=>w.slug===slug),retrieved:'2026-11-13'}}
 if(old.image===update.image&&old.asset)continue;
 const data={...old,...update};delete data.asset;delete data.coverNeedsReview;
 if(!data.coverSource)data.coverSource=data.source;
 await writeFile(path,JSON.stringify(data,null,2)+'\n');
 console.log(slug,data.image);
}
