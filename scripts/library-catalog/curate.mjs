// Explicit corrections after reading the linked work pages; never infer credits from an image.
import {readFile,writeFile} from 'node:fs/promises';
import {load} from 'cheerio';
import works from './works.mjs';
const updates={
 'edge-of-tomorrow':{coverSource:'https://www.impawards.com/2014/edge_of_tomorrow_xlg.html',image:'https://www.impawards.com/2014/posters/edge_of_tomorrow_xlg.jpg'},
 'elysium':{coverSource:'https://www.sonypictures.com/movies/elysium',image:'https://www.sonypictures.com/sites/default/files/chameleon/title-movie/462025_Elysium_1400x2100_English_US.jpg'},
 'swallowed-star-original':{coverSource:'https://wetv.vip/zh-cn/play/cnqbubq7s6nxixe',image:'https://vcover-vt-pic.wetvinfo.com/vcover_vt_pic/0/cnqbubq7s6nxixe1777528344420/0'},
 'gta-iii':{coverSource:'https://www.stephenbliss.com/rockstar-games',image:'https://images.squarespace-cdn.com/content/v1/5441c4e5e4b07b71baeb625d/1419996736413-V8JXB0T47OLGK1NVUSVM/GTA3+FOB.+USA.+2002.jpg'},
 'gta-vice-city':{coverSource:'https://www.stephenbliss.com/rockstar-games',image:'https://images.squarespace-cdn.com/content/v1/5441c4e5e4b07b71baeb625d/1419996771632-934CRFOK8NCCIKPZY771/GTAVC+FOB.+USA.+2003.jpg'},
 'shouwang-de-juli':{source:'https://search.megbook.com.tw/mall/detail.jsp?proID=3004011',heading:'守望的距离',releaseYear:2017,fields:{作者:'周国平',出版社:'浙江人民出版社',ISBN:'9787213079634'},image:'http://img.megbook.hk/upload/mall/productImages/17/20/9787213079634.jpg'},
 'mr-six':{source:'https://m.1905.com/m/film/2224190.shtml',heading:'老炮儿',releaseYear:2015,fields:{导演:'管虎',编剧:'管虎、董润年',制片商:'华谊兄弟'},image:'https://image13.m1905.cn/mdb/uploadfile/2015/1210/20151210111542997820.jpg'},
 'zhao-wen-dao':{source:'https://book.douban.com/subject/30312029/',heading:'朝闻道',releaseYear:2018,fields:{作者:'刘慈欣',出版社:'长江文艺出版社',ISBN:'9787570204403'},image:'https://img3.doubanio.com/view/subject/l/public/s29855723.jpg'},
 'ball-lightning':{source:'https://item.xhsd.com/items/1010000100501769',heading:'球状闪电',releaseYear:2005,fields:{作者:'刘慈欣',出版社:'四川科学技术出版社',ISBN:'9787536455382'},image:'https://img2.xinhuashudian.com/bookbasepic/C/00029/75364553800810554-fm.jpg'},
 'huaihai-campaign':{source:'https://m.1905.com/m/film/1929545.shtml',heading:'大决战之淮海战役',date:'1991-08-01',fields:{导演:'李俊、蔡继渭、韦廉、杨光远',编剧:'李平分、史超、王军'},image:'https://image14.m1905.cn/uploadfile/2012/0605/20120605040151243.jpg'},
 'pingjin-campaign':{source:'https://m.1905.com/m/film/1929546.shtml',heading:'大决战之平津战役',date:'1992-01-01',fields:{导演:'李俊、蔡继渭、杨光远、韦廉',编剧:'李平分、史超、王军'},image:'https://image13.m1905.cn/mdb/uploadfile/2024/0515/20240515094758441822.jpg'},
 'crossing-chishui':{source:'https://m.1905.com/m/film/61026.shtml',heading:'四渡赤水',releaseYear:1983,fields:{导演:'蔡继渭、谷德显',编剧:'黎明、王昊、王愿坚、李传弟'},image:'https://image13.m1905.cn/uploadfile/2012/0601/20120601110137598.jpg'},
 'blitz-brigade':{coverSource:'https://www.carlvaudrin.com/music.php',image:'https://www.carlvaudrin.com/img/music/blitz_brigade.jpg'},
 'brawl-stars':{coverSource:'https://www.imdb.com/title/tt9480088/',image:'https://m.media-amazon.com/images/M/MV5BMzVkNDFkNjEtZDE1OC00MTJmLWIxNjQtYWM1MWQxNThiODI0XkEyXkFqcGc%40._V1_.jpg'},
 'running-man':{coverSource:'https://ent.ifeng.com/a/20161025/42707170_0.shtml',image:'https://p2.ifengimg.com/a/2016_44/33ee04b73e96cbd_size382_w560_h630.jpg'},
 'passengers':{coverSource:'https://www.impawards.com/2016/passengers.html',image:'https://www.impawards.com/2016/posters/passengers.jpg'},
 '13-hours':{coverSource:'https://www.impawards.com/2016/thirteen_hours_the_secret_soldiers_of_benghazi.html',image:'https://www.impawards.com/2016/posters/thirteen_hours_the_secret_soldiers_of_benghazi.jpg'},
};
const selected=process.argv.slice(2);
for(const [slug,id] of [['dying-to-survive',2240923],['article-20',2257241],['taking-tiger-mountain',2214203]]){
 if(selected.length&&!selected.includes(slug))continue;
 const url=`https://m.1905.com/m/film/${id}.shtml`;
 const r=await fetch(url,{signal:AbortSignal.timeout(20000)});if(!r.ok)throw Error(r.status);
 const $=load(await r.text());const image=$('meta[property="og:image"]').attr('content');
 if(!image)throw Error('Missing cover '+slug);
 updates[slug]={coverSource:url,image};
}
for(const [slug,update] of Object.entries(updates)){
 if(selected.length&&!selected.includes(slug))continue;
 const path=new URL(`./cache/${slug}.json`,import.meta.url);
 let old;try{old=JSON.parse(await readFile(path));}catch{old={...works.find(w=>w.slug===slug),retrieved:'2026-11-13'}}
 if(old.image===update.image&&old.asset)continue;
 const data={...old,...update};delete data.asset;delete data.coverNeedsReview;
 if(!data.coverSource)data.coverSource=data.source;
 await writeFile(path,JSON.stringify(data,null,2)+'\n');
 console.log(slug,data.image);
}
