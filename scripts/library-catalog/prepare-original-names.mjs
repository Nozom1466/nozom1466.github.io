// Build an explicit, reviewable registry from the researched source records.
// Run deliberately when curating the catalog, never during a scheduled COS sync.
import {readFile,writeFile} from 'node:fs/promises';
import works from './works.mjs';
const previous=JSON.parse(await readFile(new URL('./original-names.json',import.meta.url)).catch(()=>'{"works":{}}'));
const chinese=new Set(`liaoshen-campaign huaihai-campaign pingjin-campaign crossing-chishui dying-to-survive wandering-earth wandering-earth-2 taking-tiger-mountain lost-on-journey article-20 three-body-tv crime-crackdown longest-day-in-changan nezha-2019 three-body-book dark-forest deaths-end three-body-animation swallowed-star swallowed-star-blood swallowed-star-original qin-1 nine-songs lvxing-1 mr-six zhao-wen-dao ball-lightning shouwang-de-juli`.split(' '));
const titles={
 'leon':'Léon','seoul-spring':'서울의 봄','three-body-tv':'三体','three-body-animation':'三体',
 'kaguya-first-kiss':'かぐや様は告らせたい-ファーストキッスは終わらない-',
};
const people={
 'Shuichi Okita':'沖田修一','Shirō Maeda':'前田司郎','Toshikazu Nishigaya, Yuki Nishimiya, Yasushi Yamazaki':'西ヶ谷寿一、西宮由貴、山崎康史',
 'Ren Takada':'高田漣','Shūichi Yoshida':'吉田修一','Shunji Iwai':'岩井俊二','Hidemi Satani':'佐谷秀美','Aki Mizuno':'水野昌','Takeshi Kobayashi':'小林武史',
 'Hirokazu Kore-eda':'是枝裕和','Kaoru Matsuzaki, Hijiri Taguchi':'松崎薫、田口聖','Yoko Kanno':'菅野よう子','Akimi Yoshida':'吉田秋生',
 'Chihiro Ikeda':'池田千尋','Izumi Takahashi, Chihiro Ikeda':'髙橋泉、池田千尋',
 'Yusuke Terada, Kazumi Matsui, Seiya Horio, Shoichi Ishikawa, Reiko Sasaki':'寺田悠輔、松井和美、堀尾星矢、石川翔一、佐々木礼子',
 'Nobuaki Shinzawa':'信澤宣明','Makoto Ojiro':'オジロマコト',
 'Wen Muye':'文牧野','Han Jianü, Zhong Wei, Wen Muye':'韩家女、钟伟、文牧野','Ning Hao, Xu Zheng':'宁浩、徐峥','Huang Chao':'黄超',
 'Frant Gwo':'郭帆','Gong Ge\'er, Yan Dongxu, Frant Gwo, Ye Junce, Yang Zhixue, Wu Yi, Ye Ruchang':'龚格尔、严东旭、郭帆、叶俊策、杨治学、吴荑、叶濡畅',
 'Frant Gwo, Gong Ge\'er, Yang Zhixue, Ye Ruchang':'郭帆、龚格尔、杨治学、叶濡畅','Liu Cixin, Gong Ge\'er':'刘慈欣、龚格尔','Liu Cixin':'刘慈欣',
 'Roc Chen, Liu Tao (additional music)':'阿鲲、刘韬','Roc Chen':'阿鲲',
 'Kim Sung-su':'김성수','Hong In-pyo, Hong Won-Chan, Lee Young-jong, Kim Sung-su':'홍인표、홍원찬、이영종、김성수','Kim Won-kuk':'김원국','Lee Jae-jin':'이재진',
 'Tsui Hark':'徐克','Huang Xin, Li Yang, Tsui Hark, Wu Bing, Dong Zhe, Lin Chi-an':'黄欣、李杨、徐克、吴兵、董哲、林启安','Huang Jianxin, Nansun Shi, Yu Dong':'黄建新、施南生、于冬','Wu Wai-lap':'胡伟立','Qu Bo':'曲波',
 'Raymond Yip':'叶伟民','Liu Yiwei':'刘仪伟','Manfred Wong':'文隽','Kay Chan':'陈永健',
 'Zhang Yimou':'张艺谋','Li Meng, Wang Tianyi':'李萌、王天毅',
 'Beijing Enlight Pictures, Center for Film & Television of the Supreme People\'s Procuratorate':'北京光线影业、最高人民检察院影视中心',
 'Wu Bai':'五百','Du Liang':'都梁','Cao Dun':'曹盾','Paw Studio':'爪子工作室','Ma Boyong':'马伯庸',
 'Central Political and Legal Affairs Commission, Penguin Film and Television, China Chang\'an Publishing Media, Zhongguang Film and Television, Wu Yuan Culture Company':'中央政法委、企鹅影视、中国长安出版传媒、中广影视、五元文化',
 'Nam Seung-yong, Ahn Jae-chul':'남승용、안재철','Katsuo Ono, Masakazu Yokoyama':'大野克夫、横山克',
 'Haruka Nakamura':'haruka nakamura',
};
const extraSources={
 'yokomichi-yonosuke':['https://eiga.com/movie/57830/'],
 'insomniacs-after-school-film':['https://kimisomu-movie.com/','https://www.ponycanyon.co.jp/music/PCCG000002258'],
 'seoul-spring':['https://www.koreafilm.or.kr/museum/theater/module/TI_00000078','https://news.yonsei.ac.kr/kr/people/detail?bbSeq=27493'],
 'running-man':['https://tv.charmy.global/series/9'],
 'leon':['https://www.gaumont.com/en/movie/leon'],
 'wandering-earth':['https://zh.wikipedia.org/wiki/流浪地球_(电影)'],
 'taking-tiger-mountain':['https://cinehello.com/stream/immerse/68921'],
 'lost-on-journey':['https://upimg.baike.so.com/doc/3287878-3463555.html'],
 'article-20':['https://m.1905.com/m/film/starlist/2257241.shtml'],
};
const registry={version:1,works:{}};
const catalog=JSON.parse(await readFile(new URL('../../docs/library-catalog.json',import.meta.url)));
for(const w of works){
 const d=JSON.parse(await readFile(new URL(`./cache/${w.slug}.json`,import.meta.url)));
 const c=catalog.find(c=>c.slug===w.slug);
 let title=titles[w.slug]||(chinese.has(w.slug)?w.title:d.coverSubject?.name||d.heading);
 title=title.replace(/ \((?:\d{4} film|film|miniseries|TV program)\)$/,'');
 if(/^conan-m/.test(w.slug))title=`M${w.slug.slice(-2)} · ${title}`;
 const creatorNames={...previous.works[w.slug]?.creatorNames};
 for(const value of Object.values(c.creators))if(people[value])creatorNames[value]=people[value];
 // Native subject credits correct a mistaken English-source screenwriter credit.
 if(w.slug==='crime-crackdown')creatorNames['Du Liang']=d.coverFields['编剧'];
 const sources=[d.source,d.coverBgm&&`https://bgm.tv/subject/${d.coverBgm}`,w.bgm&&`https://bgm.tv/subject/${w.bgm}`,...extraSources[w.slug]||[]].filter(Boolean);
 registry.works[w.slug]={title,aliases:[...new Set([...(previous.works[w.slug]?.aliases||[]),w.title,title])],creatorNames,sources};
}
// Existing personal reviews: only names change; summaries, ratings and prose stay intact.
const legacy={
 '5-centimeters-per-second-live-action':['秒速5センチメートル',['秒速5厘米（真人版）'],{'铃木史子':'鈴木史子','新海诚《秒速5厘米》':'新海誠『秒速5センチメートル』'},['https://5cm-movie.jp/']],
 'addictive-study-method':['科学的根拠に基づく最高の勉強法',['上瘾式学习：科学的自学成才法'],{},['https://www.kadokawa.co.jp/product/322310000993/']],
 'hibike-euphonium-2':['響け！ユーフォニアム2',['響け！ユーフォニアム2 - TV'],{'武田 绫乃':'武田綾乃'},['https://tv2.anime-eupho.com/']],
 'hibike-euphonium-3':['響け！ユーフォニアム3',['響け！ユーフォニアム3 - TV'],{'武田 绫乃':'武田綾乃'},['https://anime-eupho.com/']],
 'hibike-euphonium-fiction':['響け！ユーフォニアム',[],{'武田 绫乃':'武田綾乃'},['https://tkj.jp/info/euphonium/']],
 'how-an-economy-grows-and-why-it-crashes':['How an Economy Grows and Why It Crashes',['小岛经济学'],{},['https://www.wiley.com/en-us/how-an-economy-grows-and-why-it-crashes-p-9780470526705']],
 'isomu-anime':['君は放課後インソムニア',[],{'尾城真':'オジロマコト'},['https://kimisomu-anime.com/']],
 'isomu':['君は放課後インソムニア',['君は放課後インソムニア 漫画'],{'尾城真':'オジロマコト'},['https://kimisomu-anime.com/']],
 'jaemyeong-life-road':['인간 이재명',['李在明的人生之路'],{'叙事内容研究所[韩]':'스토리텔링콘텐츠연구소'},['https://m.newspim.com/newsamp/view/20251127000712','https://ebook.skku.edu/ebook/?cate_id=001001011&mode=new&page_num=10&sort=pubdt']],
 'project-hail-mary':['Project Hail Mary',['挽救计划'],{},['https://www.penguinrandomhouse.com/books/611060/project-hail-mary-by-andy-weir/']],
 'road-to-happiness':['The Conquest of Happiness',['幸福之路'],{},['https://www.routledge.com/The-Conquest-of-Happiness/Russell/p/book/9780415378475']],
 'snapnote-26546fe9-8f91-4260-8ba1-5e3e9c9d1b95':['신화는 없다',['经营未来'],{'李明博':'이명박'},['https://www.yes24.com/product/goods/55723','https://jubeny.com/2026/04/book-summaries-18/']],
 'the-autobiography-of-mao-tse-tung':['The Autobiography of Mao Tse-tung',['毛泽东自传'],{},['https://www.chinawriter.com.cn/xzp/2016/2016-06-13/274225.html']],
 'white-tower':['白い巨塔',['白色巨塔'],{},['https://ndlsearch.ndl.go.jp/books/R100000002-I000004202925']],
 'zelda-breath-of-the-wild':['ゼルダの伝説 ブレス オブ ザ ワイルド',['ゼルダの伝説 ブレスオブザワイルド'],{},['https://www.nintendo.com/jp/zelda/botw/']],
 'liz-and-the-blue-bird':['リズと青い鳥',[],{'山田 尚子':'山田尚子','吉田 玲子':'吉田玲子','牛尾 憲輔':'牛尾憲輔'},['https://liz-bluebird.com/']],
 'pingpong':['ピンポン THE ANIMATION',[],{'湯浅 政明':'湯浅政明','松本 大洋':'松本大洋','牛尾 憲輔':'牛尾憲輔','伊東 伸高':'伊東伸高'},['https://www.pingpong-anime.tv/']],
};
for(const [id,[title,aliases,creatorNames,sources]]of Object.entries(legacy))registry.works[id]={title,aliases:[...new Set([...aliases,title])],creatorNames,sources};
await writeFile(new URL('./original-names.json',import.meta.url),JSON.stringify(registry,null,2)+'\n');
console.log(`Prepared original names for ${Object.keys(registry.works).length} works.`);
