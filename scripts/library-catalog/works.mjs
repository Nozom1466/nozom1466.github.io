// Explicit work/version selection. Ambiguous recollections stay in LIBRARY-BACKLOG.md.
const rows = [];
function add(category, data) {
  for (const line of data.trim().split('\n')) {
    const [slug, title, wiki] = line.split('|');
    rows.push({slug, title, category, wiki});
  }
}
add('film', `
yokomichi-yonosuke|横道世之介|A Story of Yonosuke
liaoshen-campaign|大决战之辽沈战役|zh:大决战之辽沈战役
huaihai-campaign|大决战之淮海战役|zh:大决战之淮海战役
pingjin-campaign|大决战之平津战役|zh:大决战之平津战役
crossing-chishui|四渡赤水|zh:四渡赤水 (电影)
hana-and-alice|花与爱丽丝|Hana and Alice
april-story|四月物语|April Story
last-letter|最后一封信|Last Letter (2020 film)
our-little-sister|海街日记|Our Little Sister
we-made-a-beautiful-bouquet|花束般的恋爱|We Made a Beautiful Bouquet
last-ten-years|余命十年|The Last 10 Years
insomniacs-after-school-film|放学后失眠的你（真人电影）|Insomniacs After School (film)
shawshank-redemption|肖申克的救赎|The Shawshank Redemption
dying-to-survive|我不是药神|Dying to Survive
wandering-earth|流浪地球|The Wandering Earth
wandering-earth-2|流浪地球2|The Wandering Earth 2
avengers|复仇者联盟|The Avengers (2012 film)
avengers-age-of-ultron|复仇者联盟2：奥创纪元|Avengers: Age of Ultron
avengers-infinity-war|复仇者联盟3：无限战争|Avengers: Infinity War
avengers-endgame|复仇者联盟4：终局之战|Avengers: Endgame
iron-man|钢铁侠|Iron Man (2008 film)
iron-man-2|钢铁侠2|Iron Man 2
iron-man-3|钢铁侠3|Iron Man 3
captain-america-civil-war|美国队长3：内战|Captain America: Civil War
seoul-spring|首尔之春|12.12: The Day
taking-tiger-mountain|智取威虎山|The Taking of Tiger Mountain
pacific-rim|环太平洋|Pacific Rim (film)
resident-evil-final-chapter|生化危机：终章|Resident Evil: The Final Chapter
dune-2021|沙丘|Dune (2021 film)
star-wars-7|星球大战7：原力觉醒|Star Wars: The Force Awakens
star-wars-8|星球大战8：最后的绝地武士|Star Wars: The Last Jedi
star-wars-9|星球大战9：天行者崛起|Star Wars: The Rise of Skywalker
lost-on-journey|人在囧途|Lost on Journey
x-men|X战警|X-Men (film)
x-men-days-of-future-past|X战警：逆转未来|X-Men: Days of Future Past
x-men-apocalypse|X战警：天启|X-Men: Apocalypse
13-hours|危机13小时|13 Hours: The Secret Soldiers of Benghazi
tron-legacy|创：战纪|Tron: Legacy
martian|火星救援|The Martian (film)
passengers|太空旅客|Passengers (2016 film)
leon|这个杀手不太冷|Léon: The Professional
inception|盗梦空间|Inception
tenet|信条|Tenet (film)
article-20|第二十条|Article 20
truman-show|楚门的世界|The Truman Show
expendables|敢死队|The Expendables (2010 film)
expendables-2|敢死队2|The Expendables 2
`);
add('tv', `
three-body-tv|三体（电视剧）|Three-Body
band-of-brothers|兄弟连|Band of Brothers (miniseries)
agents-of-shield-1|神盾局特工 第一季|Agents of S.H.I.E.L.D. season 1
agents-of-shield-2|神盾局特工 第二季|Agents of S.H.I.E.L.D. season 2
agents-of-shield-3|神盾局特工 第三季|Agents of S.H.I.E.L.D. season 3
crime-crackdown|扫黑风暴|Crime Crackdown
longest-day-in-changan|长安十二时辰|The Longest Day in Chang'an
`);
add('anime', `
your-name|你的名字。|Your Name
look-back|蓦然回首|Look Back (2024 film)
colors-within|你的颜色|The Colors Within
garden-of-words|言叶之庭|The Garden of Words
five-centimeters-anime|秒速5厘米（动画）|5 Centimeters per Second
suzume|铃芽之旅|Suzume
place-promised|云之彼端，约定的地方|The Place Promised in Our Early Days
children-chase-stars|追逐繁星的孩子|Children Who Chase Lost Voices
she-and-her-cat|她和她的猫（新海诚短片）|She and Her Cat
pancreas-anime|我想吃掉你的胰脏（动画）|I Want to Eat Your Pancreas (film)
summer-ghost|夏日幽灵|Summer Ghost
josee-anime|乔西的虎与鱼（动画）|Josee, the Tiger and the Fish (2020 film)
violet-side-story|紫罗兰永恒花园 外传：永远与自动手记人偶|Violet Evergarden: Eternity and the Auto Memory Doll
violet-movie|紫罗兰永恒花园 剧场版|Violet Evergarden: The Movie
k-on-movie|轻音少女 剧场版|K-On! The Movie
nezha-2019|哪吒之魔童降世|Ne Zha (2019 film)
toy-story|玩具总动员|Toy Story
toy-story-2|玩具总动员2|Toy Story 2
toy-story-3|玩具总动员3|Toy Story 3
zootopia|疯狂动物城|Zootopia
minions|小黄人大眼萌|Minions (film)
minions-2|小黄人大眼萌2：神偷奶爸前传|Minions: The Rise of Gru
`);
add('book', `
three-body-book|三体|The Three-Body Problem (novel)
dark-forest|三体Ⅱ：黑暗森林|The Dark Forest
deaths-end|三体Ⅲ：死神永生|Death's End
futari-kyori|两人距离的概算|ja:ふたりの距離の概算
imasara-tsubasa|迟来的翅膀|ja:いまさら翼といわれても
universe-nutshell|果壳中的宇宙|The Universe in a Nutshell
totto-chan|窗边的小豆豆|Totto-Chan: The Little Girl at the Window
namiya-store|解忧杂货店|The Miracles of the Namiya General Store
`);
add('game', `
gta-chinatown|侠盗猎车手：血战唐人街|Grand Theft Auto: Chinatown Wars
gta-san-andreas|侠盗猎车手：圣安地列斯|Grand Theft Auto: San Andreas
polytopia|低模之战|The Battle of Polytopia
league-of-legends|英雄联盟|League of Legends
brawl-stars|荒野乱斗|Brawl Stars
blitz-brigade|闪电部队|Blitz Brigade
`);
add('variety', `running-man|Running Man|Running Man (TV program)`);
const conan = [
['引爆摩天楼','The Time Bombed Skyscraper'],['第14个目标','The Fourteenth Target'],['世纪末的魔术师','The Last Wizard of the Century'],['瞳孔中的暗杀者','Captured in Her Eyes'],['通往天国的倒计时','Countdown to Heaven'],['贝克街的亡灵','The Phantom of Baker Street'],['迷宫的十字路口','Crossroad in the Ancient Capital'],['银翼的魔术师','Magician of the Silver Sky'],['水平线上的阴谋','Strategy Above the Depths'],['侦探们的镇魂歌',"The Private Eyes' Requiem"],['绀碧之棺','Jolly Roger in the Deep Azure'],['战栗的乐谱','Full Score of Fear'],['漆黑的追踪者','The Raven Chaser'],['天空的遇难船','The Lost Ship in the Sky'],['沉默的15分钟','Quarter of Silence'],['第11个前锋','The Eleventh Striker'],['绝海的侦探','Private Eye in the Distant Sea'],['异次元的狙击手','Dimensional Sniper'],['业火的向日葵','Sunflowers of Inferno'],['纯黑的噩梦','The Darkest Nightmare'],['唐红的恋歌','The Crimson Love Letter'],['零的执行人','Zero the Enforcer'],['绀青之拳','The Fist of Blue Sapphire'],['绯色的子弹','The Scarlet Bullet'],['万圣夜的新娘','The Bride of Halloween'],['黑铁的鱼影','Black Iron Submarine'],['百万美元的五棱星','The Million-dollar Pentagram'],['独眼的残像','One-eyed Flashback'],['高速公路的堕天使','Fallen Angel of the Highway']];
conan.forEach(([title, en], i) => rows.push({slug:`conan-m${String(i+1).padStart(2,'0')}`,title:`名侦探柯南：${title}`,category:'anime',wiki:`Detective Conan: ${en}`}));
const bgmRows = `
honobono-log|温暖的印记|180670
maquia|朝花夕誓|218971
silent-voice|声之形|117777
shoshimin-1|小市民 第一季|474906
shoshimin-2|小市民 第二季|513018
nichijou|日常|9912
hyouka|冰菓|27364
kaguya-1|辉夜大小姐想让我告白 第一季|248175
kaguya-2|辉夜大小姐想让我告白？ 第二季|293049
kaguya-3|辉夜大小姐想让我告白－究极浪漫－|317613
kaguya-1-cn|辉夜大小姐想让我告白 第一季 · 中配版|248175
kaguya-2-cn|辉夜大小姐想让我告白？ 第二季 · 中配版|293049
kaguya-3-cn|辉夜大小姐想让我告白－究极浪漫－ · 中配版|317613
kaguya-first-kiss|辉夜大小姐想让我告白－初吻不会结束－|389474
frieren-1|葬送的芙莉莲 第一季|400602
tsuki-ga-kirei|月色真美|207573
conan-tv|名侦探柯南（TV）|899
three-body-animation|三体（艺画开天动画）|285437
tsuredure-children|徒然喜欢你|208754
swallowed-star|吞噬星空|244239
swallowed-star-blood|吞噬星空：血洛大陆|507795
swallowed-star-original|吞噬星空：决战原始星|577199
qin-1|秦时明月之百步飞剑|2473
qin-2|秦时明月之夜尽天明|2769
qin-3|秦时明月之诸子百家|6815
qin-4|秦时明月之万里长城|17045
qin-5|秦时明月之君临天下|86073
qin-6|秦时明月之沧海横流|231210
nine-songs|天行九歌|127448
k-on-1|轻音少女 第一季|1424
k-on-2|轻音少女 第二季|3774
violet-tv|紫罗兰永恒花园|183878
tatami-galaxy|四叠半神话大系|4019
tatami-time-machine|四叠半时光机布鲁斯|345980
city-animation|小城日常|514358`;
for(const line of bgmRows.trim().split('\n')) {
  const [slug,title,id]=line.split('|'); rows.push({slug,title,category:'anime',bgm:Number(id)});
}
for(const [slug,title,bgm] of [
['lvxing-1','侣行 第一季',584982],['lvxing-2','侣行 第二季',584984],['lvxing-3','侣行 第三季',584985],['our-lvxing-1','我们的侣行 第一季',584998],['earth-lvxing-3','地球之极·侣行 第三季',585006],['earth-lvxing-4','地球之极·侣行 第四季',585008],['earth-lvxing-5','地球之极·侣行 第五季',585010],['earth-lvxing-6','地球之极·侣行 第六季',585012]]) rows.push({slug,title,bgm,category:'documentary'});
rows.push(
 {slug:'mr-six',title:'老炮儿',category:'film'},
 {slug:'zhao-wen-dao',title:'朝闻道',category:'book'},
 {slug:'ball-lightning',title:'球状闪电',category:'book'}
);
export default rows;
