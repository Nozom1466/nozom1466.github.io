import {readFile,writeFile} from 'node:fs/promises';
const catalog=JSON.parse(await readFile(new URL('../../docs/library-catalog.json',import.meta.url)));
const names={anime:'动画',film:'电影',book:'书籍',tv:'电视剧',documentary:'纪录片',variety:'综艺',game:'游戏'};
const lines=['# 书馆整理结果','','整理日期：2026-11-13。新增 '+catalog.length+' 个作品条目，保留原有作品和评价。新增条目均未填写评分、摘要或评价正文。','',
'封面及资料逐条来源见 [library-catalog.json](library-catalog.json)。封面优先使用作品宣传图、正式发行封面；部分通过资料站取得，不包含生成式封面。','',
'## 已创建条目',''];
for(const [category,label] of Object.entries(names)){
 const group=catalog.filter(x=>x.category===category);
 lines.push(`### ${label}（${group.length}）`,'');
 for(const c of group)lines.push(`- [${c.title}](../content/library/${c.slug}.md)${c.releaseYear?' · '+c.releaseYear:''}`);
 lines.push('');
}
lines.push('## 暂未收录','',
'- 提到雷佳音的“大明王朝／帝国”相关电影：用户要求暂缓。',
'- 《辉夜大小姐》OVA 与特别篇、《秦时明月》其他番外：没有明确确认，不自动扩展。','',
'## 版本与去重','',
'- 辉夜中配版不单独收录，保留原版三季及《初吻不会结束》。',
'- 《侣行》合并为一个系列条目：原《侣行》第一至三季、《我们的侣行》第一季、《地球之极·侣行》第三至六季；沿用第一季封面，不增加未确认的观看范围。',
'- 《秦时明月》第一至六部合并为一个系列条目，沿用第一部封面；《天行九歌》单独保留。系列年份表示开始年份。',
'- 用户通过封面确认周国平的《守望的距离》，采用浙江人民出版社 2017 年平装版（ISBN 9787213079634）。',
'- 用户纠正“Game of War”为 Gameloft 那款游戏；不收录 Game of War，保留已确认的《闪电部队》，不重复创建。',
'- 用户确认《老炮儿》并追加《球状闪电》；刘慈欣短篇集按《朝闻道》收录，选用此前给出的长江文艺版资料。',
'- 《敢死队》只收录第一、二部；《闪电部队》按 Blitz Brigade 记录。',
'- 《星球大战》只收录第七、八、九部。',
'- 《名侦探柯南》包括 TV 和用户指定的第 1—29 部剧场版。',
'- 《葬送的芙莉莲》仅第一季；《哪吒》仅《魔童降世》；《沙丘》仅 2021 年电影。',
'- 《她和她的猫》采用新海诚原作短片；《乔西的虎与鱼》采用动画电影。',
'- 《情书》《玉子爱情故事》《乒乓》等已有条目不重复创建。',
'- 《吹响！上低音号》小说保留已有总评，不拆分各卷和番外。','');
await writeFile(new URL('../../docs/LIBRARY-CATALOG.md',import.meta.url),lines.join('\n'));
console.log('Documented '+catalog.length+' entries.');
