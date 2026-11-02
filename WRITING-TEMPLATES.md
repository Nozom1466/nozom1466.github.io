# 手机写作模板与字段速查

依据当前已实现的解析规则整理，更新于 2026-11-01。

## 1. 使用前先看

| 内容 | 在闪念中选择的原生标签 | 网站位置 |
| --- | --- | --- |
| 日常记录 | 不选标签，或选择 `threads` | 此刻 |
| 博客文章 | `blog` | 博客 |
| 书评、影评、动画或游戏评价 | `review` | 书馆 |
| 摄影照片 | 不使用闪念标签，仍走 OneDrive 上传 | 展廊 Gallery |

- `blog`、`review`、`threads` 是闪念里已有的标签，不是正文里手打的一行文字。不要同时选择 blog 和 review；同时选择时 review 优先。
- Blog / Review 第一非空行直接写标题或作品名，不用写 `title:`。
- 顶部字段放在标题之后、正文之前。每个字段单独一段，使用普通文字，不要设成标题、列表、引用或代码块。字段顺序不限，但同一字段不能重复。
- 下面代码框只用于展示模板；复制到闪念后作为普通文本使用，不要连同代码框标记一起复制。
- `[插入图片]` 等方括号内容是操作提示，要替换成实际附件，不要保留这些提示文字。
- 模板默认 `draft: true`，写完后改成 `draft: false`。**不写 draft 会默认发布。** Threads 不解析 draft。
- Blog / Review 约每 30 分钟自动检查并部署，可能延迟；网页刷新不会立即发布文章。Threads 在手机完成同步后，网页刷新直接检查更新。
- 当前 COS 同步目录公开可读，草稿、分类或网页隐藏不等于私密存储。不要放隐私内容。

## 2. 可复制模板

**A. Threads／此刻**

不要求标题，不需要填写任何字段。选择 threads 标签，或不选 blog / review 即可。

```text
今天的一点记录。

这里写第二段。

[插入图片]
```

图片、普通文字和链接可直接插入。`labels:`、`caption:`、`audio:`、`draft:` 在此刻中不会按 Blog / Review 的规则解析；音乐播放器暂不适用于此刻。

**B. Blog 文章**

原生标签：`blog`。

```text
文章标题
draft: true
labels: 随笔, 摄影
summary: 用一句话概括这篇文章。

这里开始写正文。

[插入图片]
caption: 这张图片的说明。

继续写下一段。
```

只想写最少内容时，保留标题、draft 和正文即可；labels、summary、配图和 caption 都可省略。Blog 不解析顶部的 `rating:`、`cover:`、`date:` 等 Review 专用字段。

**C. Review 书评**

原生标签：`review`，不需要另加 blog。

```text
经营未来
draft: true
labels: book
rating: 4.6
author: 李明博
summary: 从穷小子到总统。
cover:
[插入作品封面]

这里写读后感。

[插入图片]
caption: 正文配图的说明。
```

有译者时，在正文之前增加 `translator: 译者姓名`。

**D. Review 影评**

原生标签：`review`。

```text
Interstellar
draft: true
labels: film
rating: 4.9
director: Christopher Nolan
writer: Jonathan Nolan, Christopher Nolan
music: Hans Zimmer
summary: 关于星际航行、时间与亲情的故事。
cover:
[插入作品封面]

这里写观后感。
```

这里的 `music:` 只显示配乐作者。要插入可播放的音乐，在正文中使用下方的音乐模板。

**E. Review 动画评价**

原生标签：`review`。

```text
动画作品名
draft: true
labels: anime
rating: 4.5
director: 导演姓名
original: 原著作者
composer: 配乐作者
summary: 一句话评价。
cover:
[插入作品封面]

这里写评价正文。
```

**F. Review 游戏评价**

原生标签：`review`。

```text
游戏作品名
draft: true
labels: game
rating: 4.5
developer: 开发商
publisher: 发行商
summary: 一句话评价。
cover:
[插入作品封面]

这里写游玩体验。
```

Review 发布时需要作品名和作品封面，其他字段按需填写。不需要的字段整行删除，不要把 `rating:` 或 `draft:` 留空。

**G. 正文配图与 caption（Blog / Review 通用）**

```text
[插入图片]
caption: 第一张图片的说明。

[插入图片]
caption: 第二张图片的说明。
```

caption 放在对应图片后的下一非空行。连续插入多张图后只写一行 caption，只会配给最后一张。没有说明就不写 caption，不会出现占位文字。

**H. 带封面的音乐（Blog / Review 通用）**

在正文需要播放器的位置填写，顺序固定：曲名 → 作者（可选）→ 音频 → 音乐封面（可选）。

```text
audio: Cornfield Chase
artist: Hans Zimmer
[插入 MP3 附件]
audio_cover:
[插入专辑封面]
```

多首音乐重复上述组合。每个音频或封面单独一行，允许中间空行。

没有作者或封面时，使用最简格式：

```text
audio: 曲名
[插入 MP3 附件]
```

也可以直接使用现有 COS 音频与封面链接。以下使用站内已有《星际穿越》的资源：

```text
audio: Cornfield Chase
artist: Hans Zimmer
https://myblog-1316371247.cos.ap-shanghai.myqcloud.com/myblog/cornfield%20_chase.mp3
audio_cover: https://myblog-1316371247.cos.ap-shanghai.myqcloud.com/myblog/interstellar_ost.jpg
```

不要把 `audio:` 填成链接，它填写的是曲名。播放器不重复展示音乐封面，也不会把音乐字段计入自动摘要。

**I. Gallery 上传说明（发给 Codex，不是闪念自动解析模板）**

将新照片上传到 OneDrive 的 `Blog Inbox/Gallery`，等待同步到电脑，再发送：

```text
把 OneDrive Gallery 里的新照片上传并推送。
这一批都是黄石公园，拍摄年份是 2026。
不加标题或文字说明，保留原图。
```

每张地点、年份不同时，可按文件名列出：

```text
上传 OneDrive Gallery 的新照片：
DSC_0001.JPG：2026，黄石公园
DSC_0002.JPG：2025，上海
主题和颜色按画面分类，不加文字说明。
```

Gallery 仍由 Codex 读取照片、通过 PicGo 上传、更新分类并推送；没有启用 `#gallery` 笔记自动同步，也不需要将这些照片同步到 Threads。

## 3. 全部字段、限制与发布检查

**顶部字段**

| 字段 | 适用范围 | 含义与写法 |
| --- | --- | --- |
| 第一非空行 | Blog / Review | 标题或作品名，必填，不写 `title:` |
| `draft:` | Blog / Review | `true` 暂不发布，`false` 发布；不写默认发布 |
| `labels:` | Blog / Review | 逗号分隔，例如 `AI, 随笔`；支持中文逗号、顿号，重复项合并 |
| `summary:` | Blog / Review | 同一段内的一句话摘要；留空或不写时自动提取正文前约 100 字 |
| `rating:` | Review | 0–5 的数字，可有小数；不写则不设置评分 |
| `cover:` | Review | 下一行插入作品封面，或同一行填现有 COS 图片直链 |
| `date:` | Review | `YYYY-MM-DD`，例如 `2026-11-01`；不写时使用笔记创建时间 |
| `status:` | Review | `Reading`、`Completed`、`Not Started` |

Blog 的 labels 是文章标签；Review 的 labels 是书馆分类，常用 `book`、`film`、`anime`、`game`，省略时默认 book。闪念中其他原生标签不会自动变成 labels。

Review 不写 `cover:` 时，会取正文第一张非音乐封面图片作为作品封面，并从正文移除；建议显式填写，避免选错。发布时不能缺作品封面；`draft: true` 的未完成笔记可以暂缺附件。

**Review 创作者字段（均可选，写在正文前）**

| 字段 | 用途 |
| --- | --- |
| `author:` | 作者 |
| `director:` | 导演 |
| `screenplay:` / `writer:` | 编剧 |
| `original:` | 原著 |
| `translator:` | 译者 |
| `developer:` | 开发商 |
| `publisher:` | 出版／发行方 |
| `producer:` | 制作人／制片 |
| `music:` / `composer:` | 配乐作者，不是播放器 |
| `artwork:` | 美术 |
| `cinematographer:` | 摄影 |
| `character:` | 角色相关署名 |

字段按填写顺序生成署名。同一角色有多人时写在同一行，不要重复同一个字段。`music:` 和 `composer:` 都会生成配乐署名，通常选一个即可。

**正文字段（Blog / Review 通用）**

| 字段 | 位置与用途 |
| --- | --- |
| `caption:` | 配图后下一非空行；不显示关键字，只显示说明 |
| `audio:` | 音乐块第一行，填写曲名；下一行可选 artist，然后必须有音频附件或直链 |
| `artist:` | 紧接 audio 后，填写演唱者／作曲者等，可省略 |
| `audio_cover:` | 音频之后；下一行插入封面，或同一行填写 COS 图片直链，可省略 |

`cover:` 是 Review 作品封面，`audio_cover:` 是单首音乐封面，两者独立。

**Gallery 分类字段（由 Codex 整理，不必自己记英文）**

| 字段 | 当前规则 |
| --- | --- |
| `subject` | 建筑 architecture、风景 landscape、动物 animal、人文纪实 documentary、人像 portrait、静物 still-life、实验 experimental，选一个 |
| `locations` | 可多个；优先城市，允许明确指定的地点，如黄石公园；不猜测未知地点 |
| `color` | 可选一个：红 red、橙 orange、黄 yellow、绿 green、蓝 blue、黑 black、白 white；无明确主色留空 |
| `year` | 拍摄年份，优先照片拍摄元数据或你的明确说明，不用导出时间 |

照片不再添加标题、comments 或 caption。你只需说明地点、年份，主题和颜色可根据画面整理。

**支持范围与容易出错的地方**

- 英文字段不区分大小写，冒号支持 `:` 和 `：`。兼容 Blog 顶部 `标签：` 和 Blog / Review 配图后的 `图注：`；新内容统一使用英文关键字。
- 顶部字段遇到正文就停止解析。不要把 draft、labels、summary 等写到文章末尾，也不要在它们前面插入正文图片。
- 使用闪念的富文本按钮设置粗体、斜体、标题、简单列表和引用；不把笔记中的普通 Markdown 源码当作完整 Markdown 再执行，手写 Hugo shortcode 也不会执行。
- Blog / Review 配图支持 JPG、JPEG、PNG、WebP、GIF；音频支持 MP3、M4A、OGG、WAV。视频、公式、表格、代码块、待办列表等尚未完整适配，会停止转换，不会静默忽略。
- 音频直链与封面直链限现有闪念 COS 的 `SnapNote_Sync/resources/`、图床 COS 的 `myblog/`；使用 HTTPS 直链，不带查询参数或片段。音乐平台的分享页不是音频直链。
- 已用站内《星际穿越》的现有音频和封面验证 Blog 播放及暂停。手机附件的实际导出结构尚待真实样本验证；不同结构会报错，不会根据显示名称猜资源路径。
- 单独清理 COS 中仍被文章使用的附件，会导致配图或音频失效。未完成上传、附件格式错误或资源读取失败时，本次部署会停止，线上保留上次成功版本。
- 已发布 Blog / Review 改为 `draft: true` 后，下次成功同步会撤下；草稿不会出现在 Threads。移除分类标签后可能转入默认 Threads，因此未写完请用 draft，不要用“取消标签”代替草稿。
- Gallery 与闪念文章是两条不同流程；本文的 Gallery 消息只是发给 Codex 的上传说明，不是新增加的字段解析功能。

**发布前检查**

1. 原生标签选对：blog 或 review；第一行有标题。
2. 顶部字段在正文前，不需要的字段整行删掉。
3. Review 有作品封面；每首音乐的附件和音乐封面对应正确。
4. 写完将 `draft: true` 改为 `draft: false`，并确认闪念同步完成。
5. 等待 Blog / Review 定时部署；Threads 则在同步后刷新页面查看。
