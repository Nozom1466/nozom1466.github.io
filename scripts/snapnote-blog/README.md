# 闪念博客：自动发布

手机写作可直接参考根目录的 [写作模板与字段速查](../../WRITING-TEMPLATES.md)，包含 Blog、Review、Threads、音乐和 Gallery 上传说明。

`review`、`blog` 与 `threads` 标签 ID 在 `static/moments/channels.mjs` 中配置，依据用户截图和同步笔记确认。分类优先级为 review > blog > threads；没有 review 或 blog 标签的笔记默认进入此刻，不必显式添加 threads（包括未分类或仅有其他标签的记录）。归档、锁定、删除的笔记仍不展示。标签改名不影响 ID，删除重建标签需重新确认。

`blog` 标签的笔记默认发布，不需要额外的「发布」标签；标题后填写 `draft: true` 则不发布，写完改为 `draft: false`。未填写 draft 时保留原有发布行为。Blog 草稿不会进入 Threads。已发布文章改为草稿、移除 blog、归档、锁定或删除后，下次成功同步会撤下对应文章。不确定分类的加密记录、读取失败、已发布内容的图片缺失会停止部署，保留线上版本；已发布文章完全消失于索引（无删除记录）也会停止自动撤稿，需要检查同步完整性。

## 手机写作约定

第一非空行作为标题。标题后可写一行 `labels: Blender, AI, 随笔`，中英文逗号、顿号均可，重复标签自动合并。这一行不会出现在正文；闪念其他原生标签暂不映射为文章标签，因为公开同步数据中只有标签 ID。

标题后还可写 `summary: 这里是一句话摘要`，与 labels 顺序不限，允许中间空行；遇到正文后停止识别元数据。summary 内容写在同一段内（屏幕自动换行不影响），不出现在正文，不自动截断；留空或不写时继续从正文生成摘要。关键字不区分大小写，同一字段不能重复。示例：

```text
关于 codex + blender 建模的碎碎念
draft: true
labels: Blender, AI, 随笔
summary: 一个月北宇治复原计划中的建模体验与局限。

这里开始写正文。
```

`draft:` 和 labels、summary 一样放在标题后、正文前，顺序不限；值只接受 true/false，不区分大小写，空值、拼写错误或重复字段会停止同步，避免误发布。草稿在正文转换和图片检查前排除；仍需要第一行作品名称。此字段控制 blog 和 review，Threads 不解析它。草稿仍存储在当前公开读取的 COS 中，不是私密存储。

在图片后下一非空行写 `caption: 教师办公室的参考图`。说明显示在图片下方，也作为替代文字；连续多张图片后的一行图注只属于紧邻的最后一张。没有图注不显示占位文字。普通正文中的同类文字不自动改成图注。英文关键字不区分大小写；旧写法 `标签：`、`图注：` 保留兼容。

正文、段落、粗体、斜体、链接、标题、简单列表、引用及 COS 原图会保留。尚不支持的附件、表格等停止转换，不静默丢弃。图片当前引用闪念 COS 资源，不额外上传 PicGo。删除云端附件会影响旧文章，不应单独清理仍在使用的图片。

## 同步和部署

### 独立书馆评价

在闪念选择原生 `review` 标签，不需要添加 blog。第一非空行写书名或片名（普通文字或标题均可），之后写元数据，再写正文。例如：

```text
经营未来
draft: false
rating: 4.6
author: 李明博
labels: book
summary: 从穷小子到总统
cover:
```

在 `cover:` 下一行直接粘贴封面，后面继续写正文。也可填写现有两个 COS 空间的图片 HTTPS 链接；未指定 cover 时取第一张图片作为封面。封面不重复出现在正文里。正文配图仍支持下一行 `caption:`。

`rating` 为 0–5；`labels` 对应书馆筛选标签，省略时为 book（常用 book、film、anime、game）。`summary`、`draft` 行为同博客；`date` 可指定 YYYY-MM-DD，`status` 支持 Reading、Completed、Not Started。创作者字段支持 author、director、screenplay、writer、original、translator、developer、publisher、producer、music、composer、artwork、cinematographer、character。`music:` / `composer:` 只是配乐作者署名，播放器使用下方独立的 `audio:` 格式。视频附件仍不支持自动转换，原有手写评价与音乐 shortcode 不受影响。

### Blog / Review 正文音乐

在正文需要播放器的位置按顺序填写（方括号是插入附件的操作提示，不要输入）：

```text
audio: Cornfield Chase
artist: Hans Zimmer
[插入 MP3 附件]
audio_cover:
[插入专辑封面图片]
```

`audio` 填曲名，必填；`artist` 和 `audio_cover` 可省略。音频及图片各占一行，允许空行。多首音乐重复这一组。字段不区分大小写，支持中英文冒号，请用普通段落而非代码块或列表。音频支持 MP3/M4A/OGG/WAV；也可以在附件位置单独粘贴现有 COS 音频直链，音乐封面可写成 `audio_cover: https://…/album.jpg`。不支持 Spotify / Apple Music 等分享页代替音频文件。

每组转换为带封面的音乐播放器，不重复展示附件，不进入自动摘要。Blog 与 Review 使用同一格式，可插入多首；Threads 不作同样转换。Review 的 `cover:` 是作品封面，`audio_cover:` 是歌曲封面，不能互相替代；Review 推荐显式写 `cover:`，Blog 无需此字段。音频/封面缺失、格式不支持或同步未完成会停止部署，保留线上版本；写作时请先用 `draft: true`。Blog 仅在有音乐时加载播放器，不自动播放；离开文章时停止播放。播放器加载失败时保留浏览器原生音频控件。

附件解析支持 audio 文件名、file/attachment 的资源路径对象以及 COS 直链；若客户端导出的附件结构不同，会报错而不是猜测资源。首次使用应以真实手机附件验证。

review 生成到 `content/library/snapnote-<UUID>.md`，不进入博客首页或此刻。与 blog 使用同一自动同步任务，修改、撤稿和草稿规则相同；不会修改原有手写书评。

执行 `node scripts/snapnote-blog/sync.mjs` 从最新索引生成全部 blog 文章到 `content/ideas/snapnote-<UUID>.md`，以 UUID 固定文章地址。只保存展示字段，不保存设备或 OCR 信息。读取、转换及配图检查全部成功后才写入生成文件；不会覆盖没有同步标记的手写文章。线上更新不会自动提交文章到 Git；COS 是闪念文章的编辑来源，每次构建重新生成。

GitHub Actions 每小时第 7、37 分钟检查（约每 30 分钟），仅当文章摘要哈希或代码版本与已部署的 `snapnote-version.json` 不同时编译部署。推送代码和手动运行始终完整构建。版本标记随站点一起部署，不提前更新，部署失败会在下次重试。网页刷新不会触发 Action，也不会即时发布博客。Threads 仍直接读取 COS。

GitHub 定时任务可能延迟；公开仓库超过 60 天无活动时可能被自动停用，需在 Actions 中重新启用。参考 https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule 。

不需要新增密钥或腾讯云配置。COS 的正文、索引和附件仍按现有策略公开，分类和撤稿不是隐私保护。

## 本地预览与测试

单篇预览仍可用 `node scripts/snapnote-blog/preview.mjs <笔记 UUID>`，生成 `draft: true`，配合 Hugo `--buildDrafts`。正式同步会覆盖为 `draft: false`。`static/snapnote-version.json` 是构建生成文件，不手动编辑或提交。

测试：`node --test scripts/snapnote-blog/*.test.mjs scripts/moments-cos.test.mjs`。
