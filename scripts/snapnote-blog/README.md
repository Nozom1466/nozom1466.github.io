# 闪念博客：自动发布

`blog` 与 `threads` 标签 ID 在 `static/moments/channels.mjs` 中配置，依据用户截图和同步笔记确认。双标签时 blog 优先；没有 blog 标签的笔记默认进入此刻，不必显式添加 threads（包括未分类或仅有其他标签的记录）。归档、锁定、删除的笔记仍不展示。标签改名不影响 ID，删除重建标签需重新确认。

`blog` 标签即发布，不需要额外的「发布」标签；`threads` 只在此刻展示。移除 blog、归档、锁定或删除笔记会在下次成功同步时撤下对应文章。不确定分类的加密记录、读取失败、图片缺失会停止部署，保留线上版本；已发布文章完全消失于索引（无删除记录）也会停止自动撤稿，需要检查同步完整性。

## 手机写作约定

第一非空行作为标题。标题后可写一行 `labels: Blender, AI, 随笔`，中英文逗号、顿号均可，重复标签自动合并。这一行不会出现在正文；闪念其他原生标签暂不映射为文章标签，因为公开同步数据中只有标签 ID。

在图片后下一非空行写 `caption: 教师办公室的参考图`。说明显示在图片下方，也作为替代文字；连续多张图片后的一行图注只属于紧邻的最后一张。没有图注不显示占位文字。普通正文中的同类文字不自动改成图注。英文关键字不区分大小写；旧写法 `标签：`、`图注：` 保留兼容。

正文、段落、粗体、斜体、链接、标题、简单列表、引用及 COS 原图会保留。尚不支持的附件、表格等停止转换，不静默丢弃。图片当前引用闪念 COS 资源，不额外上传 PicGo。删除云端附件会影响旧文章，不应单独清理仍在使用的图片。

## 同步和部署

执行 `node scripts/snapnote-blog/sync.mjs` 从最新索引生成全部 blog 文章到 `content/ideas/snapnote-<UUID>.md`，以 UUID 固定文章地址。只保存展示字段，不保存设备或 OCR 信息。读取、转换及配图检查全部成功后才写入生成文件；不会覆盖没有同步标记的手写文章。线上更新不会自动提交文章到 Git；COS 是闪念文章的编辑来源，每次构建重新生成。

GitHub Actions 每小时第 7、37 分钟检查（约每 30 分钟），仅当文章摘要哈希或代码版本与已部署的 `snapnote-version.json` 不同时编译部署。推送代码和手动运行始终完整构建。版本标记随站点一起部署，不提前更新，部署失败会在下次重试。网页刷新不会触发 Action，也不会即时发布博客。Threads 仍直接读取 COS。

GitHub 定时任务可能延迟；公开仓库超过 60 天无活动时可能被自动停用，需在 Actions 中重新启用。参考 https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule 。

不需要新增密钥或腾讯云配置。COS 的正文、索引和附件仍按现有策略公开，分类和撤稿不是隐私保护。

## 本地预览与测试

单篇预览仍可用 `node scripts/snapnote-blog/preview.mjs <笔记 UUID>`，生成 `draft: true`，配合 Hugo `--buildDrafts`。正式同步会覆盖为 `draft: false`。`static/snapnote-version.json` 是构建生成文件，不手动编辑或提交。

测试：`node --test scripts/snapnote-blog/*.test.mjs scripts/moments-cos.test.mjs`。
