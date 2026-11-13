# 书馆批量整理

本次条目、封面来源与创作信息保存在 `docs/library-catalog.json`。`content/library/` 是网站内容源，之后的个人评价直接写入对应文件；本次只创建作品资料，不生成评分或影评。

## 校验与维护

在本目录执行 `npm ci` 安装依赖，然后从仓库根目录执行：

```sh
node scripts/library-catalog/check.mjs
node scripts/library-catalog/check.mjs <Hugo输出目录>
```

`works.mjs` 是本次选定作品列表。研究脚本将外部资料保存在被忽略的 `cache/`；`prepare.mjs` 从这些缓存生成来源清单，`import.mjs` 仅创建不存在的文件，绝不覆盖已有评价。重新研究时需要人工核对版本、人员角色和封面，不能只凭搜索排名自动选取。

封面来自官方作品宣传图、发行封面及收录这些图片的资料站；来源和原图链接逐项保留在 JSON 中，并非全部直接托管于官方站点。显示用图片转为 WebP，最大 640×960，不放大小图，不裁切海报，不生成替代封面。

`date` 是收录日期，`release_year` 是资料中查到的作品发行年份。尚不确定的作品保留在整理文档中，不创建正式条目。
