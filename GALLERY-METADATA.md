# Gallery 分类规则

每张照片使用四个字段，不再使用自由文本 tags。图片地址、说明和排序不变。

- subject：必填，且只选一个：architecture、landscape、animal、documentary、portrait、still-life、experimental。
- locations：城市名称列表，例如 ["上海"]；可以多个。无可靠资料时为 []，不把国家、公园或时区当作城市。
- color：可选一个主色，固定为 red、orange、yellow、green、blue、black、white。没有明确主色则为 ""，不强行将粉色归为红色。
- year：拍摄年份字符串，例如 "2025"。优先使用 EXIF DateTimeOriginal / XMP DateCreated，不使用导出日期。缺失或可疑值暂填 ""。

主题和颜色的唯一词表为 data/gallery_taxonomy.json；Hugo 构建会拒绝词表以外的值。界面从同一词表生成。人文类使用 Documentary，包括街头生活、人物与环境的关系；Portrait 专指人像。Landscape 包括天空、星空与自然场景；Experimental 包括抽象、反射及形式实验。

分类参考：[Adobe 摄影类型](https://www.adobe.com/creativecloud/photography/type.html)。这是本站精简词表，不是唯一的摄影分类标准。

## 本次迁移依据及待补信息

共 22 张。逐张查看画面，并读取 OneDrive 原图及图床文件的 EXIF/XMP。18 张年份来自元数据，另外 4 张由用户确认，现已全部填写。城市由现有标题和已知地点标签规范化：同济/嘉定 → 上海，瘦西湖 → 扬州，武侯祠 → 成都，pittsburgh → 匹兹堡。原图均未找到可用 GPS 或 City 字段。

2026-10-30，用户按截图从左到右确认以下年份；用户确认优先于缺失或错误的 EXIF：

- 嘉定图书馆顶层：2024（原 EXIF 为 2000-08-29）。
- 黄石公园银河：2026（原文件没有拍摄日期）。
- 从樱花树走向河岸旁：2024（原 EXIF 为 2000-08-29）。
- 大一暑假的月牙：2022（原文件没有拍摄日期）。

以下 11 张城市保留为空：冬天温暖的背影、描金的大地、向天空宣誓主权！、又是大雪、线条、京阪电车、车站、月亮、月亮月亮、失真、黄石公园银河。黄石是公园，不虚构所属城市。日本相关照片未凭线路、国家或时区猜测城市。

嘉定樱花为粉色，不在当前七色词表中，颜色留空。其余颜色结合原标签及画面重新标注。

## 筛选与检查

四维筛选取交集；地点列表中任一城市匹配即可。空字段有单独的“待补/未标”选项。支持重置、网址分享、刷新、前进后退；大图关闭后保留筛选条件。

本地构建到 .gallery-preview 后，运行 node scripts/test-gallery.mjs，验证筛选逻辑及照片数量。也可通过首个参数指定构建后的 gallery/index.html 路径；数量与源文件自动核对。

## 2026-10-31 新增

从 OneDrive 的 Blog Inbox/Gallery 添加两张原图，共 24 张；通过 PicGo 上传，未再次压缩。

- DSC_6434.jpg：XMP 标题“银河”，主题 landscape，颜色 black，城市留空。EXIF 为 2025-01-01 08:00:05，疑似初始时间；用户确认实际拍摄年份为 2026，以用户确认为准。
- DSC_8005.jpg：XMP 标题“日照金山”，主题 landscape，颜色 blue，城市留空。EXIF 拍摄时间为 2026-10-14，年份 2026。

## 2026-11-01 新增

从 OneDrive 的 Blog Inbox/Gallery 添加 5 张原图，共 29 张，经 PicGo 上传，未额外压缩。均无标题、说明、城市和 GPS 元数据，标题与城市留空。年份取 EXIF DateTimeOriginal，与 XMP 拍摄时间一致；不使用 2026-11-01 的导出时间。主题、颜色根据画面填写：

- DSC_4091.JPG：雪地红衣行人，documentary / red，拍摄时间 2025-12-15，年份 2025。
- DSC_8869.JPG：雾中持三脚架的人像，portrait，画面偏灰紫，颜色留空，拍摄时间 2026-10-14。
- DSC_8911.JPG：树枝上的黑色鸟，animal / black，拍摄时间 2026-10-14。
- DSC_9405.JPG：暮色中的三人剪影，documentary / black，拍摄时间 2026-10-15。
- DSCF4067.JPG：雪山与水面倒影，landscape / blue，拍摄时间 2026-10-15。
