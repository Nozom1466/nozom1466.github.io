# Gallery 分类规则

每张照片使用四个字段，不再使用自由文本 tags。图片地址、说明和排序不变。

- subject：必填，且只选一个：architecture、landscape、animal、documentary、portrait、still-life、experimental。
- locations：城市名称列表，例如 ["上海"]；可以多个。无可靠资料时为 []，不把国家、公园或时区当作城市。
- color：可选一个主色，固定为 red、orange、yellow、green、blue、black、white。没有明确主色则为 ""，不强行将粉色归为红色。
- year：拍摄年份字符串，例如 "2025"。优先使用 EXIF DateTimeOriginal / XMP DateCreated，不使用导出日期。缺失或可疑值暂填 ""。

主题和颜色的唯一词表为 data/gallery_taxonomy.json；Hugo 构建会拒绝词表以外的值。界面从同一词表生成。人文类使用 Documentary，包括街头生活、人物与环境的关系；Portrait 专指人像。Landscape 包括天空、星空与自然场景；Experimental 包括抽象、反射及形式实验。

分类参考：[Adobe 摄影类型](https://www.adobe.com/creativecloud/photography/type.html)。这是本站精简词表，不是唯一的摄影分类标准。

## 本次迁移依据及待补信息

共 22 张。逐张查看画面，并读取 OneDrive 原图及图床文件的 EXIF/XMP。已确认年份 18 张。城市由现有标题和已知地点标签规范化：同济/嘉定 → 上海，瘦西湖 → 扬州，武侯祠 → 成都，pittsburgh → 匹兹堡。原图均未找到可用 GPS 或 City 字段。

以下四张年份保留为空：

- 大一暑假的月牙：图床文件没有拍摄日期。
- 从樱花树走向河岸旁：EXIF 为 2000-08-29，疑似相机时钟未设置，待确认。
- 嘉定图书馆顶层：同样记录为 2000-08-29，待确认。
- 黄石公园银河：只有处理日期，没有拍摄日期。

以下 11 张城市保留为空：冬天温暖的背影、描金的大地、向天空宣誓主权！、又是大雪、线条、京阪电车、车站、月亮、月亮月亮、失真、黄石公园银河。黄石是公园，不虚构所属城市。日本相关照片未凭线路、国家或时区猜测城市。

嘉定樱花为粉色，不在当前七色词表中，颜色留空。其余颜色结合原标签及画面重新标注。

## 筛选与检查

四维筛选取交集；地点列表中任一城市匹配即可。空字段有单独的“待补/未标”选项。支持重置、网址分享、刷新、前进后退；大图关闭后保留筛选条件。

本地构建到 .gallery-preview 后，运行 node scripts/test-gallery.mjs，验证筛选逻辑及生成的 22 张照片。
