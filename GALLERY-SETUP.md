# Gallery - 组件借用方式

## 架构说明

采用**组件借用**方式集成 hugo-theme-gallery：
- hugo-theme-gallery 作为 git submodule 位于 `themes/hugo-theme-gallery/`
- 手动复制关键组件到主站点（layouts、assets）
- 主题仍然是 hugo-theme-next
- Gallery 功能通过复制的组件实现

## 文件结构

```
nozom1466.github.io/
├── themes/
│   ├── hugo-theme-next/           # 主题
│   └── hugo-theme-gallery/        # Git submodule (仅参考)
├── layouts/
│   ├── partials/
│   │   ├── gallery.html           # 复制自 hugo-theme-gallery
│   │   └── get-gallery.html       # 复制自 hugo-theme-gallery
│   └── gallery/
│       └── single.html            # Gallery 页面模板
├── assets/
│   └── gallery-theme/             # 复制的 CSS/JS
└── content/
    └── gallery/
        ├── _index.md              # Gallery 首页
        └── sample-album/
            └── index.md           # 示例相册
```

## 使用方法

### 1. 添加相册

在 `content/gallery/` 创建新文件夹：

```bash
mkdir content/gallery/my-album
```

### 2. 创建 index.md

```markdown
---
title: 我的相册
date: 2024-12-17
description: 相册描述
categories: ["旅行", "风景"]
resources:
  - src: photo1.jpg
    title: "照片标题"
    params:
      cover: true  # 用作相册封面
---

相册详细说明（支持 Markdown）
```

### 3. 添加照片

将图片放在同一文件夹：
```
content/gallery/my-album/
├── index.md
├── photo1.jpg
├── photo2.jpg
└── photo3.jpg
```

### 4. 构建和预览

```bash
hugo server
```

访问：http://localhost:1313/gallery/

## 配置

在 `hugo.yaml` 中配置：

```yaml
params:
  gallery:
    boxSpacing: 8              # 图片间距
    targetRowHeight: 300       # 行高度
    targetRowHeightTolerance: 0.25  # 高度容差
```

## 更新 hugo-theme-gallery

因为是 submodule，可以轻松更新：

```bash
cd themes/hugo-theme-gallery
git pull origin main
cd ../..
```

更新后需要重新复制修改的文件：

```powershell
# 复制 layouts
Copy-Item themes\hugo-theme-gallery\layouts\partials\gallery.html layouts\partials\ -Force
Copy-Item themes\hugo-theme-gallery\layouts\partials\get-gallery.html layouts\partials\ -Force

# 如果 gallery theme 有 CSS/JS 更新
Copy-Item themes\hugo-theme-gallery\assets\* assets\gallery-theme\ -Recurse -Force
```

## 优点

- ✅ 单一站点，使用 hugo server 即可
- ✅ hugo-theme-gallery 作为 submodule，易于更新
- ✅ 保持 hugo-theme-next 主题不变
- ✅ 只复制需要的组件

## 注意事项

- ⚠️ 更新 hugo-theme-gallery 后需手动重新复制文件
- ⚠️ 样式可能需要调整以适配 hugo-theme-next
- ⚠️ 功能可能不如完整的 hugo-theme-gallery 主题

## 文件说明

**复制的关键文件**：
- `layouts/partials/gallery.html` - Gallery 瀑布流布局
- `layouts/partials/get-gallery.html` - Gallery 数据获取
- `layouts/gallery/single.html` - Gallery 页面模板
- `assets/gallery-theme/` - CSS、JS、PhotoSwipe 等资源

**内容结构**：
- `content/gallery/_index.md` - Gallery 首页
- `content/gallery/*/index.md` - 各个相册

---

**参考**: [hugo-theme-gallery](https://github.com/nicokaiser/hugo-theme-gallery)
