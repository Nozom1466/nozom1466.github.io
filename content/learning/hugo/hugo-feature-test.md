---
title: 'Hugo Feature Test'
date: 2024-09-22T15:08:10+08:00
slug:
summary:
description:
cover:
    image:
    alt:
    caption:
    relative: false
showtoc: false
draft: false
tags: ['hugo']
categories:
---

### Text
This is a plain text. Text with **Stress**, *Italic*, ~~Del line~~. Text with `inline item`.

<abbr title="Graphics Interchange Format">GIF</abbr> is a bitmap image format.

H<sub>2</sub>O

X<sup>n</sup> + Y<sup>n</sup> = Z<sup>n</sup>

Press <kbd><kbd>CTRL</kbd>+<kbd>ALT</kbd>+<kbd>Delete</kbd></kbd> to end the session.



### Katex
Inline `MathJax`: @\sum_{i}a_i@

Block `MathJax`:
$$
    \int_{a}^{b}f(t)\text{d}t
$$

`\mathbb`, `\mathcal` test
$$
    \mathbb{ABCDEFG}\mathcal{ABCDEFG}
$$

Environment test
$$
    \begin{aligned}
        A &= aaa \\
          &= bbb \rightarrow
    \end{aligned}
$$



### Coding
Test for `Python`
```Python
def test(param):
    """Test for hugo website
    Args:
        param: parameter of function

    Return:
        type of param
    """
    return type(param)
```

Test for `C++`
```c++
#include <iostream>
using namespace std;

int test(int param){
    // Test comment
    return true;
}
```

line number:
```html {linenos=true}
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Example HTML5 Document</title>
    <meta
      name="description"
      content="Sample article showcasing basic Markdown syntax and formatting for HTML elements."
    />
  </head>
  <body>
    <p>Test</p>
  </body>
</html>
```

highlights
```html {linenos=true,hl_lines=[2,8]}
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Example HTML5 Document</title>
    <meta
      name="description"
      content="Sample article showcasing basic Markdown syntax and formatting for HTML elements."
    />
  </head>
  <body>
    <p>Test</p>
  </body>
</html>
```



### Image
![Love-Letter](/Love-Letter-岩井俊二/cover.jpg "alt")

### Tables
|A|B|C|D|
|---:|:---|---:|:---:|
|a|b|c|d|

