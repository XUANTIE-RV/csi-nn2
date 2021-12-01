## 简介

aie用于对接AiEngine。

### 1.支持的特性

- 加载模型
- 卸载模型
- 获取输入输出tensor
- 设置输入输出tensor
- 执行模型

### 2. 如何使用

1) 使用HHB工具生成模型文件：model.c model.params
2) 使用修改makefile：替换csinn2库的路径和npu driver的路径
3）make， 得到libaie.so