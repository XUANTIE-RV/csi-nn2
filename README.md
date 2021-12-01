## 简介

CSI-NN2 是 T-HEAD 提供的一组针对无剑 SoC 平台的神经网络库 API。抽象了各种常用的网络层的接口，并且提供一系列已优化的二进制库。

CSI-NN2 的特性：

- 开源 c 代码版本的参考实现。
- 提供玄铁 CPU 的汇编优化实现。
- 支持 i8/u8 数据类型的对称和非对称量化。
- 兼容 NCHW 和 NHWC 格式。
- 搭配 HHB 实现代码自动调用。
- 覆盖 CPU，GPU, NPU 体系结构。
- 附加一些接口辅助使用，作为参考实现。

CSI-NN2 只提供接口声明和接口的参考实现，对各个接口的优化工作交由各个设备提供商完成。

## 文档说明

- [中文文档](https://www.yuque.com/za4k4z/isgz8o)

## 致谢

CSI-NN2 参考、借鉴了下列项目：
- [Caffe](https://github.com/BVLC/caffe)
- [Tensorflow](https://github.com/tensorflow/tensorflow)
- [ncnn](https://github.com/Tencent/ncnn)
- [MNN](https://github.com/alibaba/MNN)
