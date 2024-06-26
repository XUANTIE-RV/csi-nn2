 [English](./README.md) | 简体中文

SHL 是 XuanTie 提供的一组针对玄铁 CPU 平台的神经网络库 API。抽象了各种常用的网络层的接口，并且提供一系列已优化的二进制库。

SHL 的特性：

- C 代码版本的参考实现。
- 提供玄铁系列 CPU 的汇编优化实现。
- 支持对称量化和非对称量化。
- 支持8位定点，16位定点和16位浮点等数据类型。
- 兼容 NCHW 和 NHWC 格式。
- 搭配 [HHB](https://www.yuque.com/za4k4z/oxlbxl) 实现代码自动调用。
- 覆盖 CPU，NPU 等不同体系结构。
- 附加异构参考实现。

SHL 提供了完成的接口声明和接口的参考实现，各个设备提供商可以依此针对性的完成各个接口的优化工作。

# 使用 SHL

- [SHL](https://csi-nn2.opensource.alibaba.com/)
- [SHL 接口和设计文档](https://www.yuque.com/za4k4z/isgz8o)
- [SHL 配套部署工具](https://www.yuque.com/za4k4z/oxlbxl)

# 安装

## 通过 PyPi 安装

SHL 的预编译库可以通过 PyPi 安装 hhb 时，一起安装。

```
pip3 install hhb
```

二进制库的安装目录在 /usr/local/lib/python3.6/dist-packages/tvm/install_nn2/

## 通过源码重新编译

以 Ubuntu 上编译 c906 优化为例。

编译 C906 需要用到 XuanTie RISC-V GCC， 从 OCC 下载 GCC 2.6 版本，解压并设置路径。

```
wget https://occ-oss-prod.oss-cn-hangzhou.aliyuncs.com/resource//1663142514282/Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1-20220906.tar.gz
tar xf Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1-20220906.tar.gz
export PATH=${PWD}/Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1/bin:$PATH
```

下载源码

```
git clone https://github.com/XUANTIE-RV/csi-nn2.git
```

编译 c906

```
cd csi-nn2
make nn2_c906
```

安装 c906

```
make install_nn2
```

# 快速上手示例

以玄铁 CPU C906 执行 mobilenetv1 为例，可以参考 example 中的示例，示例中以较简易的方式描述了如何调用 SHL 的接口。

编译命令如下：

```
cd example
make c906_m1_f16
```

完成后会生成 c906_mobilenetv1_f16.elf 文件。将其复制到带 C906 CPU 的开发板【比如 D1】后，执行：

```
./c906_mobilenetv1_f16.elf
```

NOTE: 原始 mobilenetv1 中每层 conv2d 后接一层 batch norm，示例中假设已经通过部署工具将其融合进 conv2d。关于如何使用部署工具融合 batch norm，以及生成对应的权重数值，可以参考 [HHB](https://www.yuque.com/za4k4z/oxlbxl) 的使用。

# 资源

- [XuanTie 芯片开放社区](https://occ.t-head.cn/)
- [SHL 应用在 MLPerf tiny](https://github.com/mlcommons/tiny_results_v0.7/tree/main/open/Alibaba)

# 致谢

SHL 参考、借鉴了下列项目：
- [Caffe](https://github.com/BVLC/caffe)
- [Tensorflow](https://github.com/tensorflow/tensorflow)
- [ncnn](https://github.com/Tencent/ncnn)
- [MNN](https://github.com/alibaba/MNN)
- [Tengine](https://github.com/OAID/Tengine)
- [CMSIS_5](https://github.com/ARM-software/CMSIS_5)
- [ONNX](https://github.com/onnx/onnx)
- [XNNPACK](https://github.com/google/XNNPACK)
