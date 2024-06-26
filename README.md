English | [简体中文](./README_CN.md)

SHL(Structure of Heterogeneous Library, Chinese name: ShiHulan) is a high-performance Heterogeneous computing library provided by XuanTie.
The interface of SHL uses XuanTie neural network library API for XuanTie CPU platform: CSI-NN2, and provides a series of optimized binary libraries.

Features for SHL:

- Reference implementation of c code version
- Assembly optimization implementation for XuanTie CPU
- Supports symmetric quantization and asymmetric quantization
- Support 8bit, 16bit, and f16 data types
- compaatible with NCHW and NHWC formates
- Use [HHB](https://www.yuque.com/za4k4z/kvkcoh) to automatically call API
- Covers different architectures, such as CPU and NPU
- Reference heterogeneous schedule implementation

In principle, SHL only provides the reference implementation of XuanTie CPU platform, and the optimization of each NPU target platform is completed by the vendor of the specific platform.

# Use SHL

- [SHL](https://csi-nn2.opensource.alibaba.com/)
- [SHL API](https://www.yuque.com/za4k4z/kkzsw9)
- [SHL deployment tools](https://www.yuque.com/za4k4z/kvkcoh)

# Installation

## Official Python packages

SHL released packages are published in PyPi, can install with hhb.

```
pip3 install hhb
```

binary libary is at /usr/local/lib/python3.6/dist-packages/tvm/install_nn2/

## Build SHL from Source

Here is one example to build C906 library.

We need to install XuanTie RISC-V GCC 2.6, which can get from XuanTie OCC, download, decompress, and set path environment.

```
wget https://occ-oss-prod.oss-cn-hangzhou.aliyuncs.com/resource//1663142514282/Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1-20220906.tar.gz
tar xf Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1-20220906.tar.gz
export PATH=${PWD}/Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1/bin:$PATH
```

Download source code

```
git clone https://github.com/XUANTIE-RV/csi-nn2.git
```

compile c906

```
cd csi-nn2
make nn2_c906
```

install c906

```
make install_nn2
```

# Quick Start Example

Here is one example for XuanTie C906 to run mobilenetv1. It shows how to call SHL API to inference the whole model.

compile command:

```
cd example
make c906_m1_f16
```

c906_mobilenetv1_f16.elf will be generated after completion.
After copying it to the development board with C906 CPU [such as D1], execute:

```
./c906_mobilenetv1_f16.elf
```

NOTE: Original mobilenetv1's every conv2d has one BN(batch norm), but the example assumes BN had been fused into conv2d。About how to use deployment tools to fuse BN, and emit right weight float16 value, can reference [HHB](https://www.yuque.com/za4k4z/kvkcoh).

# Resources

- [XuanTie Open Chip Community](https://xrvm.com/)
- [Use SHL to run MLPerf tiny](https://github.com/mlcommons/tiny_results_v0.7/tree/main/open/Alibaba)

## Acknowledgement

SHL refers to the following projects:

- [Caffe](https://github.com/BVLC/caffe)
- [Tensorflow](https://github.com/tensorflow/tensorflow)
- [ncnn](https://github.com/Tencent/ncnn)
- [MNN](https://github.com/alibaba/MNN)
- [Tengine](https://github.com/OAID/Tengine)
- [CMSIS_5](https://github.com/ARM-software/CMSIS_5)
- [ONNX](https://github.com/onnx/onnx)
- [XNNPACK](https://github.com/google/XNNPACK)
