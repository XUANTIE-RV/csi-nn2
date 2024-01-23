#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
from torch import tensor
from torch.nn import functional as fn

def convolution_f32(test_type):
    para = []
    # init the input data and parameters
    batch       = int(np.random.randint(1, high=4, size=1))
    in_size_x   = int(np.random.randint(64, high=128, size=1))
    in_channel  = int(np.random.randint(1, high=64, size=1))
    stride_x    = int(np.random.randint(1, high=3, size=1))
    kernel_x    = int(np.random.randint(stride_x + 1, high=7, size=1))
    dilation_x  = int(np.random.randint(1, high=5, size=1))

    if test_type == "conv1s1":
        stride_x    = 1
        kernel_x    = 1
        dilation_x  = 1

    kernel_x_t  = kernel_x + (kernel_x - 1) * (dilation_x - 1)
    pad_left    = pad_right = 0

    pad_x      = (in_size_x - kernel_x_t) -  int((in_size_x - kernel_x_t) / stride_x) * stride_x
    if(pad_x !=0):
        pad_x      = int((in_size_x - kernel_x_t) / stride_x) * stride_x + stride_x - (in_size_x - kernel_x_t)
        pad_left   = int(np.random.randint(0, high=pad_x, size=1))
        pad_right  = pad_x - pad_left

    out_channel = int(np.random.randint(1, high=64, size=1))
    zero_point1 = int(np.random.randint(-3, high=3, size=1))
    std1        = int(np.random.randint(1, high=3, size=1))
    zero_point2 = int(np.random.randint(-3, high=3, size=1))
    std2        = int(np.random.randint(1, high=3, size=1))
    zero_point3 = int(np.random.randint(-6, high=6, size=1))
    std3        = int(np.random.randint(1, high=10, size=1))

    src_in = np.random.normal(zero_point1, std1, (batch, in_channel, in_size_x))
    weight = np.random.normal(zero_point2, std2, (out_channel, in_channel, kernel_x))
    bias   = np.random.normal(zero_point3, std3, out_channel)
    src_in = src_in.astype(np.float32)
    weight = weight.astype(np.float32)
    bias   = bias.astype(np.float32)

    t_src_in  = tensor(src_in)
    t_weight  = tensor(weight)
    t_bias    = tensor(bias)

    t_src_in  = fn.pad(t_src_in, (pad_left, pad_right), 'constant', 0)
    t_src_out1 = fn.conv1d(t_src_in, t_weight, bias=t_bias, stride=stride_x, dilation=dilation_x).numpy()

    out_size_x = np.shape(t_src_out1)[2]

    src_in_1   = src_in.flatten()
    weight_1   = weight.flatten()
    src_out_1  = t_src_out1.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + len(weight_1) + len(bias) + 17

    para.append(total_size)
    para.append(batch)
    para.append(in_channel)
    para.append(in_size_x)  #width
    para.append(stride_x)
    para.append(kernel_x)
    para.append(pad_left)
    para.append(pad_right)
    para.append(out_channel)
    para.append(dilation_x)
    para.append(out_size_x)
    print(para)

    with open("convolution1d_ncw_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(weight_1)), *weight_1)
        fp.write(data)
        data = struct.pack(('%df' % len(bias)), *bias)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    test_type = sys.argv[1]
    convolution_f32(test_type)
    print("end")
