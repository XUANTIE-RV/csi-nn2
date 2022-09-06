#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
from torch import tensor
from torch.nn import functional as fn

def group_convolution_f32(test_type):
    para = []
    # init the input data and parameters
    batch      = int(np.random.randint(1, high=4, size=1))
    in_size_x  = int(np.random.randint(32, high=33, size=1))
    in_size_y  = int(np.random.randint(32, high=33, size=1))
    in_channel = int(np.random.randint(8, high=16, size=1))
    # init the input data and parameters
    if test_type == "random":
        stride_x   = int(np.random.randint(1, high=3, size=1))
        stride_y   = int(np.random.randint(1, high=3, size=1))
        kernel_x   = int(np.random.randint(stride_x + 1, high=7, size=1))
        kernel_y   = int(np.random.randint(stride_y + 1, high=7, size=1))
        dilation_x = int(np.random.randint(1, high=5, size=1))
        dilation_y = int(np.random.randint(1, high=5, size=1))
    elif test_type == "conv3x3s1d1":
        stride_x    = 1
        stride_y    = 1
        kernel_x    = 3
        kernel_y    = 3
        dilation_x  = 1
        dilation_y  = 1
    
    group      = int(np.random.randint(2, high=7, size=1))
    in_channel = int(in_channel / group) * group
    kernel_x_t = kernel_x + (kernel_x - 1) * (dilation_x - 1)
    kernel_y_t = kernel_y + (kernel_y - 1) * (dilation_y - 1)
    pad_left   = pad_right = pad_top = pad_down = 0

    pad_x      = (in_size_x - kernel_x_t) -  int((in_size_x - kernel_x_t) / stride_x) * stride_x
    if(pad_x !=0):
        pad_x      = int((in_size_x - kernel_x_t) / stride_x) * stride_x + stride_x - (in_size_x - kernel_x_t)
        pad_left   = int(np.random.randint(0, high=pad_x, size=1))
        pad_right  = pad_x - pad_left

    pad_y      = (in_size_y - kernel_y_t) -  int((in_size_y - kernel_y_t) / stride_y) * stride_y
    if(pad_y != 0):
        pad_y      = int((in_size_y - kernel_y_t) / stride_y) * stride_y + stride_y - (in_size_y - kernel_y_t)
        pad_top    = int(np.random.randint(0, high=pad_y, size=1))
        pad_down   = pad_y - pad_top

    out_channel = int(np.random.randint(16, high=32, size=1))
    out_channel = int(out_channel / group) * group
    zero_point1 = int(np.random.randint(-2, high=2, size=1))
    std1        = int(np.random.randint(1, high=3, size=1))
    zero_point2 = int(np.random.randint(-2, high=2, size=1))
    std2        = int(np.random.randint(1, high=3, size=1))
    zero_point3 = int(np.random.randint(-6, high=6, size=1))
    std3        = int(np.random.randint(1, high=20, size=1))

    src_in = np.random.normal(zero_point1, std1, (batch, in_channel, in_size_y, in_size_x))
    weight = np.random.normal(zero_point2, std2, (out_channel, int(in_channel/group), kernel_y, kernel_x))
    bias   = np.random.normal(zero_point3, std3, out_channel)
    src_in = src_in.astype(np.float32)
    weight = weight.astype(np.float32)
    bias   = bias.astype(np.float32)

    t_src_in  = tensor(src_in)
    t_weight  = tensor(weight)
    t_bias    = tensor(bias)

    t_src_in  = fn.pad(t_src_in, (pad_left, pad_right, pad_top, pad_down), 'constant', 0)
    t_src_out1 = fn.conv2d(t_src_in, t_weight, bias=t_bias, stride=(stride_y, stride_x), padding=0, dilation=(dilation_y, dilation_x), groups=group).numpy()

    out_size_x = np.shape(t_src_out1)[3]
    out_size_y = np.shape(t_src_out1)[2]
    
    src_in_1 = src_in.flatten()
    src_out_1 = t_src_out1.flatten()
    weight_1  = weight.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + len(weight_1) + len(bias) + 18

    para.append(total_size)
    para.append(batch)      # 0
    para.append(in_channel) # 1
    para.append(in_size_y)  # 2
    para.append(in_size_x)  # 3
    para.append(stride_y)   # 4
    para.append(stride_x)   # 5
    para.append(kernel_y)   # 6
    para.append(kernel_x)   # 7
    para.append(pad_left)   # 8
    para.append(pad_right)  # 9
    para.append(pad_top)    # 10
    para.append(pad_down)   # 11
    para.append(out_channel)# 12
    para.append(dilation_x) # 13
    para.append(dilation_y) # 14
    para.append(out_size_x) # 15
    para.append(out_size_y) # 16
    para.append(group)      # 17
    print(para)

    with open("group_convolution_nchw_data_f32.bin", "wb") as fp:
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
    group_convolution_f32(test_type)
    print("end")
