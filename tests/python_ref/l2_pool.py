#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
from torch import tensor
from torch.nn import functional as fn

def l2_pool_f32():
    para = []
    # init the input data and parameters
    batch      = int(np.random.randint(1, high=4, size=1))
    in_size_x  = int(np.random.randint(128, high=512, size=1))
    in_size_y  = int(np.random.randint(128, high=512, size=1))
    in_channel = int(np.random.randint(1, high=64, size=1))
    stride_x   = int(np.random.randint(1, high=3, size=1))
    stride_y   = int(np.random.randint(1, high=3, size=1))
    kernel_x   = int(np.random.randint(stride_x + 1, high=7, size=1))
    kernel_y   = int(np.random.randint(stride_y + 1, high=7, size=1))

    pad_left   = pad_right = pad_top = pad_down = 0
    pad_x      = (in_size_x - kernel_x) -  int((in_size_x - kernel_x) / stride_x) * stride_x
    if(pad_x !=0):
        pad_x      = int((in_size_x - kernel_x) / stride_x) * stride_x + stride_x - (in_size_x - kernel_x)
        pad_left   = int(np.random.randint(0, high=pad_x, size=1))
        pad_right  = pad_x - pad_left

    pad_y      = (in_size_y - kernel_y) -  int((in_size_y - kernel_y) / stride_y) * stride_y
    if(pad_y != 0):
        pad_y      = int((in_size_y - kernel_y) / stride_y) * stride_y + stride_y - (in_size_y - kernel_y)
        pad_top    = int(np.random.randint(0, high=pad_y, size=1))
        pad_down   = pad_y - pad_top
    zero_point = int(np.random.randint(-60000, high=60000, size=1))
    std        = int(np.random.randint(1, high=20, size=1))

    src_in = np.random.normal(zero_point, std, (batch, in_channel, in_size_y, in_size_x))

    t_src_in  = tensor(src_in)
    t_src_in  = fn.pad(t_src_in, (pad_left, pad_right, pad_top, pad_down), 'constant', 0)
    t_src_out = fn.max_pool2d(t_src_in, (kernel_y, kernel_x), stride=(stride_y, stride_x), padding=0).numpy()

    #permute nchw to nhwc
    src_in_nhwc = np.transpose(src_in, [3, 1, 0, 2])
    out_nhwc    = np.transpose(t_src_out, [3, 1, 0, 2])

    src_in_1  = src_in_nhwc.flatten()
    src_out_1 = out_nhwc.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 12

    para.append(total_size)
    para.append(batch)
    para.append(in_size_y)
    para.append(in_size_x)
    para.append(in_channel)
    para.append(stride_y)
    para.append(stride_x)
    para.append(kernel_y)
    para.append(kernel_x)
    para.append(pad_left)
    para.append(pad_right)
    para.append(pad_top)
    para.append(pad_down)

    with open("l2_pool_f32_data.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    l2_pool_f32()
    print("end")
