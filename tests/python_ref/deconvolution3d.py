#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
from torch import tensor
from torch.nn import functional as fn

def deconvolution3d_f32():
    para = []
    # init the input data and parameters
    batch       = int(np.random.randint(1, high=4, size=1))
    in_channel  = int(np.random.randint(4, high=16, size=1))
    in_depth    = int(np.random.randint(16, high=32, size=1))
    in_height   = int(np.random.randint(16, high=32, size=1))
    in_width    = int(np.random.randint(16, high=32, size=1))

    out_channel = int(np.random.randint(8, high=16, size=1))

    stride_d    = int(np.random.randint(1, high=3, size=1))
    stride_h    = int(np.random.randint(1, high=3, size=1))
    stride_w    = int(np.random.randint(1, high=3, size=1))

    kernel_d    = int(np.random.randint(stride_d+1, high=8, size=1))
    kernel_h    = int(np.random.randint(stride_h+1, high=8, size=1))
    kernel_w    = int(np.random.randint(stride_w+1, high=8, size=1))


    pad_d = pad_front = pad_back  = int(np.random.randint(1, high=stride_d+1, size=1))
    pad_h = pad_top   = pad_down  = int(np.random.randint(1, high=stride_h+1, size=1))
    pad_w = pad_left  = pad_right = int(np.random.randint(1, high=stride_w+1, size=1))


    out_pad_d = int(np.random.randint(0, high=1, size=1))   # 0
    out_pad_h = int(np.random.randint(0, high=1, size=1))   # 0
    out_pad_w = int(np.random.randint(0, high=1, size=1))   # 0


    zero_point1 = int(np.random.randint(-6, high=6, size=1))
    std1        = int(np.random.randint(1, high=20, size=1))
    zero_point2 = int(np.random.randint(-6, high=6, size=1))
    std2        = int(np.random.randint(1, high=20, size=1))
    zero_point3 = int(np.random.randint(-6, high=6, size=1))
    std3        = int(np.random.randint(1, high=20, size=1))

    dilation_d  = int(np.random.randint(1, high=2, size=1))
    dilation_h  = int(np.random.randint(1, high=2, size=1))
    dilation_w  = int(np.random.randint(1, high=2, size=1))

    src_in = np.random.normal(zero_point1, std1, (batch, in_channel, in_depth, in_height, in_width))
    weight = np.random.normal(zero_point2, std2, (in_channel, out_channel, kernel_d, kernel_h, kernel_w))
    bias   = np.random.normal(zero_point3, std3, out_channel)

    # src_in = np.random.randint(-8, 8, (batch, in_channel, in_depth, in_height, in_width))
    # weight = np.random.randint(-5, 5, (in_channel, out_channel, kernel_d, kernel_h, kernel_w))
    # bias   = np.random.randint(-10, 10, out_channel)

    t_src_in  = tensor(src_in)
    t_weight  = tensor(weight)
    t_bias    = tensor(bias)

    # t_src_in1  = fn.pad(t_src_in, (pad_left, pad_right, pad_top, pad_down, pad_front, pad_back), 'constant', 0)
    t_src_out = fn.conv_transpose3d(t_src_in, t_weight, bias=t_bias, stride=(stride_d, stride_h, stride_w),
                                    padding =(pad_d, pad_h, pad_w), dilation=(dilation_d, dilation_h, dilation_w)).numpy()


    out_depth  = np.shape(t_src_out)[2]
    out_height = np.shape(t_src_out)[3]
    out_width  = np.shape(t_src_out)[4]

    src_in_1  = src_in.flatten()
    weight_1  = weight.flatten()
    src_out_1 = t_src_out.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + len(weight_1) + len(bias) + 27

    para.append(total_size)
    para.append(batch)      # 0
    para.append(in_channel) # 1
    para.append(in_depth)   # 2
    para.append(in_height)  # 3
    para.append(in_width)   # 4
    para.append(out_channel)# 5
    para.append(kernel_d)   # 6
    para.append(kernel_h)   # 7
    para.append(kernel_w)   # 8
    para.append(out_depth)  # 9
    para.append(out_height) # 10
    para.append(out_width)  # 11
    para.append(stride_d)   # 12
    para.append(stride_h)   # 13
    para.append(stride_w)   # 14
    para.append(pad_left)   # 15
    para.append(pad_right)  # 16
    para.append(pad_top)    # 17
    para.append(pad_down)   # 18
    para.append(pad_front)  # 19
    para.append(pad_back)   # 20

    para.append(out_pad_d)  # 21
    para.append(out_pad_h)  # 22
    para.append(out_pad_w)  # 23

    para.append(dilation_d) # 24
    para.append(dilation_h) # 25
    para.append(dilation_w) # 26
    print(para)

    with open("deconvolution3d_data_f32.bin", "wb") as fp:
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
    deconvolution3d_f32()
    print("end")
