#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
from torch import tensor
from torch.nn import functional as fn

def deconvolution_f32():
    para = []
    # init the input data and parameters
    batch       = int(np.random.randint(1, high=4, size=1))
    in_size_x   = int(np.random.randint(8, high=16, size=1))
    in_size_y   = int(np.random.randint(8, high=16, size=1))
    in_channel  = int(np.random.randint(2, high=16, size=1))
    stride_x    = int(np.random.randint(1, high=3, size=1))
    stride_y    = int(np.random.randint(1, high=3, size=1))
    kernel_x    = int(np.random.randint(stride_x+1, high=8, size=1))
    kernel_y    = int(np.random.randint(stride_y+1, high=8, size=1))


    pad_x = pad_left = pad_right = int(np.random.randint(1, high=stride_x+1, size=1))
    pad_y = pad_top = pad_down = int(np.random.randint(1, high=stride_y+1, size=1))

    out_channel = int(np.random.randint(4, high=20, size=1))
    zero_point1 = int(np.random.randint(-2, high=2, size=1))
    std1        = int(np.random.randint(1, high=5, size=1))
    zero_point2 = int(np.random.randint(-2, high=2, size=1))
    std2        = int(np.random.randint(1, high=5, size=1))
    zero_point3 = int(np.random.randint(-2, high=2, size=1))
    std3        = int(np.random.randint(1, high=10, size=1))
    dilation_x  = int(np.random.randint(1, high=2, size=1))
    dilation_y  = int(np.random.randint(1, high=2, size=1))

    src_in = np.random.normal(zero_point1, std1, (batch, in_channel, in_size_y, in_size_x))
    weight = np.random.normal(zero_point2, std2, (in_channel, out_channel, kernel_y, kernel_x))
    bias   = np.random.normal(zero_point3, std3, out_channel)

    t_src_in  = tensor(src_in)
    t_weight  = tensor(weight)
    t_bias    = tensor(bias)
    # t_src_in  = fn.pad(t_src_in, (pad_left, pad_right, pad_top, pad_down), 'constant', 0)
    t_src_out = fn.conv_transpose2d(t_src_in, t_weight, bias=t_bias, stride=(stride_y, stride_x), padding=(pad_y,pad_x), dilation=(dilation_y, dilation_x)).numpy()

    out_size_y = np.shape(t_src_out)[2]
    out_size_x = np.shape(t_src_out)[3]

    src_in_1  = src_in.flatten()
    weight_1  = weight.flatten()
    src_out_1 = t_src_out.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + len(weight_1) + len(bias) + 17

    para.append(total_size)
    para.append(batch)      #0
    para.append(in_channel) #1
    para.append(in_size_y)  #2
    para.append(in_size_x)  #3
    para.append(stride_y)   #4
    para.append(stride_x)   #5
    para.append(kernel_y)   #6
    para.append(kernel_x)   #7
    para.append(pad_left)   #8
    para.append(pad_right)  #9
    para.append(pad_top)    #10
    para.append(pad_down)   #11
    para.append(dilation_x) #12
    para.append(dilation_y) #13
    para.append(out_channel)#14
    para.append(out_size_x) #15
    para.append(out_size_y) #16
    print(para)

    with open("deconvolution_nchw_data_f32.bin", "wb") as fp:
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
    deconvolution_f32()
    print("end")
