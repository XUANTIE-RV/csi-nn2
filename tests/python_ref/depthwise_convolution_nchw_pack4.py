#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
from torch import tensor
from torch.nn import functional as fn

def depthwise_convolution_f32():
    para = []
    # init the input data and parameters
    batch      = int(np.random.randint(1, high=2, size=1))
    in_size_x  = int(np.random.randint(13, high=14, size=1))
    in_size_y  = int(np.random.randint(7, high=8, size=1))
    in_channel = int(np.random.randint(8, high=9, size=1))
    stride_x   = int(np.random.randint(1, high=2, size=1))
    stride_y   = int(np.random.randint(1, high=2, size=1))
    kernel_x   = int(np.random.randint(3, high=4, size=1))
    kernel_y   = int(np.random.randint(3, high=4, size=1))
    dilation_x = int(np.random.randint(1, high=2, size=1))
    dilation_y = int(np.random.randint(1, high=2, size=1))
    kernel_x_t = kernel_x + (kernel_x - 1) * (dilation_x - 1)
    kernel_y_t = kernel_y + (kernel_y - 1) * (dilation_y - 1)
    pad_left   = pad_right = pad_top = pad_down = 0

    pad_x      = (in_size_x - kernel_x_t) -  int((in_size_x - kernel_x_t) / stride_x) * stride_x
    if(pad_x !=0):
        pad_left   = int(np.random.randint(0, high=pad_x, size=1))
        pad_right  = pad_x - pad_left

    pad_y      = (in_size_y - kernel_y_t) -  int((in_size_y - kernel_y_t) / stride_y) * stride_y
    if(pad_y != 0):
        pad_top    = int(np.random.randint(0, high=pad_y, size=1))
        pad_down   = pad_y - pad_top
    zero_point1 = int(np.random.randint(-2, high=2, size=1))
    std1        = int(np.random.randint(1, high=3, size=1))
    zero_point2 = int(np.random.randint(-2, high=2, size=1))
    std2        = int(np.random.randint(1, high=3, size=1))
    zero_point3 = int(np.random.randint(-3, high=3, size=1))
    std3        = int(np.random.randint(1, high=20, size=1))

    src_in = np.random.normal(zero_point1, std1, (batch, in_channel, in_size_y, in_size_x))
    weight = np.random.normal(zero_point2, std2, (in_channel, 1, kernel_y, kernel_x))
    bias   = np.random.normal(zero_point3, std3, in_channel)
    src_in = src_in.astype(np.float32)
    weight = weight.astype(np.float32)
    bias   = bias.astype(np.float32)

    t_src_in  = tensor(src_in)
    t_weight  = tensor(weight)
    t_bias    = tensor(bias)
    pad_left   = pad_right = pad_top = pad_down = 0
    pad_right = pad_down = 0


    t_src_in  = fn.pad(t_src_in, (pad_left, pad_right, pad_top, pad_down), 'constant', 0)
    t_src_out = fn.conv2d(t_src_in, t_weight, bias=t_bias, stride=(stride_y, stride_x), padding=0, dilation=(dilation_y, dilation_x), groups=in_channel).numpy()

    out_size_x = np.shape(t_src_out)[3]
    out_size_y = np.shape(t_src_out)[2]
    out_channel = np.shape(t_src_out)[1]

    # layout rearrangement: from [c, h, w] to [c/4, h, w, 4]
    src_in = src_in.reshape(in_channel, in_size_y, in_size_x)
    src_in = np.transpose(src_in, [1,2,0])
    src_in = src_in.reshape(in_size_y, in_size_x, (int)(in_channel/4), 4)
    src_in = np.transpose(src_in, [2,0,1,3])

    weight = weight.reshape(in_channel, kernel_y, kernel_x)
    weight = np.transpose(weight, [1,2,0])
    weight = weight.reshape(kernel_y, kernel_x, (int)(in_channel/4), 4)
    weight = np.transpose(weight, [2,0,1,3])

    t_src_out=np.array(t_src_out)
    t_src_out = t_src_out.reshape(in_channel, out_size_y, out_size_x)
    t_src_out = np.transpose(t_src_out, [1,2,0])
    t_src_out = t_src_out.reshape(out_size_y, out_size_x, (int)(in_channel/4), 4)
    t_src_out = np.transpose(t_src_out, [2,0,1,3])


    src_in_1  = src_in.flatten()
    weight_1  = weight.flatten()
    src_out_1 = t_src_out.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + len(weight_1) + len(bias) + 17

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
    para.append(dilation_y) # 13
    para.append(dilation_x) # 14
    para.append(out_size_y) # 15
    para.append(out_size_x) # 16
    print(para)


    with open("dwconv_pack4.bin", "wb") as fp:
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
    depthwise_convolution_f32()
    print("end")
