#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
from torch import tensor
from torch.nn import functional as fn

def convolution3d_f32():
    para = []
    # init the input data and parameters
    batch       = int(np.random.randint(1, high=4, size=1))
    in_channel  = int(np.random.randint(2, high=8, size=1))
    in_depth    = int(np.random.randint(16, high=128, size=1))
    in_height   = int(np.random.randint(16, high=128, size=1))
    in_width    = int(np.random.randint(16, high=128, size=1))

    out_channel = int(np.random.randint(8, high=16, size=1))

    stride_d    = int(np.random.randint(1, high=5, size=1))
    stride_h    = int(np.random.randint(1, high=5, size=1))
    stride_w    = int(np.random.randint(1, high=5, size=1))

    kernel_d    = int(np.random.randint(stride_d, high=8, size=1))
    kernel_h    = int(np.random.randint(stride_h, high=8, size=1))
    kernel_w    = int(np.random.randint(stride_w, high=8, size=1))

    dilation_d  = int(np.random.randint(1, high=4, size=1))
    dilation_h  = int(np.random.randint(1, high=4, size=1))
    dilation_w  = int(np.random.randint(1, high=4, size=1))

    kernel_d_t  = kernel_d + (kernel_d - 1) * (dilation_d - 1)
    kernel_h_t  = kernel_h + (kernel_h - 1) * (dilation_h - 1)
    kernel_w_t  = kernel_w + (kernel_w - 1) * (dilation_w - 1)


    pad_left  = pad_right = 0
    pad_top   = pad_down  = 0
    pad_front = pad_back  = 0

    pad_w      = (in_width - kernel_w_t) -  int((in_width - kernel_w_t) / stride_w) * stride_w
    if(pad_w !=0):
        pad_w      = int((in_width - kernel_w_t) / stride_w) * stride_w + stride_w - (in_width - kernel_w_t)
        pad_left   = int(np.random.randint(0, high=pad_w, size=1))
        pad_right  = pad_w - pad_left

    pad_h      = (in_height - kernel_h_t) -  int((in_height - kernel_h_t) / stride_h) * stride_h
    if(pad_h != 0):
        pad_h      = int((in_height - kernel_h_t) / stride_h) * stride_h + stride_h - (in_height - kernel_h_t)
        pad_top    = int(np.random.randint(0, high=pad_h, size=1))
        pad_down   = pad_h - pad_top

    pad_d      = (in_depth - kernel_d_t) -  int((in_depth - kernel_d_t) / stride_d) * stride_d
    if(pad_d != 0):
        pad_d      = int((in_depth - kernel_d_t) / stride_d) * stride_d + stride_d - (in_depth - kernel_d_t)
        pad_front  = int(np.random.randint(0, high=pad_d, size=1))
        pad_back   = pad_d - pad_front

    zero_point1 = int(np.random.randint(-6, high=6, size=1))
    std1        = int(np.random.randint(1, high=2, size=1))
    zero_point2 = int(np.random.randint(-6, high=6, size=1))
    std2        = int(np.random.randint(1, high=2, size=1))
    zero_point3 = int(np.random.randint(-6, high=6, size=1))
    std3        = int(np.random.randint(1, high=2, size=1))

    src_in = np.random.normal(zero_point1, std1, (batch, in_channel, in_depth, in_height, in_width))
    weight = np.random.normal(zero_point2, std2, (out_channel, in_channel, kernel_d, kernel_h, kernel_w))
    bias   = np.random.normal(zero_point3, std3, out_channel)

    # src_in = np.random.randint(-16, 16, (batch, in_channel, in_depth, in_height, in_width))
    # weight = np.random.randint(-5, 5, (out_channel, in_channel, kernel_d, kernel_h, kernel_w))
    # bias   = np.random.randint(-10, 10, out_channel)

    src_in = src_in.astype(np.float32)
    weight = weight.astype(np.float32)
    bias   = bias.astype(np.float32)

    #print(src_in)
    #print(weight)
    t_src_in  = tensor(src_in)
    t_weight  = tensor(weight)
    t_bias    = tensor(bias)

    t_src_in1  = fn.pad(t_src_in, (pad_left, pad_right, pad_top, pad_down, pad_front, pad_back), 'constant', 0)
    t_src_out = fn.conv3d(t_src_in1, t_weight, bias=t_bias, stride=(stride_d, stride_h, stride_w), dilation=(dilation_d, dilation_h, dilation_w)).numpy()

    out_depth  = np.shape(t_src_out)[2]
    out_height = np.shape(t_src_out)[3]
    out_width  = np.shape(t_src_out)[4]

    #print(np.shape(t_src_in1))
    #print(np.shape(t_src_out))
    #print((kernel_y, kernel_x, stride_y, stride_x, dilation_y, dilation_x))
    src_in_1   = t_src_in.flatten()
    weight_1   = t_weight.flatten()
    src_out_1  = t_src_out.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + len(weight_1) + len(bias) + 24

    para.append(total_size)
    para.append(batch)
    para.append(in_channel)
    para.append(in_depth)
    para.append(in_height)
    para.append(in_width)
    para.append(out_channel)
    para.append(kernel_d)
    para.append(kernel_h)
    para.append(kernel_w)
    para.append(out_depth)
    para.append(out_height)
    para.append(out_width)
    para.append(stride_d)
    para.append(stride_h)
    para.append(stride_w)
    para.append(pad_left)
    para.append(pad_right)
    para.append(pad_top)
    para.append(pad_down)
    para.append(pad_front)
    para.append(pad_back)
    para.append(dilation_d)
    para.append(dilation_h)
    para.append(dilation_w)

    with open("convolution3d_data_f32.bin", "wb") as fp:
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
    convolution3d_f32()
    print("end")
