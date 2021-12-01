#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
from torch import tensor
from torch.nn import functional as fn

def avgpool3d_f32():
    para = []
    # init the input data and parameters
    batch      = int(np.random.randint(1, high=4, size=1))
    channel    = int(np.random.randint(2, high=6, size=1))
    in_depth  = int(np.random.randint(32, high=64, size=1))
    in_height = int(np.random.randint(32, high=64, size=1))
    in_width  = int(np.random.randint(32, high=64, size=1))

    stride_d   = int(np.random.randint(1, high=4, size=1))
    stride_h   = int(np.random.randint(1, high=4, size=1))
    stride_w   = int(np.random.randint(1, high=4, size=1))

    kernel_d   = int(np.random.randint(stride_d, high=9, size=1))
    kernel_h   = int(np.random.randint(stride_h, high=9, size=1))
    kernel_w   = int(np.random.randint(stride_w, high=9, size=1))

    include_pad  = int(np.random.randint(0, high=2, size=1))    # 0: false  1: true

    pad_left  = pad_right = 0
    pad_top   = pad_down  = 0
    pad_front = pad_back  = 0


    pad_w      = (in_width - kernel_w) -  int((in_width - kernel_w) / stride_w) * stride_w
    if(pad_w !=0):
        pad_w      = int((in_width - kernel_w) / stride_w) * stride_w + stride_w - (in_width - kernel_w)
        pad_left   = int(np.random.randint(0, high=pad_w, size=1))
        pad_right  = pad_w - pad_left

    pad_h      = (in_height - kernel_h) -  int((in_height - kernel_h) / stride_h) * stride_h
    if(pad_h !=0):
        pad_h      = int((in_height - kernel_h) / stride_h) * stride_h + stride_h - (in_height - kernel_h)
        pad_top   = int(np.random.randint(0, high=pad_h, size=1))
        pad_down  = pad_h - pad_top

    pad_d      = (in_depth - kernel_d) -  int((in_depth - kernel_d) / stride_d) * stride_d
    if(pad_d !=0):
        pad_d      = int((in_depth - kernel_d) / stride_d) * stride_d + stride_d - (in_depth - kernel_d)
        pad_front   = int(np.random.randint(0, high=pad_d, size=1))
        pad_back  = pad_d - pad_front

    zero_point = int(np.random.randint(-8, high=8, size=1))
    std        = int(np.random.randint(1, high=3, size=1))

    src_in = np.random.normal(zero_point, std, (batch, channel, in_depth, in_height, in_width))

    t_src_in  = tensor(src_in)
    t_src_in1  = fn.pad(t_src_in, (pad_left, pad_right, pad_top, pad_down, pad_front, pad_back), 'constant', 0)

    t_src_out = fn.avg_pool3d(t_src_in1, (kernel_d, kernel_h, kernel_w), stride=(stride_d, stride_h, stride_w), padding=0, 
                              count_include_pad = True if include_pad else False).numpy()

    out_depth  = np.shape(t_src_out)[2]
    out_height = np.shape(t_src_out)[3]
    out_width  = np.shape(t_src_out)[4]

    src_in_1  = t_src_in.flatten()
    src_out_1 = t_src_out.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 21

    para.append(total_size)
    para.append(batch)      # 0
    para.append(channel)    # 1
    para.append(in_depth)   # 2
    para.append(in_height)  # 3
    para.append(in_width)   # 4
    para.append(stride_d)   # 5
    para.append(stride_h)   # 6
    para.append(stride_w)   # 7
    para.append(kernel_d)   # 8
    para.append(kernel_h)   # 9
    para.append(kernel_w)   # 10
    para.append(pad_left)   # 11
    para.append(pad_right)  # 12
    para.append(pad_top)    # 13
    para.append(pad_down)   # 14
    para.append(pad_front)  # 15
    para.append(pad_back)   # 16
    para.append(out_depth)  # 17
    para.append(out_height) # 18
    para.append(out_width)  # 19
    para.append(include_pad)# 20
    print(para)

    with open("avgpool3d_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    avgpool3d_f32()
    print("end")
