#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
from torch import tensor
from torch.nn import functional as fn

def maxpool2d_f32(test_type):
    para = []
    # init the input data and parameters
    batch      = int(np.random.randint(1, high=4, size=1))
    channel    = int(np.random.randint(2, high=6, size=1))
    in_height = int(np.random.randint(16, high=32, size=1))
    in_width  = int(np.random.randint(16, high=32, size=1))

    if test_type == "random":
        stride_h   = int(np.random.randint(1, high=4, size=1))
        stride_w   = int(np.random.randint(1, high=4, size=1))

        kernel_h   = int(np.random.randint(stride_h, high=9, size=1))
        kernel_w   = int(np.random.randint(stride_w, high=9, size=1))
        pad_left  = pad_right = 0
        pad_top   = pad_down  = 0

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

    elif test_type == "2x2s2":
        stride_h    =  stride_w    = 2
        kernel_h    =  kernel_w    = 2
        pad_left  = pad_top = 0
        pad_down  = pad_right = 1
        in_height = 2 * in_height + 1
        in_width = 2 * in_width + 1

    elif test_type == "2x2s2_p1":
        stride_h    =  stride_w   = 2
        kernel_h    =  kernel_w   = 2
        pad_left  = pad_top = 1
        pad_down  = pad_right = 1
        in_height = 2 * in_height
        in_width = 2 * in_width


    elif test_type == "3x3s2":
        stride_h    =  stride_w    = 2
        kernel_h    =  kernel_w    = 3
        pad_left  = pad_top = 0
        pad_down  = pad_right = 1
        in_height = 2 * in_height
        in_width = 2 * in_width

    elif test_type == "3x3s2_p1":
        stride_h    =  stride_w    = 2
        kernel_h    =  kernel_w     = 3
        pad_left  = pad_top = 1
        pad_down  = pad_right = 1
        in_height = 2 * in_height + 1
        in_width = 2 * in_width + 1

    elif test_type == "3x3s1_p1":
        stride_h    =  stride_w     = 1
        kernel_h    =  kernel_w     = 3
        pad_left = pad_right = pad_top = pad_down = 1

    src_in = np.random.uniform(1, 10, (batch, channel, in_height, in_width))

    t_src_in  = tensor(src_in)
    t_src_in1  = fn.pad(t_src_in, (pad_left, pad_right, pad_top, pad_down), 'constant', 0)

    t_src_out = fn.max_pool2d(t_src_in1, kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w)).numpy()


    out_height = np.shape(t_src_out)[2]
    out_width  = np.shape(t_src_out)[3]

    src_in_1  = t_src_in.flatten()
    src_out_1 = t_src_out.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 14

    para.append(total_size)
    para.append(batch)
    para.append(channel)
    para.append(in_height)
    para.append(in_width)
    para.append(stride_h)
    para.append(stride_w)
    para.append(kernel_h)
    para.append(kernel_w)
    para.append(pad_left)
    para.append(pad_right)
    para.append(pad_top)
    para.append(pad_down)
    para.append(out_height)
    para.append(out_width)
    print(para)

    with open("maxpool_nchw_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    test_type = sys.argv[1]
    maxpool2d_f32(test_type)
    print("end")
