#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
from torch import tensor
from torch.nn import functional as fn

def pad_f32():
    para = []
    # init the input data and parameters
    batch       = int(np.random.randint(1, high=4, size=1))
    in_size_x   = int(np.random.randint(32, high=64, size=1))
    in_size_y   = int(np.random.randint(32, high=64, size=1))
    in_channel  = int(np.random.randint(2, high=16, size=1))
    pad_left    = int(np.random.randint(0, high=3, size=1))
    pad_right   = int(np.random.randint(0, high=3, size=1))
    pad_top     = int(np.random.randint(0, high=3, size=1))
    pad_down    = int(np.random.randint(0, high=3, size=1))
    zero_point1 = int(np.random.randint(-6, high=6, size=1))
    std1        = int(np.random.randint(1, high=20, size=1))

    src_in = np.random.normal(zero_point1, std1, (batch, in_channel, in_size_y, in_size_x))

    t_src_in  = tensor(src_in)
    t_src_out  = fn.pad(t_src_in, (pad_left, pad_right, pad_top, pad_down), 'constant', 0).numpy()

    src_in_nchw = src_in
    out_nchw = t_src_out

    size_all  = batch*in_size_y*in_size_x*in_channel
    src_in_1  = src_in_nchw.reshape(size_all)
    src_out_1 = out_nchw.reshape(batch * (in_size_y + pad_top + pad_down) * (in_size_x + pad_left + pad_right) * in_channel)

    total_size = (len(src_in_1) + len(src_out_1)) + 8

    para.append(total_size)
    para.append(batch)
    para.append(in_channel)
    para.append(in_size_y)
    para.append(in_size_x)
    para.append(pad_left)
    para.append(pad_right)
    para.append(pad_top)
    para.append(pad_down)

    with open("pad_nchw_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    pad_f32()
    print("end")
