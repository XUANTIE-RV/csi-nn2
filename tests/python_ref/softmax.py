#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
from torch import tensor
from torch.nn import functional as fn

def softmax_f32():
    para = []
    # init the input data and parameters
    batch       = int(np.random.randint(1, high=4, size=1))
    in_size_x   = int(np.random.randint(64, high=256, size=1))
    in_size_y   = int(np.random.randint(64, high=256, size=1))
    in_channel  = int(np.random.randint(1, high=64, size=1))
    zero_point = int(np.random.randint(-60000, high=60000, size=1))
    std        = int(np.random.randint(1, high=20, size=1))

    axis = int(np.random.randint(0, high=4, size=1))

    src_in = np.random.normal(zero_point, std, (batch, in_size_y, in_size_x, in_channel))
    src_in = src_in.astype(np.float32)
    t_src_in = tensor(src_in)

    src_out = fn.softmax(t_src_in, dim = axis)

    size_all = batch * in_size_y * in_size_x * in_channel
    src_in_1  = src_in.reshape(size_all)
    src_out_1 = src_out.reshape(size_all)

    total_size = (len(src_in_1) + len(src_out_1)) + 5

    para.append(total_size)
    para.append(batch)
    para.append(in_size_y)
    para.append(in_size_x)
    para.append(in_channel)
    para.append(axis)

    with open("softmax_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    softmax_f32()
    print("end")
