#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
from torch import tensor
from torch.nn import functional as fn

def prelu_f32():
    para = []
    # init the input data and parameters
    batch       = int(np.random.randint(1, high=4, size=1))
    in_size_x   = int(np.random.randint(4, high=8, size=1))
    in_size_y   = int(np.random.randint(4, high=8, size=1))
    in_channel  = int(np.random.randint(8, high=16, size=1))
    zero_point1 = int(np.random.randint(-6, high=6, size=1))
    std1        = int(np.random.randint(1, high=20, size=1))
    zero_point2 = int(np.random.randint(0, high=1, size=1))
    std2        = int(np.random.randint(1, high=10, size=1))

    src_in = np.random.normal(zero_point1, std1, (batch, in_channel, in_size_y, in_size_x))
    weight = np.random.normal(zero_point2, std2, in_channel)
    src_in = src_in.astype(np.float32)
    weight = weight.astype(np.float32)

    t_src_in  = tensor(src_in)
    t_weight  = tensor(weight)
    t_src_out = fn.prelu(t_src_in, t_weight).numpy()

    #permute nchw to nhwc
    src_in_nhwc = np.transpose(src_in, [0, 2, 3, 1])
    out_nhwc    = np.transpose(t_src_out, [0, 2, 3, 1])

    src_in_1  = src_in_nhwc.flatten()
    src_out_1 = out_nhwc.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + len(weight) + 4

    para.append(total_size)
    para.append(batch)
    para.append(in_size_y)
    para.append(in_size_x)
    para.append(in_channel)

    with open("prelu_nhwc_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(weight)), *weight)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    prelu_f32()
    print("end")
