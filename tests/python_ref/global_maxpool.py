#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
from torch import tensor
from torch.nn import AdaptiveMaxPool2d

def global_maxpool2d_f32():
    para = []
    # init the input data and parameters
    batch      = int(np.random.randint(1, high=4, size=1))
    in_size_x  = int(np.random.randint(64, high=128, size=1))
    in_size_y  = int(np.random.randint(64, high=128, size=1))
    in_channel = int(np.random.randint(1, high=64, size=1))

    out_height  = int(np.random.randint(1, high=2, size=1))
    out_width  = int(np.random.randint(1, high=2, size=1))

    zero_point = int(np.random.randint(-600, high=600, size=1))
    std        = int(np.random.randint(1, high=20, size=1))

    src_in = np.random.normal(zero_point, std, (batch, in_channel, in_size_y, in_size_x))

    t_src_in  = tensor(src_in)   
    gmp = AdaptiveMaxPool2d((out_height, out_width))
    t_src_out = gmp(t_src_in).numpy()


    #permute nchw to nhwc
    src_in_nhwc = np.transpose(src_in, [0, 2, 3, 1])
    out_nhwc    = np.transpose(t_src_out, [0, 2, 3, 1])


    src_in_1  = src_in_nhwc.flatten()
    src_out_1 = out_nhwc.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 6

    para.append(total_size)
    para.append(batch)
    para.append(in_size_y)
    para.append(in_size_x)
    para.append(in_channel)
    para.append(out_height)
    para.append(out_width)

    print(para)


    with open("global_maxpool_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    global_maxpool2d_f32()
    print("end")
