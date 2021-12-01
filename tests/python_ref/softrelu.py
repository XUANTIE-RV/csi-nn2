#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import math
from torch import tensor
import torch


def softrelu_f32():
    para = []
    # init the input data and parameters
    batch       = int(np.random.randint(1, high=4, size=1))
    in_size_x   = int(np.random.randint(1, high=5, size=1))
    in_size_y   = int(np.random.randint(1, high=5, size=1))
    in_channel  = int(np.random.randint(1, high=64, size=1))
    n = int(np.random.randint(1, high=50, size=1))
    zero_point = int(np.random.randint(-60000, high=60000, size=1))
    std        = int(np.random.randint(1, high=2, size=1))

    src_in = np.random.normal(zero_point, std, (batch, in_size_y, in_size_x, in_channel))

    size_all = batch*in_size_y*in_size_x*in_channel
    src_in_1  = src_in.reshape(size_all)

    src_out_1 = []
    for i in range(len(src_in_1)):
        temp = (math.log(1 + math.exp(max(min(src_in_1[i], n), n))))
        src_out_1.append(temp)


    total_size = (len(src_in_1) + len(src_out_1)) + 5

    para.append(total_size)
    para.append(batch)
    para.append(in_size_y)
    para.append(in_size_x)
    para.append(in_channel)
    para.append(n)

    with open("softrelu_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    softrelu_f32()
    print("end")
