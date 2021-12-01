#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np


def dequantize_f32():
    para = []
    # init the input data and parameters
    in_size  = int(np.random.randint(128, high=512, size=1))
    zero_point = int(np.random.randint(-60000, high=60000, size=1))
    std        = int(np.random.randint(1, high=20, size=1))

    src_in = np.random.normal(zero_point, std, in_size)
    src_in = src_in.astype(np.float32)

    total_size = in_size + 1

    para.append(total_size)
    para.append(in_size)

    with open("dequantize_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in)), *src_in)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    dequantize_f32()
    print("end")
