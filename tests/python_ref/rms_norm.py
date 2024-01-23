#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import math
import struct
import numpy as np


def rms_norm(x: np.array, weight: np.array, axis, eps=1e-8) -> np.array:
    axis = axis if axis >= 0 else axis + len(x.shape)
    batches = np.prod(x.shape[:axis])
    norm_size = np.prod(x.shape[axis:])

    x = x.flatten()
    weight = weight.flatten()
    output = np.empty(x.shape, dtype=np.float32).flatten()
    for b in range(batches):
        avg = np.sum([x[b * norm_size + i]**2 for i in range(norm_size)]) / norm_size
        scale = 1.0 / math.sqrt(avg + eps)
        for i in range(norm_size):
            output[b * norm_size + i] = x[b * norm_size + i] * scale * weight[i]

    return output

def rms_norm_f32():
    para = []
    # init the input data and parameters
    batch       = int(np.random.randint(1, high=4, size=1))
    in_channel  = int(np.random.randint(1, high=8, size=1))
    in_size_y   = int(np.random.randint(4, high=16, size=1))
    in_size_x   = int(np.random.randint(4, high=16, size=1))
    zero_point1 = int(np.random.randint(-60, high=60, size=1))
    std1        = int(np.random.randint(1, high=20, size=1))
    zero_point2 = int(np.random.randint(-60, high=60, size=1))
    std2        = int(np.random.randint(1, high=20, size=1))

    axis = -1
    eps = 1e-8

    shape = [batch, in_channel, in_size_y, in_size_x]
    norm_shape = np.array(shape[axis:])
    src_in = np.random.normal(zero_point1, std1, shape)
    weight = np.random.normal(zero_point2, std2, norm_shape)
    src_in = src_in.astype(np.float32)
    weight = weight.astype(np.float32)

    src_out = rms_norm(src_in, weight, axis, eps)

    size_all = batch * in_channel * in_size_y * in_size_x
    src_in_1  = src_in.reshape(size_all)
    weight_1  = weight.reshape(np.prod(norm_shape))
    src_out_1 = src_out.reshape(size_all)

    total_size = len(src_in_1) + len(weight_1) + len(src_out_1) + 6
    para.append(total_size) # 0
    para.append(batch)      # 1
    para.append(in_channel) # 2
    para.append(in_size_y)  # 3
    para.append(in_size_x)  # 4
    para.append(axis)       # 5
    para_f = [eps]          # 6
    print(para + para_f)

    with open("rms_norm_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(para_f)), *para_f)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(weight_1)), *weight_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

if __name__ == '__main__':
    rms_norm_f32()
    print("end")
