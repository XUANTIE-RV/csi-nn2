#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf

def where_f32():
    para = []
    # init the input data and parameters
    shape_rank = int(np.random.randint(1, high=4, size=1))
    shape = [int(np.random.randint(1, high=16, size=1)) for i in range(shape_rank)]

    zero_point1 = int(np.random.randint(-3, high=3, size=1))
    std1        = int(np.random.randint(1, high=3, size=1))
    zero_point2 = int(np.random.randint(-3, high=3, size=1))
    std2        = int(np.random.randint(1, high=3, size=1))

    condition = np.random.choice([0, 1], size=shape)
    x = np.random.normal(zero_point1, std1, shape)
    y = np.random.normal(zero_point2, std2, shape)

    condition = condition.astype(np.float32)
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    tf.enable_eager_execution()
    output = tf.where(condition, x, y).numpy()

    condition = condition.flatten()
    x = x.flatten()
    y = y.flatten()
    output = output.flatten()

    total_size = len(condition) + len(x) + len(y) + len(output) + 1 + shape_rank
    para.append(total_size)
    para.append(shape_rank)
    for i in range(shape_rank):
        para.append(shape[i])
    print(para)

    with open("where_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(condition)), *condition)
        fp.write(data)
        data = struct.pack(('%df' % len(x)), *x)
        fp.write(data)
        data = struct.pack(('%df' % len(y)), *y)
        fp.write(data)
        data = struct.pack(('%df' % len(output)), *output)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    where_f32()
    print("end")
