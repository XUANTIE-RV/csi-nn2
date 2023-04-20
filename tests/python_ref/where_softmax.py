#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf

def where_softmax_f32():
    para = []
    # init the input data and parameters
    shape_rank = int(np.random.randint(1, high=4, size=1))
    shape = [int(np.random.randint(1, high=16, size=1)) for i in range(shape_rank)]

    zero_point = int(np.random.randint(-3, high=3, size=1))
    std        = int(np.random.randint(1, high=3, size=1))
    axis       = int(np.random.randint(0, high=shape_rank, size=1))

    condition = np.random.choice([0, 1], size=shape)
    x = np.full(shape, fill_value=-np.inf)
    y = np.random.normal(zero_point, std, shape)

    condition = condition.astype(np.float32)
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    tf.enable_eager_execution()
    where_output = tf.where(condition, x, y)
    output = tf.nn.softmax(where_output, axis).numpy()

    condition = condition.flatten()
    y = y.flatten()
    output = output.flatten()

    total_size = len(condition) + len(y) + len(output) + 2 + shape_rank
    para.append(total_size)
    para.append(shape_rank)
    for i in range(shape_rank):
        para.append(shape[i])
    para.append(axis)
    print(para)

    with open("where_softmax_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(condition)), *condition)
        fp.write(data)
        data = struct.pack(('%df' % len(y)), *y)
        fp.write(data)
        data = struct.pack(('%df' % len(output)), *output)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    where_softmax_f32()
    print("end")
