#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf

def equal_f32():
    para = []
    # init the input data and parameters
    dim = int(np.random.randint(2, high=5, size=1))
    shape = np.random.randint(16, high=32, size=dim).astype(int)
    zero_point1 = int(np.random.randint(-600, high=600, size=1))
    std1        = int(np.random.randint(1, high=20, size=1))
    zero_point2 = int(np.random.randint(-600, high=600, size=1))
    std2        = int(np.random.randint(1, high=20, size=1))

    # src_in1 = np.random.normal(zero_point1, std1, shape)
    # src_in2 = np.random.normal(zero_point2, std2, shape)
    src_in1 = np.random.randint(-5, 5, shape)
    src_in2 = np.random.randint(-5, 5, shape)

    out_calcu = tf.math.equal(src_in1, src_in2)

    with tf.Session() as sess:
        src_out = sess.run(out_calcu)

    src_in_1  = src_in1.flatten()
    src_in_2  = src_in2.flatten()
    shape_1 =shape.flatten()

    src_out_1 = src_out.flatten()
    
    total_size = (len(shape_1) + len(src_in_1) + len(src_in_2) + len(src_out_1)) + 1

    para.append(total_size)
    para.append(dim)
    print(para)
    print(shape_1)
    print(len(src_out_1))

    with open("equal_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%di' % len(shape_1)), *shape_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_2)), *src_in_2)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    equal_f32()
    print("end")
