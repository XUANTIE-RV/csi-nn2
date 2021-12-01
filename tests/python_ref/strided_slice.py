#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import random
import tensorflow as tf

# t = tf.constant([[[1, 1, 1], [2, 2, 2]],
#                  [[3, 3, 3], [4, 4, 4]],
#                  [[5, 5, 5], [6, 6, 6]]])

# print(t.shape)

# t = tf.constant([[[1, 1, 1], [2, 2, 2]],
#                  [[3, 3, 3], [4, 4, 4]],
#                  [[5, 5, 5], [6, 6, 6]]])
# b = tf.strided_slice(t, [1,0], [2,1], [2,1])  # [[[3, 3, 3]]]
# sess = tf.Session()
# c = sess.run(b)

def stride_slice_f32():
    para = []
    # init the input data and parameters
    in_dim   = int(np.random.randint(1, high=4, size=1))
    in_shape = []
    for i in range(0, in_dim):
        in_shape.append(int(np.random.randint(5, high=10, size=1)))

    slice_count = int(np.random.randint(1, high=in_dim+1, size=1))
    begin = []
    end = []
    stride = []

    for i in range(0, slice_count):
        begin.append(int(np.random.randint(0, high=in_shape[i], size=1)))
        end.append(int(np.random.randint(begin[i]+1, high=in_shape[i]+1, size=1)))
        # stride.append(int(np.random.randint(1, high=in_shape[i]+1, size=1)))
        stride.append(int(np.random.randint(1, high=4, size=1)))


    zero_point = int(np.random.randint(-600, high=600, size=1))
    std        = int(np.random.randint(1, high=20, size=1))

    src_in = np.random.normal(zero_point, std, in_shape)
    out_calcu = tf.strided_slice(src_in, begin=begin, end=end, strides=stride)

    sess = tf.Session()
    src_out = sess.run(out_calcu)

    src_in_1  = src_in.flatten()
    src_out_1 = src_out.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 3 + in_dim + 3 * slice_count

    para.append(total_size)
    para.append(in_dim)
    for i in range(0, in_dim):
        para.append(in_shape[i])
    para.append(slice_count)
    for i in range(0, slice_count):
        para.append(begin[i])
        para.append(end[i])
        para.append(stride[i])
    para.append(len(src_out_1))

    print(para)

    with open("strided_slice_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        # data = struct.pack(('%di' % len(begin)), *begin)
        # fp.write(data)
        # data = struct.pack(('%di' % len(end)), *end)
        # fp.write(data)
        # data = struct.pack(('%di' % len(stride)), *stride)
        # fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    stride_slice_f32()
    print("end")
