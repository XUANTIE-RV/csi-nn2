#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf

# a = [[1,0],[0,0]]
# b = [[[[1, 2], [3, 4]],
#      [[5, 6], [7, 8]]],

# [[[9, 10], [11, 12]],
# [[13, 14], [15, 16]]]]

# c = [[1,2,3,4,5,6,7,8],[7,8,9,10,11,12,13,14]]
# d = [1,2,3,4,5,6,7,8]

# with tf.Session() as sess:
#     print(sess.run(tf.gather_nd(b, a)))

def gather_nd_f32():
    para = []
    # init the input data and parameters
    in_dim   = int(np.random.randint(1, high=5, size=1))
    in_shape = []
    for i in range(0, in_dim):
        in_shape.append(int(np.random.randint(4, high=16, size=1)))

    indices_dim = int(np.random.randint(1, high=6, size=1))
    indices_shape = []
    for i in range(0, indices_dim-1):
        indices_shape.append(int(np.random.randint(1, high=24, size=1)))

    # indices.shape[-1] must be <= params.rank
    indices_shape.append(int(np.random.randint(1, high=in_dim+1, size=1)))


    zero_point  = int(np.random.randint(-600, high=600, size=1))
    std         = int(np.random.randint(1, high=20, size=1))

    src_in = np.random.normal(zero_point, std, in_shape)
    src_in = src_in.astype(np.float32)

    indices_in = np.random.randint(0, high=32, size = indices_shape)

    out_calcu = tf.gather_nd(src_in, indices_in)

    with tf.Session() as sess:
        src_out = sess.run(out_calcu)

    src_in_1  = src_in.ravel('C')
    src_out_1 = src_out.flatten()
    indices_in_1 = indices_in.ravel('C')

    total_size = (len(indices_in_1) + len(src_in_1) + len(src_out_1)) + 2 + in_dim + indices_dim

    para.append(total_size)
    para.append(in_dim)
    for i in range(0, in_dim):
        para.append(in_shape[i])

    para.append(indices_dim)
    for i in range(0, indices_dim):
        para.append(indices_shape[i])
    print(para)

    with open("gather_nd_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%di' % len(indices_in_1)), *indices_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    gather_nd_f32()
    print("end")
