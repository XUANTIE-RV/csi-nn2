#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf

def unstack_f32():
    para = []

    # init the input data and parameters
    in_dim   = int(np.random.randint(2, high=5, size=1))
    in_shape = []
    for i in range(0, in_dim):
        in_shape.append(int(np.random.randint(16, high=32, size=1)))
    unstack_axis = int(np.random.randint(0, high=in_dim, size=1))

    zero_point = int(np.random.randint(-6, high=6, size=1))
    std        = int(np.random.randint(1, high=20, size=1))


    src_in = np.random.normal(zero_point, std, in_shape)


    t_src_in = tf.convert_to_tensor(src_in)

    out_calcu    = tf.unstack(t_src_in, axis=unstack_axis)

    with tf.Session() as sess:
        src_out = sess.run(out_calcu)

 
    src_in_1 = src_in.flatten()
    src_out_1 = np.array(src_out).ravel('C')

    total_size = (len(src_in_1) + len(src_out_1)) + 2 + in_dim

    para.append(total_size)
    para.append(unstack_axis)
    para.append(in_dim)
    for i in range(0, in_dim):
        para.append(in_shape[i])

    with open("unstack_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()
    print(para)
    return 0


if __name__ == '__main__':
    unstack_f32()
    print("end")
