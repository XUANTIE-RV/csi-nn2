#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf
from torch import tensor

def shuffle_unit(x, groups):
    with tf.variable_scope('shuffle_unit'):
        n, h, w, c = x.get_shape().as_list()
        x = tf.reshape(x, shape=tf.convert_to_tensor([tf.shape(x)[0], h, w, groups, c // groups]))
        x = tf.transpose(x, tf.convert_to_tensor([0, 1, 2, 4, 3]))
        x = tf.reshape(x, shape=tf.convert_to_tensor([tf.shape(x)[0], h, w, c]))
    return x

def shuffle_channel_f32():
    para = []
    # init the input data and parameters
    batch       = int(np.random.randint(2, high=6, size=1))
    channel_tmp = int(np.random.randint(2, high=8, size=1))
    in_height   = int(np.random.randint(8, high=64, size=1))
    in_width    = int(np.random.randint(8, high=64, size=1))
    group       = int(np.random.randint(2, high=9, size=1))
    in_channel  = channel_tmp * group

    zero_point = int(np.random.randint(-600, high=600, size=1))
    std        = int(np.random.randint(1, high=20, size=1))

    src_in = np.random.normal(zero_point, std, (batch, in_height, in_width, in_channel))
    t_src_in = tf.convert_to_tensor(src_in)

    out_calcu = shuffle_unit(t_src_in, group)

    with tf.Session() as sess:
        src_out = sess.run(out_calcu)

    src_in_nchw = src_in.transpose(0, 3, 1, 2)
    src_out_nchw = src_out.transpose(0, 3, 1, 2)

    src_in_1  = src_in_nchw.flatten()
    src_out_1 = src_out_nchw.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 5

    para.append(total_size)
    para.append(batch)
    para.append(in_channel)
    para.append(in_height)
    para.append(in_width)
    para.append(group)
    print(para)

    with open("shuffle_channel_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    shuffle_channel_f32()
    print("end")
