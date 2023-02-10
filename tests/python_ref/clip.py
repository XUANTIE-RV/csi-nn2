#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf

def clip_f32():
    para = []
    # init the input data and parameters
    batch      = int(np.random.randint(1, high=4, size=1))
    in_size_x  = int(np.random.randint(16, high=32, size=1))
    in_size_y  = int(np.random.randint(16, high=32, size=1))
    in_channel = int(np.random.randint(16, high=64, size=1))

    zero_point = int(np.random.randint(-8, high=8, size=1))
    std        = int(np.random.randint(1, high=3, size=1))

    src_in = np.random.normal(zero_point, std, (batch, in_channel, in_size_x, in_size_y))
    t_src_in = tf.convert_to_tensor(src_in)

    clip_min_val = int(np.random.uniform(zero_point-std, high=zero_point, size=1))
    clip_max_val = int(np.random.uniform(zero_point, high=zero_point+std, size=1))

    t_src_out = tf.keras.backend.clip(t_src_in, min_value=clip_min_val, max_value=clip_max_val)

    sess = tf.Session()
    src_out = sess.run(t_src_out)

    src_in_1 = src_in.flatten()
    src_out_1 = src_out.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 6

    para.append(total_size)
    para.append(batch)
    para.append(in_channel)
    para.append(in_size_x)
    para.append(in_size_y)
    para.append(clip_min_val)
    para.append(clip_max_val)
    print(para)

    with open("clip_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()


if __name__ == '__main__':
    clip_f32()
    print('end')