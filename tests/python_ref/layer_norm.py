#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf

def rand_float(low, high):
    return np.random.rand() * (high - low) + low

def layer_norm_f32():
    para = []
    # init the input data and parameters
    batch       = int(np.random.randint(1, high=4, size=1))
    in_channel  = int(np.random.randint(1, high=8, size=1))
    in_size_y   = int(np.random.randint(4, high=16, size=1))
    in_size_x   = int(np.random.randint(4, high=16, size=1))
    zero_point  = int(np.random.randint(-60, high=60, size=1))
    std         = int(np.random.randint(1, high=20, size=1))

    axis = -1
    g = rand_float(0, 3)
    b = rand_float(-1, 1)

    src_in = np.random.normal(zero_point, std, (batch, in_channel, in_size_y, in_size_x))
    src_in = src_in.astype(np.float32)

    tf.enable_eager_execution()
    layer_norm = tf.keras.layers.LayerNormalization(
        axis=axis,
        gamma_initializer=tf.constant_initializer(g),
        beta_initializer=tf.constant_initializer(b),
    )
    src_out = layer_norm(tf.convert_to_tensor(src_in)).numpy()

    size_all = batch * in_channel * in_size_y * in_size_x
    src_in_1  = src_in.reshape(size_all)
    src_out_1 = src_out.reshape(size_all)

    norm_num = in_channel * in_size_y * in_size_x
    gamma = np.full(norm_num, fill_value=g)
    beta = np.full(norm_num, fill_value=b)

    total_size = len(src_in_1) + len(src_out_1) + len(gamma) + len(beta) + 5
    para.append(total_size)
    para.append(batch)
    para.append(in_channel)
    para.append(in_size_y)
    para.append(in_size_x)
    para.append(axis)
    print(para)
    print("gamma:", g)
    print("beta:", b)

    with open("layer_norm_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        data = struct.pack(('%df' % len(gamma)), *gamma)
        fp.write(data)
        data = struct.pack(('%df' % len(beta)), *beta)
        fp.write(data)
        fp.close()

if __name__ == '__main__':
    layer_norm_f32()
    print("end")
