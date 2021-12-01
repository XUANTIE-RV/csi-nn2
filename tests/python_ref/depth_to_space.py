#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf
from torch import tensor

def depth_to_space_f32():
    para = []
    # init the input data and parameters
    batch      = int(np.random.randint(1, high=5, size=1))
    in_height  = int(np.random.randint(16, high=128, size=1))
    in_width   = int(np.random.randint(16, high=128, size=1))
    channel_temp = int(np.random.randint(4, high=16, size=1))
    block_size  = int(np.random.randint(2, high=5, size=1))
    block_size2 = block_size*block_size
    in_channel = channel_temp*block_size2

    zero_point = int(np.random.randint(-600, high=600, size=1))
    std        = int(np.random.randint(1, high=20, size=1))
    
    src_in = np.random.normal(zero_point, std, (batch, in_height, in_width, in_channel))
    # src_in = np.random.randint(0, 20, (batch, in_channel, in_height, in_width))

    src_in = src_in.astype(np.float32)

    # with tf.device('/gpu:0'):
    #     out_calcu = tf.depth_to_space(src_in, block_size = block_size, data_format="NCHW")
    #     sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    #     src_out = sess.run(out_calcu)

    out_calcu = tf.depth_to_space(src_in, block_size = block_size)
    sess = tf.Session()
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
    para.append(block_size)

    with open("depth_to_space_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    depth_to_space_f32()
    print("end")
