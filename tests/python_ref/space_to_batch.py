#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf

def space_to_batch_f32():
    para = []
    # init the input data and parameters
    batch      = int(np.random.randint(4, high=8, size=1))
    in_channel = int(np.random.randint(2, high=8, size=1))
    in_height  = int(np.random.randint(32, high=128, size=1))
    in_width   = int(np.random.randint(32, high=128, size=1))
    block_size  = int(np.random.randint(2, high=5, size=1))

    pad_top = pad_down = pad_left = pad_right = 0
    pad_h = in_height - int(in_height / block_size) * block_size
    if(pad_h != 0):
        pad_h      = (int(in_height / block_size) + 1) * block_size - in_height
        pad_top   = int(np.random.randint(0, high=pad_h+1, size=1))
        pad_down  = pad_h - pad_top

    pad_w = in_width - int(in_width / block_size) * block_size
    if(pad_w != 0):
        pad_w      = (int(in_width / block_size) + 1) * block_size - in_width
        pad_left    = int(np.random.randint(0, high=pad_w+1, size=1))
        pad_right   = pad_w - pad_left

    zero_point = int(np.random.randint(-600, high=600, size=1))
    std        = int(np.random.randint(1, high=20, size=1))

    src_in = np.random.normal(zero_point, std, (batch, in_height, in_width, in_channel))

    out_calcu = tf.space_to_batch(src_in, paddings=[[pad_top, pad_down], [pad_left, pad_right]], block_size=block_size)

    sess = tf.Session()
    src_out = sess.run(out_calcu)

    src_in_nchw = src_in.transpose(0, 3, 1, 2)
    src_out_nchw = src_out.transpose(0, 3, 1, 2)

    src_in_1  = src_in_nchw.flatten()
    src_out_1 = src_out_nchw.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 9

    para.append(total_size)
    para.append(batch)
    para.append(in_channel)
    para.append(in_height)
    para.append(in_width)
    para.append(block_size)
    para.append(pad_top)
    para.append(pad_down)
    para.append(pad_left)
    para.append(pad_right)
    print(para)

    with open("space_to_batch_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    space_to_batch_f32()
    print("end")
