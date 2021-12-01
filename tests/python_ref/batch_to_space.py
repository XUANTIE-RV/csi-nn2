#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf

def batch_to_space_f32():
    para = []
    # init the input data and parameters
    batch_temp = int(np.random.randint(2, high=8, size=1))
    in_channel = int(np.random.randint(2, high=8, size=1))
    in_height  = int(np.random.randint(32, high=64, size=1))
    in_width   = int(np.random.randint(32, high=64, size=1))

    block_size  = int(np.random.randint(2, high=3, size=1))
    block_size2 = block_size * block_size
    batch = batch_temp * block_size2

    crop_top = crop_down = crop_left = crop_right = 0
    crop_top  = int(np.random.randint(0, high=in_height * block_size, size=1))
    crop_down = int(np.random.randint(0, high=in_height * block_size - crop_top, size=1))
    crop_left = int(np.random.randint(0, high=in_width * block_size, size=1))
    crop_right = int(np.random.randint(0, high=in_width * block_size - crop_left, size=1))

    zero_point = int(np.random.randint(-60000, high=60000, size=1))
    std        = int(np.random.randint(1, high=20, size=1))

    src_in = np.random.normal(zero_point, std, (batch, in_height, in_width, in_channel))
    out_calcu = tf.batch_to_space(src_in, crops=[[crop_top, crop_down], [crop_left, crop_right]], block_size = block_size)
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
    para.append(crop_top)
    para.append(crop_down)
    para.append(crop_left)
    para.append(crop_right)
    print(para)

    with open("batch_to_space_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    batch_to_space_f32()
    print("end")
