#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf


def reorg_f32():
    para = []
    # init the input data and parameters
    batch           = int(np.random.randint(1, high=2, size=1))
    channel_temp    = int(np.random.randint(1, high=2, size=1))
    height_temp     = int(np.random.randint(1, high=2, size=1))
    width_temp      = int(np.random.randint(1, high=2, size=1))
    stride          = int(np.random.randint(2, high=3, size=1))
    in_channel      = channel_temp * stride * stride
    in_height       = height_temp * stride
    in_width        = width_temp * stride

    zero_point = int(np.random.randint(-6, high=6, size=1))
    std        = int(np.random.randint(1, high=20, size=1))

    src_in = np.random.normal(zero_point, std, (batch, in_height, in_width, in_channel))
    src_in = src_in.astype(np.float32)

    out_calcu = tf.reshape(src_in, shape = (batch, int(in_height * stride), int(in_width * stride), int(in_channel / stride / stride)))
    out_calcu = tf.space_to_depth(out_calcu, block_size = stride)
    out_calcu = tf.reshape(out_calcu, shape = (batch, int(in_height / stride), int(in_width / stride), int(in_channel * stride * stride)))

    sess = tf.Session()

    src_out = sess.run(out_calcu)

    src_in_nchw = src_in.transpose(0, 3, 1, 2)      # nhwc -> nchw
    src_out_nchw = src_out.transpose(0, 3, 1, 2)

    src_in_1  = src_in_nchw.flatten()
    src_out_1 = src_out_nchw.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 5

    para.append(total_size)
    para.append(batch)
    para.append(in_channel)
    para.append(in_height)
    para.append(in_width)
    para.append(stride)

    print(para)
    print(src_out_nchw.shape)

    with open("reorg_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    reorg_f32()
    print("end")
