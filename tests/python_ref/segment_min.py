#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf


def segment_min_f32():
    para = []
    # init the input data and parameters
    batch      = int(np.random.randint(10, high=20, size=1))
    in_size_x  = int(np.random.randint(1, high=16, size=1))
    in_size_y  = int(np.random.randint(1, high=16, size=1))
    in_channel = int(np.random.randint(1, high=20, size=1))
    zero_point = int(np.random.randint(-6, high=6, size=1))
    std        = int(np.random.randint(1, high=10, size=1))

    src_in = np.random.normal(zero_point, std, (batch, in_size_y, in_size_x, in_channel))
    src_in = src_in.astype(np.float32)
    segment_ids = np.random.randint(0, high=batch-1, size=batch)
    segment_ids = np.sort(segment_ids)
    num_segments = segment_ids[batch-1] + 1

    out_calcu = tf.unsorted_segment_min(tf.convert_to_tensor(src_in), segment_ids, num_segments)

    sess = tf.Session()

    src_out = sess.run(out_calcu)

    src_in_1  = src_in.flatten()
    src_out_1 = src_out.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + len(segment_ids) + 5

    para.append(total_size)
    para.append(batch)
    para.append(in_size_y)
    para.append(in_size_x)
    para.append(in_channel)
    para.append(num_segments)

    with open("segment_min_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%di' % len(segment_ids)), *segment_ids)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    segment_min_f32()
    print("end")
