#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf


def arange_f32():
    para = []
    # init the input data and parameters
    start = int(np.random.randint(1, high=512, size=1))
    stop = int(np.random.randint(532, high=1024, size=1))
    step = int(np.random.randint(1, high=20, size=1))


    out_calcu = tf.keras.backend.arange(start,stop,step)
    with tf.Session() as sess:
        src_out = sess.run(out_calcu)
    src_out_1 = src_out.flatten()

    total_size = len(src_out_1) + 4

    para.append(total_size)
    para.append(start)
    para.append(stop)
    para.append(step)
    para.append(len(src_out_1))

    with open("arange_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    arange_f32()
    print("end")
