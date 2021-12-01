#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf


def yuv_rgb_scale_f32():
    para = []
    # init the input data and parameters
    batch      = int(np.random.randint(1, high=4, size=1))
    in_size_x  = int(np.random.randint(128, high=512, size=1))
    in_size_y  = int(np.random.randint(128, high=512, size=1))

    yuv_images = np.random.random(size=(batch, in_size_y, in_size_x, 3))
    sub_images = [0, -0.5, -0.5]
    preprocessed_yuv_images = np.add(yuv_images, sub_images)
    preprocessed_yuv_images = preprocessed_yuv_images.astype(np.float32)
    
    out_calcu = tf.image.yuv_to_rgb(preprocessed_yuv_images)

    sess = tf.Session()

    src_out = sess.run(out_calcu)

    src_in_1  = preprocessed_yuv_images.flatten()
    src_out_1 = src_out.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 3

    para.append(total_size)
    para.append(batch)
    para.append(in_size_y)
    para.append(in_size_x)

    with open("yuv_rgb_scale_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    yuv_rgb_scale_f32()
    print("end")
