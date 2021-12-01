#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import random
import tensorflow as tf

def slice_f32():
    para = []
    # init the input data and parameters
    batch = int(np.random.randint(2, high=6, size=1))
    in_size_x = int(np.random.randint(128, high=512, size=1))
    in_size_y = int(np.random.randint(128, high=512, size=1))
    in_channel = int(np.random.randint(2, high=64, size=1))

    begin = []
    size = []
    end = []
    siz0 = int(np.random.randint(1, high=batch, size=1))
    size.append(siz0)
    gap0 = batch - siz0
    if (gap0 != 0):
        begin0  = int(np.random.randint(0, high=gap0, size=1))
        end0    = begin0 + siz0
        begin.append(begin0)
        end.append(end0)
    else:
        begin.append(0)
        end.append(batch)
    
    siz1 = int(np.random.randint(1, high=in_size_y, size=1))
    size.append(siz1)
    gap1 = in_size_y - siz1
    if(gap1 != 0):
        begin1  = int(np.random.randint(0, high=gap1, size=1))
        end1    = begin1 + siz1
        begin.append(begin1)
        end.append(end1)
    else:
        begin.append(0)
        end.append(in_size_x)

    siz2 = int(np.random.randint(1, high=in_size_x, size=1))
    size.append(siz2)
    gap2 = in_size_x - siz2
    if(gap2 != 0):
        begin2  = int(np.random.randint(0, high=gap2, size=1))
        end2    = begin2 + siz2
        begin.append(begin2)
        end.append(end2)
    else:
        begin.append(0)
        end.append(in_size_y)

    siz3 = int(np.random.randint(1, high=in_channel, size=1))
    size.append(siz3)
    gap3 = in_channel - siz3
    if(gap3 != 0):
        begin3  = int(np.random.randint(0, high=gap3, size=1))
        end3    = begin3 + siz3
        begin.append(begin3)
        end.append(end3)
    else:
        begin.append(0)
        end.append(in_channel)

    zero_point = int(np.random.randint(-60000, high=60000, size=1))
    std        = int(np.random.randint(1, high=20, size=1))

    src_in = np.random.normal(zero_point, std, (batch, in_size_y, in_size_x , in_channel))
    out_calcu = tf.slice(src_in, begin=begin, size=size)

    sess = tf.Session()

    src_out = sess.run(out_calcu)
    
    src_in_1  = src_in.flatten()
    src_out_1 = src_out.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + len(begin) + len(end) + 4

    para.append(total_size)
    para.append(batch)
    para.append(in_size_y)
    para.append(in_size_x)
    para.append(in_channel)

    with open("slice_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%di' % len(begin)), *begin)
        fp.write(data)
        data = struct.pack(('%di' % len(end)), *end)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    slice_f32()
    print("end")
