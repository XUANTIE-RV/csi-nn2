#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
from torch import tensor
from torch.nn import AdaptiveMaxPool2d

def getpackn(test_dtype, test_vlen):
    if int(test_dtype) == 8:
        return int(test_vlen)/int(test_dtype)/2
    else:
        return int(test_vlen)/int(test_dtype)

def global_maxpool2d_f32(test_dtype, test_vlen, test_type):
    para = []
    # init the input data and parameters
    batch      = int(np.random.randint(1, high=2, size=1))
    in_size_x  = int(np.random.randint(16, high=32, size=1))
    in_size_y  = int(np.random.randint(16, high=32, size=1))
    in_channel = int(np.random.randint(1, high=16, size=1))

    out_height  = int(np.random.randint(1, high=2, size=1))
    out_width  = int(np.random.randint(1, high=2, size=1))

    zero_point = int(np.random.randint(-6, high=6, size=1))
    std        = int(np.random.randint(1, high=20, size=1))

    packn = int(getpackn(test_dtype, test_vlen))
    n = int(np.random.randint(1, high=2, size=1))

    if test_type == "packn":
        in_channel    = int(n*packn)
    elif test_type == "pack1":
        in_channel    = int(n*packn) + 1

    src_in = np.random.normal(zero_point, std, (batch, in_channel, in_size_y, in_size_x))

    t_src_in  = tensor(src_in)
    gmp = AdaptiveMaxPool2d((out_height, out_width))
    t_src_out = gmp(t_src_in).numpy()


    #permute nchw to nhwc
    src_in_nhwc = np.transpose(src_in, [0, 2, 3, 1])
    out_nhwc    = np.transpose(t_src_out, [0, 2, 3, 1])


    src_in_1  = src_in_nhwc.flatten()
    src_out_1 = out_nhwc.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 6

    para.append(total_size)
    para.append(batch)
    para.append(in_size_y)
    para.append(in_size_x)
    para.append(in_channel)
    para.append(out_height)
    para.append(out_width)

    print(para)


    with open("global_maxpool_nhwc_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    test_dtype = sys.argv[1]
    test_vlen = sys.argv[2]
    test_type = sys.argv[3]
    global_maxpool2d_f32(test_dtype, test_vlen, test_type)
    print("end")
