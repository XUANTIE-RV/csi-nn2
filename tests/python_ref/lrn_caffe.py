#!/usr/bin/python
#-*- coding:utf-8 -*-
import sys
import struct
import numpy as np
import random
import caffe
import numpy as np
from caffe import layers as L
from caffe.proto import caffe_pb2
import os

def local_response_normalization_f32():
    para_int  = []
    para_float = []

    # init the input data and parameters
    batch       = int(np.random.randint(1, high=4, size=1))
    in_channel  = int(np.random.randint(16, high=32, size=1))
    in_size_y   = int(np.random.randint(32, high=64, size=1))
    in_size_x   = int(np.random.randint(32, high=64, size=1))

    zero_point = int(np.random.randint(-6, high=6, size=1))
    std        = int(np.random.randint(1, high=20, size=1))
    src_in = np.random.normal(zero_point, std, size=(batch, in_channel, in_size_y, in_size_x))
    src_in = src_in.astype(np.float32)

    # alpha = np.random.randint(1, high=5, size=1) * 0.5
    bias = float(np.random.uniform(1, high=5, size=1))
    alpha = float(np.random.uniform(1e-5, high=1e-3, size=1))
    beta = float(np.random.uniform(0.5, high=1, size=1))
    depth = int(np.random.randint(3, high=6, size=1))

    # caffe
    prototxt = "lrn.prototxt"
    net = caffe.NetSpec() 
    net["data"] = L.Input(shape=[dict(dim=[batch, in_channel, in_size_y, in_size_x])], ntop=1)
    net["lrn"] = L.LRN(net["data"],
                                norm_region = 1,    # 0:ACROSS_CHANNELS  1:WITHIN_CHANNEL
                                local_size=depth * 2 + 1,
                                alpha=alpha,
                                beta=beta,
                                k=bias)
    with open(prototxt, 'w') as f:
        f.write(str(net.to_proto()))

    lrn_net = caffe.Net(prototxt, caffe.TEST)
    lrn_net.blobs['data'].data[...] = src_in
    src_out = lrn_net.forward()

    src_out   = np.array(src_out["lrn"])

    src_in_1 = src_in.flatten()
    src_out_1 = src_out.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 8

    para_int.append(total_size)
    para_int.append(batch)
    para_int.append(in_channel)
    para_int.append(in_size_y)
    para_int.append(in_size_x)
    para_int.append(depth)
    para_float.append(bias)
    para_float.append(alpha)
    para_float.append(beta)
    print(para_int)
    print(para_float)

    with open("lrn_caffe_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para_int)), *para_int)
        fp.write(data)
        data = struct.pack(('%df' % len(para_float)), *para_float)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    local_response_normalization_f32()
    os.remove('lrn.prototxt')
    print("end")
