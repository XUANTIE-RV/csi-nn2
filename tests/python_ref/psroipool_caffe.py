#!/usr/bin/python
#-*- coding:utf-8 -*-
import sys
import struct
import random
import caffe
import numpy as np
from caffe import layers as L
from caffe.proto import caffe_pb2
import os

def psroipool_f32():
    para = []
    # init the input data and parameters
    out_dims    = int(np.random.randint(1, high=8, size=1))
    group_size  = int(np.random.randint(2, high=6, size=1))

    batch      = int(np.random.randint(1, high=2, size=1))
    in_channel = out_dims * group_size * group_size
    in_size_x  = int(np.random.randint(64, high=256, size=1))
    in_size_y  = int(np.random.randint(64, high=256, size=1))

    in_dim = [batch, in_channel, in_size_y, in_size_x]

    rois = []
    rois_num    = int(np.random.randint(2, high=8, size=1))
    for i in range(0, rois_num):
        batch_index = float(np.random.randint(0, high=batch, size=1))
        start_w     = float(np.random.randint(0, high=16, size=1))
        start_h     = float(np.random.randint(0, high=16, size=1))
        end_w       = float(np.random.randint(start_w, high=in_size_x, size=1))
        end_h       = float(np.random.randint(start_h, high=in_size_y, size=1))
        rois_one = [batch_index, start_w, start_h, end_w, end_h]
        rois.append(rois_one)

    rois_dim = [rois_num, 5]    # (image_id, x1, y1, x2, y2)

    scale = np.random.uniform(0, 2)
    spatial_scale = [scale]

    zero_point = int(np.random.randint(-6, high=6, size=1))
    std        = int(np.random.randint(1, high=20, size=1))
    src_in = np.random.normal(zero_point, std, size=in_dim)
    src_in = src_in.astype(np.float32)

    # caffe
    prototxt = "psroipool.prototxt"
    n = caffe.NetSpec()
    n.data = L.Input(shape = [dict(dim = in_dim)], ntop=1)
    n.rois_data = L.Input(shape = [dict(dim = rois_dim)], ntop=1)
    n.psroipool = L.PSROIPooling(n.data, n.rois_data, spatial_scale=spatial_scale[0], output_dim=out_dims, group_size=group_size)

    s = n.to_proto()
    with open(prototxt, "w") as f:
        f.write(str(s))

    psroipool_net = caffe.Net(prototxt, caffe.TEST)
    psroipool_net.blobs['data'].data[...] = src_in
    psroipool_net.blobs['rois_data'].data[...] = rois
    src_out = psroipool_net.forward()


    out_size_y = np.shape(psroipool_net.blobs['psroipool'].data[...])[2]    # out_size_y = group_size
    out_size_x = np.shape(psroipool_net.blobs['psroipool'].data[...])[3]    # out_size_x = group_size

    src_in_1 = psroipool_net.blobs['data'].data[...].flatten()
    src_rois = psroipool_net.blobs['rois_data'].data[...].flatten()
    # src_out_1 = psroipool_net.blobs['roipool'].data[...].flatten()
    src_out_1 = src_out["psroipool"].flatten()


    total_size = (len(src_in_1) + len(src_out_1) + len(src_rois)) + 9 + 1

    para.append(total_size)
    para.append(batch)
    para.append(in_channel)
    para.append(in_size_y)
    para.append(in_size_x)
    para.append(out_size_y)
    para.append(out_size_x)
    para.append(rois_num)
    para.append(out_dims)
    para.append(group_size)
    print(para)

    with open("psroipool_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(spatial_scale)), *spatial_scale)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_rois)), *src_rois)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

if __name__ == '__main__':
    psroipool_f32()
    os.remove('psroipool.prototxt')
    print("end")
