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

def roipool_f32():
    para = []
    # init the input data and parameters
    batch      = int(np.random.randint(1, high=2, size=1))
    in_channel = int(np.random.randint(4, high=16, size=1))
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

    pooled_height = int(np.random.randint(2, high=16, size=1))
    pooled_width  = int(np.random.randint(2, high=16, size=1))
    scale = np.random.uniform(0, 2)
    spatial_scale = [scale]


    zero_point = int(np.random.randint(-6, high=6, size=1))
    std        = int(np.random.randint(1, high=20, size=1))
    src_in = np.random.normal(zero_point, std, size=in_dim)
    src_in = src_in.astype(np.float32)

    # caffe
    prototxt = "roipool.prototxt"
    n = caffe.NetSpec()
    n.data = L.Input(shape = [dict(dim = in_dim)], ntop=1)
    n.rois_data = L.Input(shape = [dict(dim = rois_dim)], ntop=1)
    n.roipool = L.ROIPooling(n.data, n.rois_data, pooled_h = pooled_height, pooled_w = pooled_width, spatial_scale = spatial_scale[0])

    s = n.to_proto()
    with open(prototxt, "w") as f:
        f.write(str(s))

    roipool_net = caffe.Net(prototxt, caffe.TEST)
    roipool_net.blobs['data'].data[...] = src_in
    roipool_net.blobs['rois_data'].data[...] = rois
    src_out = roipool_net.forward()


    out_size_y = np.shape(roipool_net.blobs['roipool'].data[...])[2]    # out_size_y = pooled_height
    out_size_x = np.shape(roipool_net.blobs['roipool'].data[...])[3]    # out_size_x = pooled_width

    src_in_1 = roipool_net.blobs['data'].data[...].flatten()
    src_rois = roipool_net.blobs['rois_data'].data[...].flatten()
    # src_out_1 = roipool_net.blobs['roipool'].data[...].flatten()
    src_out_1 = src_out["roipool"].flatten()


    total_size = (len(src_in_1) + len(src_out_1) + len(src_rois)) + 9 + 1

    para.append(total_size)
    para.append(batch)
    para.append(in_channel)
    para.append(in_size_y)
    para.append(in_size_x)
    para.append(out_size_y)
    para.append(out_size_x)
    para.append(rois_num)
    para.append(pooled_height) # out_size_y
    para.append(pooled_width)  # out_size_x
    print(para)

    with open("roipool_data_f32.bin", "wb") as fp:
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
    roipool_f32()
    os.remove('roipool.prototxt')
    print("end")
