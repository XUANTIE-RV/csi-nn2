import os
import numpy as np
import caffe
import sys
import struct
import numpy as np


def roipooling_f32():

    model_file = './roipooling.prototxt'  # deploy文件
    pretrained = './roipooling.caffemodel'  # 训练的caffemodel

    net = caffe.Net(model_file, pretrained, caffe.TEST)   #加载model和network
    data = np.random.rand(1, 3, 10,10)
    net.blobs['data'].data[...] = data
    out = net.forward()

    para = []

    pooled_h = 1
    pooled_w = 2

    batch       = np.shape(net.blobs['data'].data[...])[0]
    in_channel  = np.shape(net.blobs['data'].data[...])[1]
    in_size_y   = np.shape(net.blobs['data'].data[...])[2]
    in_size_x   = np.shape(net.blobs['data'].data[...])[3]

    bias_channel = np.shape(net.blobs['conv1'].data[...])[1]
    bias_size_y   = np.shape(net.blobs['conv1'].data[...])[2]
    bias_size_x   = np.shape(net.blobs['conv1'].data[...])[3]


    out_size_y = np.shape(net.blobs['pooling'].data[...])[2]
    out_size_x = np.shape(net.blobs['pooling'].data[...])[3]


    src_in_1  = net.blobs['data'].data[...].flatten()
    src_rois = net.blobs['conv1'].data[...].flatten()
    src_out_1 = net.blobs['pooling'].data[...].flatten()


    spatial_scale=[0.0625]

    total_size = (len(src_in_1) + len(src_out_1) + len(src_rois) + len(spatial_scale)) + 11

    para.append(total_size)
    para.append(batch)
    para.append(in_channel)
    para.append(in_size_y)
    para.append(in_size_x)
    para.append(bias_channel)
    para.append(bias_size_y)
    para.append(bias_size_x)
    para.append(out_size_y)
    para.append(out_size_x)
    para.append(pooled_h)
    para.append(pooled_w)

    print(para)

    with open("roipooling_data_f32.bin", "wb") as fp:
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

    return 0


if __name__ == '__main__':
    roipooling_f32()
    print("end")



