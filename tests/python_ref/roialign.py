#!/usr/bin/python
#-*- coding:utf-8 -*-
import sys
import struct
import random
import numpy as np
import torchvision
from torch import tensor
def roialign_f32():
    para = []
    # init the input data and parameters
    batch      = int(np.random.randint(1, high=2, size=1))
    in_channel = int(np.random.randint(4, high=16, size=1))
    in_size_x  = int(np.random.randint(32, high=64, size=1))
    in_size_y  = int(np.random.randint(32, high=64, size=1))
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
    spatial_scale = np.random.uniform(0, 2)
    sampling_ratio = np.random.randint(-1, 2 ,1)
    zero_point = int(np.random.randint(-6, high=6, size=1))
    std        = int(np.random.randint(1, high=20, size=1))
    src_in = np.random.normal(zero_point, std, size=in_dim)
    src_in = src_in.astype(np.float32)


    t_src_in  = tensor(src_in)
    t_boxes = tensor(rois)
    t_size = tensor([pooled_height,pooled_width])


    t_src_out = torchvision.ops.roi_align(t_src_in, t_boxes, t_size, spatial_scale=spatial_scale, sampling_ratio=sampling_ratio).numpy()
    

    out_size_y = np.shape(t_src_out)[2]
    out_size_x = np.shape(t_src_out)[3]
    spatial_scale = [spatial_scale]
 
    
    src_in_1 = src_in.flatten()
    src_rois = t_boxes.numpy().flatten()
    src_out_1 = t_src_out.flatten()

    total_size = (len(src_in_1) + len(src_out_1) + len(src_rois)) + 9 + 2
    para.append(total_size)
    para.append(batch)
    para.append(in_channel)
    para.append(in_size_y)
    para.append(in_size_x)
    para.append(out_size_y)
    para.append(out_size_x)
    para.append(rois_num)
    para.append(pooled_height) 
    para.append(pooled_width)  
    print(para)

    with open("roialign_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(spatial_scale)), *spatial_scale)
        fp.write(data)
        data = struct.pack(('%di' % len(sampling_ratio)), *sampling_ratio)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_rois)), *src_rois)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()
if __name__ == '__main__':
    roialign_f32()
    print("end")