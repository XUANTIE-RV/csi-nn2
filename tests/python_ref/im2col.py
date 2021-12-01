#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf


def im2col(input_data, filter_h, filter_w, stride_h=1, stride_w=1, pad=[0, 0, 0, 0]):
    N, C, H, W = input_data.shape
    out_h = (H + pad[0] + pad[1] - filter_h) // stride_h + 1
    out_w = (W + pad[2] + pad[3] - filter_w) // stride_w + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad[0], pad[1]), (pad[2], pad[3])], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride_h*out_h
        for x in range(filter_w):
            x_max = x + stride_w*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride_h, x:x_max:stride_w]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def im2col_f32():
    para = []
    # init the input data and parameters
    batch       = int(np.random.randint(2, high=3, size=1))
    in_channel  = int(np.random.randint(2, high=3, size=1))
    in_height   = int(np.random.randint(16, high=32, size=1))
    in_width    = int(np.random.randint(16, high=32, size=1))

    stride_h    = int(np.random.randint(1, high=5, size=1))
    stride_w    = int(np.random.randint(1, high=5, size=1))

    kernel_h    = int(np.random.randint(stride_h, high=8, size=1))
    kernel_w    = int(np.random.randint(stride_w, high=8, size=1))

    dilation_h  = int(np.random.randint(1, high=2, size=1))
    dilation_w  = int(np.random.randint(1, high=2, size=1))

    kernel_h_t  = kernel_h + (kernel_h - 1) * (dilation_h - 1)
    kernel_w_t  = kernel_w + (kernel_w - 1) * (dilation_w - 1)

    pad_left  = pad_right = 0
    pad_top   = pad_down  = 0

    pad_h      = (in_height - kernel_h_t) -  int((in_height - kernel_h_t) / stride_h) * stride_h
    if(pad_h != 0):
        pad_h      = int((in_height - kernel_h_t) / stride_h) * stride_h + stride_h - (in_height - kernel_h_t)
        pad_top    = int(np.random.randint(0, high=pad_h, size=1))
        pad_down   = pad_h - pad_top

    pad_w      = (in_width - kernel_w_t) -  int((in_width - kernel_w_t) / stride_w) * stride_w
    if(pad_w !=0):
        pad_w      = int((in_width - kernel_w_t) / stride_w) * stride_w + stride_w - (in_width - kernel_w_t)
        pad_left   = int(np.random.randint(0, high=pad_w, size=1))
        pad_right  = pad_w - pad_left

    zero_point = int(np.random.randint(-600, high=600, size=1))
    std        = int(np.random.randint(1, high=20, size=1))

    src_in_nchw = np.random.normal(zero_point, std, (batch, in_channel, in_height, in_width))
    src_in_nhwc = src_in_nchw.transpose(0, 2, 3, 1)
    src_in = src_in_nchw

    out_calcu = im2col(src_in_nchw, kernel_h, kernel_w, stride_h, stride_w, [pad_top, pad_down, pad_left, pad_right])
    src_out_1 = out_calcu.transpose(1, 0)   # output: row x col = N*H*W x C*k*k  ->  C*k*k x N*H*W

    src_in_1  = src_in.flatten()
    src_out_1 = src_out_1.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 12

    para.append(total_size)
    para.append(batch)
    para.append(in_channel)
    para.append(in_height)
    para.append(in_width)
    para.append(kernel_h)
    para.append(kernel_w)
    para.append(stride_h)
    para.append(stride_w)
    para.append(pad_left)
    para.append(pad_right)
    para.append(pad_top)
    para.append(pad_down)
    print(para)

    with open("im2col_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    im2col_f32()
    print("end")