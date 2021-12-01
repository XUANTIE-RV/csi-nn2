#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf

def non_max_suppression_f32():
    para = []
    # init the input data and parameters

    box_num  = int(np.random.randint(5, high=6, size=1))
    max_output_size = int(np.random.randint(1, high=6, size=1))
    iou_threshold = 0.5

    boxes  = np.array([[1,2,3,4],[1,3,3,4],[1,3,4,4],[1,1,4,4],[1,1,3,4]], dtype = np.float32)
    scores = np.array([0.4,0.5,0.72,0.9,0.45], dtype = np.float32)

    out_calcu = tf.image.non_max_suppression(boxes = boxes, scores = scores, iou_threshold = iou_threshold, max_output_size = max_output_size)
    with tf.Session() as sess:
        indices = sess.run(out_calcu)
    # indices = [3,2,0]
    out_size = indices.shape[0]
    boxes_1  = boxes.flatten()
    scores_1 = scores.flatten()
    indices_1 = indices.flatten()

    total_size = (len(boxes_1) + len(scores_1) + len(indices_1)) + 4

    para.append(total_size)
    para.append(box_num)
    para.append(max_output_size)
    para.append(out_size)
    print(para)

    with open("non_max_suppression_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack('1f' , iou_threshold)
        fp.write(data)
        data = struct.pack(('%df' % len(boxes_1)), *boxes_1)
        fp.write(data)
        data = struct.pack(('%df' % len(scores_1)), *scores_1)
        fp.write(data)
        data = struct.pack(('%di' % len(indices_1)), *indices_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    non_max_suppression_f32()
    print("end")
