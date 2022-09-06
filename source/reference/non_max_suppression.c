/*
 * Copyright (C) 2016-2022 T-Head Semiconductor Co., Ltd. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* CSI-NN2 version 2.0.x */

#include "shl_ref.h"

static int find_max_score_idx(const float *scores, int *flag, int len)
{
    int res = 0;
    float max = FLT_MIN;
    for (int i = 0; i < len; i++) {
        if (scores[i] > max && !flag[i]) {
            max = scores[i];
            res = i;
        }
    }
    return res;
}

// box =  [y1, x1, y2, x2]
static float get_iou(const float *box1, const float *box2)
{
    // determine the (x, y)-coordinates of the intersection rectangle
    float x1 = fmax(box1[0], box2[0]);
    float y1 = fmax(box1[1], box2[1]);
    float x2 = fmin(box1[2], box2[2]);
    float y2 = fmin(box1[3], box2[3]);
    // compute the area of intersection rectangle
    float inter_area = fmax(0, x2 - x1) * fmax(0, y2 - y1);
    // compute the area of both the prediction and ground-truth rectangles
    float box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    float box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1]);
    ;
    // compute the intersection over union by taking the intersection area and
    // dividing it by the sum of prediction + ground-truth areas - the interesection area
    float iou = inter_area / (box1_area + box2_area - inter_area);
    return iou;
}

int shl_ref_non_max_suppression_std(struct csinn_tensor *input0, struct csinn_tensor *input1,
                                    struct csinn_tensor *output,
                                    struct csinn_non_max_suppression_params *params)
{
    float *boxes = (float *)input0->data;
    float *scores = (float *)input1->data;
    int *indices = (int *)output->data;

    float iou_threshold = params->iou_threshold;
    int max_output_size = params->max_output_size;

    int box_num = input1->dim[0];
    int box_num_exist = box_num;

    int *flag = (int *)shl_mem_alloc(box_num * sizeof(int));

    int box_cnt = 0;
    while (box_num_exist) {
        int max_box_idx = find_max_score_idx(scores, flag, box_num);
        flag[max_box_idx] = 1;
        box_num_exist--;
        *indices++ = max_box_idx;
        box_cnt++;
        if (box_cnt == max_output_size) {
            break;
        }
        for (int i = 0; i < box_num; i++) {
            if (!flag[i]) {
                float *box1_addr = boxes + 4 * max_box_idx;
                float *box2_addr = boxes + 4 * i;
                float iou = get_iou(box1_addr, box2_addr);
                if (iou > iou_threshold) {
                    flag[i] = 1;
                    box_num_exist--;
                }
            }
        }
    }
    shl_mem_free(flag);
    return CSINN_TRUE;
}
