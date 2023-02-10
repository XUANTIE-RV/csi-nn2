/*
 * Copyright (C) 2016-2023 T-Head Semiconductor Co., Ltd. All rights reserved.
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

#include "shl_c920.h"

static void qsort_desc_fp32(int32_t *box_idx, float *scores, int left, int right)
{
    int i = left;
    int j = right;
    float p = scores[(left + right) / 2];
    while (i <= j) {
        while (scores[i] > p) {
            i++;
        }
        while (scores[j] < p) {
            j--;
        }
        if (i <= j) {
            int32_t tmp_idx = box_idx[i];
            box_idx[i] = box_idx[j];
            box_idx[j] = tmp_idx;
            float tmp_score = scores[i];
            scores[i] = scores[j];
            scores[j] = tmp_score;
            i++;
            j--;
        }
    }
    if (j > left) {
        qsort_desc_fp32(box_idx, scores, left, j);
    }
    if (i < right) {
        qsort_desc_fp32(box_idx, scores, i, right);
    }
}

static float get_iou_fp32(const struct shl_yolov5_box box1, const struct shl_yolov5_box box2)
{
    float x1 = fmax(box1.x1, box2.x1);
    float y1 = fmax(box1.y1, box2.y1);
    float x2 = fmin(box1.x2, box2.x2);
    float y2 = fmin(box1.y2, box2.y2);
    float inter_area = fmax(0, x2 - x1) * fmax(0, y2 - y1);
    float iou = inter_area / (box1.area + box2.area - inter_area);
    return iou;
}

static int non_max_suppression_fp32(struct shl_yolov5_box *boxes, int32_t *indices, float iou_thres,
                                    int box_num)
{
    float *scores = (float *)shl_mem_alloc(box_num * sizeof(float));
    int32_t *box_indices = (int32_t *)shl_mem_alloc(box_num * sizeof(int32_t));
    for (int i = 0; i < box_num; i++) {
        scores[i] = boxes[i].score;
        box_indices[i] = i;
    }
    qsort_desc_fp32(box_indices, scores, 0, box_num - 1);

    int box_cnt = 0;
    for (int i = 0; i < box_num; i++) {
        bool keep = true;
        int32_t box_idx = box_indices[i];
        struct shl_yolov5_box box1 = boxes[box_idx];
        for (int j = 0; j < box_cnt; j++) {
            struct shl_yolov5_box box2 = boxes[indices[j]];
            float iou = get_iou_fp32(box1, box2);
            if (iou > iou_thres) {
                keep = false;
            }
        }
        if (keep) {
            indices[box_cnt++] = box_idx;
        }
    }

    shl_mem_free(box_indices);
    shl_mem_free(scores);

    return box_cnt;
}

static inline float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

static void proposal_fp32(struct csinn_tensor *input, const float *anchors, int stride,
                          float conf_thres, struct shl_yolov5_box *box, int *box_num)
{
    /* [1, 255, y, x] -> [1, 3, 85, y, x] */
    float *data = (float *)input->data;
    const int num_anchors = 3;
    const int inner_size = input->dim[1] / 3;
    const int grid_x = input->dim[2];
    const int grid_y = input->dim[3];
    const int grid_size = grid_x * grid_y;

    /* sigmoid(x) > t  <=>  x > -ln(1/t-1) */
    float threshold = -log(1.f / conf_thres - 1.f);

    float *feat_tmp = (float *)shl_mem_alloc(inner_size * sizeof(float));

    for (int q = 0; q < num_anchors; q++) {
        const float *feat = data + q * inner_size * grid_size;
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];
        for (int i = 0; i < grid_x; i++) {
            for (int j = 0; j < grid_y; j++) {
                const float *featptr = feat + (i * grid_y + j);
                float box_score = featptr[4 * grid_size];
                if (box_score <= threshold) {
                    continue;
                }

                int k = 0;
                while (k < inner_size) {
                    int vl = vsetvl_e32m4(inner_size - k);
                    vfloat32m4_t _tmp =
                        vlse32_v_f32m4(featptr + k * grid_size, grid_size * sizeof(float), vl);
                    vse32_v_f32m4(feat_tmp + k, _tmp, vl);
                    k += vl;
                }

                float max_score = -FLT_MAX;
                int max_idx = -1;
                for (int k = 5; k < inner_size; k++) {
                    float score = feat_tmp[k];
                    if (score > max_score) {
                        max_score = score;
                        max_idx = k - 5;
                    }
                }

                float box_conf = sigmoid(box_score);
                float class_conf = box_conf * sigmoid(max_score);
                if (class_conf <= conf_thres) {
                    continue;
                }

                float dx = sigmoid(feat_tmp[0]);
                float dy = sigmoid(feat_tmp[1]);
                float dw = sigmoid(feat_tmp[2]);
                float dh = sigmoid(feat_tmp[3]);

                float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                float pb_w = powf(dw * 2.f, 2) * anchor_w;
                float pb_h = powf(dh * 2.f, 2) * anchor_h;

                box[*box_num].x1 = pb_cx - pb_w * 0.5f;
                box[*box_num].y1 = pb_cy - pb_h * 0.5f;
                box[*box_num].x2 = pb_cx + pb_w * 0.5f;
                box[*box_num].y2 = pb_cy + pb_h * 0.5f;
                box[*box_num].label = max_idx;
                box[*box_num].score = class_conf;
                box[*box_num].area =
                    (box[*box_num].x2 - box[*box_num].x1) * (box[*box_num].y2 - box[*box_num].y1);
                *box_num += 1;
            }
        }
    }

    shl_mem_free(feat_tmp);
}

static void proposal_uint8(struct csinn_tensor *input, const float *anchors, int stride,
                           float conf_thres, struct shl_yolov5_box *box, int *box_num)
{
    /* [1, 255, y, x] -> [1, 3, 85, y, x] */
    uint8_t *data = (uint8_t *)input->data;
    const int num_anchors = 3;
    const int inner_size = input->dim[1] / 3;
    const int grid_x = input->dim[2];
    const int grid_y = input->dim[3];
    const int grid_size = grid_x * grid_y;

    float scale = input->qinfo->scale;
    int32_t zero_point = input->qinfo->zero_point;

    /* sigmoid(x) > t  <=>  x > -ln(1/t-1) */
    float threshold = -log(1.f / conf_thres - 1.f);
    uint8_t threshold_u8 = nearbyint(threshold / scale + zero_point);

    uint8_t *feat_u8 = (uint8_t *)shl_mem_alloc(inner_size * sizeof(uint8_t));
    float *feat_tmp = (float *)shl_mem_alloc(inner_size * sizeof(float));

    for (int q = 0; q < num_anchors; q++) {
        const uint8_t *feat = data + q * inner_size * grid_size;
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];
        for (int i = 0; i < grid_x; i++) {
            for (int j = 0; j < grid_y; j++) {
                const uint8_t *featptr = feat + (i * grid_y + j);
                uint8_t box_score_u8 = featptr[4 * grid_size];
                if (box_score_u8 <= threshold_u8) {
                    continue;
                }

                int k = 0;
                while (k < inner_size) {
                    int vl = vsetvl_e8m1(inner_size - k);
                    vuint8m1_t _u8 =
                        vlse8_v_u8m1(featptr + k * grid_size, grid_size * sizeof(uint8_t), vl);
                    vse8_v_u8m1(feat_u8 + k, _u8, vl);
                    k += vl;
                }
                shl_c920_u8_to_f32(feat_u8, feat_tmp, zero_point, &scale, inner_size);

                float max_score = -FLT_MAX;
                int max_idx = -1;
                for (int k = 5; k < inner_size; k++) {
                    float score = feat_tmp[k];
                    if (score > max_score) {
                        max_score = score;
                        max_idx = k - 5;
                    }
                }

                float box_score = ((float)box_score_u8 - zero_point) * scale;
                float box_conf = sigmoid(box_score);
                float class_conf = box_conf * sigmoid(max_score);
                if (class_conf <= conf_thres) {
                    continue;
                }

                float dx = sigmoid(feat_tmp[0]);
                float dy = sigmoid(feat_tmp[1]);
                float dw = sigmoid(feat_tmp[2]);
                float dh = sigmoid(feat_tmp[3]);

                float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                float pb_w = powf(dw * 2.f, 2) * anchor_w;
                float pb_h = powf(dh * 2.f, 2) * anchor_h;

                box[*box_num].x1 = pb_cx - pb_w * 0.5f;
                box[*box_num].y1 = pb_cy - pb_h * 0.5f;
                box[*box_num].x2 = pb_cx + pb_w * 0.5f;
                box[*box_num].y2 = pb_cy + pb_h * 0.5f;
                box[*box_num].label = max_idx;
                box[*box_num].score = class_conf;
                box[*box_num].area =
                    (box[*box_num].x2 - box[*box_num].x1) * (box[*box_num].y2 - box[*box_num].y1);
                *box_num += 1;
            }
        }
    }

    shl_mem_free(feat_u8);
    shl_mem_free(feat_tmp);
}

int shl_c920_detect_yolov5_postprocess(struct csinn_tensor **input_tensors,
                                       struct shl_yolov5_box *out, struct shl_yolov5_params *params)
{
    /* [1, 255, x, y] */
    struct csinn_tensor *input0 = input_tensors[0];
    struct csinn_tensor *input1 = input_tensors[1];
    struct csinn_tensor *input2 = input_tensors[2];
    if (!((input0->dtype == CSINN_DTYPE_FLOAT32 && input1->dtype == CSINN_DTYPE_FLOAT32 &&
           input2->dtype == CSINN_DTYPE_FLOAT32) ||
          (input0->dtype == CSINN_DTYPE_UINT8 && input1->dtype == CSINN_DTYPE_UINT8 &&
           input2->dtype == CSINN_DTYPE_UINT8))) {
        shl_debug_error("yolov5 posprocess unsupported dtype: %d", input0->dtype);
        return 0;
    }

    const int max_box = (input0->dim[2] * input0->dim[3] + input1->dim[2] * input1->dim[3] +
                         input2->dim[2] * input2->dim[3]) *
                        3;

    const float conf_thres = params->conf_thres;
    const float iou_thres = params->iou_thres;
    const int32_t *strides = params->strides;
    const float *anchors = params->anchors;

    if (!(conf_thres > 0.f && conf_thres < 1.f)) {
        shl_debug_error("Confidence threshold must be between 0 and 1!");
        return 0;
    }

    struct shl_yolov5_box proposals[max_box];
    int box_num = 0;
    if (input0->dtype == CSINN_DTYPE_FLOAT32 && input1->dtype == CSINN_DTYPE_FLOAT32 &&
        input2->dtype == CSINN_DTYPE_FLOAT32) {
        proposal_fp32(input0, anchors, strides[0], conf_thres, proposals, &box_num);
        proposal_fp32(input1, anchors + 6, strides[1], conf_thres, proposals, &box_num);
        proposal_fp32(input2, anchors + 12, strides[2], conf_thres, proposals, &box_num);
    } else if (input0->dtype == CSINN_DTYPE_UINT8 && input1->dtype == CSINN_DTYPE_UINT8 &&
               input2->dtype == CSINN_DTYPE_UINT8) {
        proposal_uint8(input0, anchors, strides[0], conf_thres, proposals, &box_num);
        proposal_uint8(input1, anchors + 6, strides[1], conf_thres, proposals, &box_num);
        proposal_uint8(input2, anchors + 12, strides[2], conf_thres, proposals, &box_num);
    }

    if (box_num == 0) {
        return 0;
    }

    int32_t *indices = (int32_t *)shl_mem_alloc(box_num * sizeof(int32_t));
    int num = non_max_suppression_fp32(proposals, indices, iou_thres, box_num);

    for (int i = 0; i < num; i++) {
        int idx = indices[i];
        out[i].label = proposals[idx].label;
        out[i].score = proposals[idx].score;
        out[i].x1 = proposals[idx].x1;
        out[i].y1 = proposals[idx].y1;
        out[i].x2 = proposals[idx].x2;
        out[i].y2 = proposals[idx].y2;
    }

    shl_mem_free(indices);
    return num;
}
