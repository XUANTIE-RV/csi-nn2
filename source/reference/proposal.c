/*
 * Copyright (C) 2016-2020 C-SKY Limited. All rights reserved.
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

#include "csi_nn.h"
#include "csi_utils.h"
#include <math.h>
#define MAX(a, b) (a > b ? a : b)
#define MIN(a, b) (a > b ? b : a)

struct bbox {
  float x1;
  float y1;
  float x2;
  float y2;
};

static struct bbox reg_iou(float x1, float y1, float x2, float y2, float dx1,
                           float dy1, float dx2, float dy2)
{
  struct bbox pred;
  pred.x1 = x1 + dx1;
  pred.y1 = y1 + dy1;
  pred.x2 = x2 + dx2;
  pred.y2 = y1 + dy2;
  return pred;
}

static struct bbox reg_bbox(float x1, float y1, float x2, float y2, float dx, float dy,
                            float dw, float dh)
{
  float bbox_w = x2 - x1 + 1.0;
  float bbox_h = y2 - y1 + 1.0;
  float ctr_x = x1 + 0.5 * (bbox_w - 1.0);
  float ctr_y = y1 + 0.5 * (bbox_h - 1.0);

  float pred_ctr_x = dx * bbox_w + ctr_x;
  float pred_ctr_y = dy * bbox_h + ctr_y;
  float pred_w = exp(dw) * bbox_w;
  float pred_h = exp(dh) * bbox_h;

  struct bbox pred;
  pred.x1 = pred_ctr_x - 0.5 * (pred_w - 1.0);
  pred.y1 = pred_ctr_y - 0.5 * (pred_h - 1.0);
  pred.x2 = pred_ctr_x + 0.5 * (pred_w - 1.0);
  pred.y2 = pred_ctr_y + 0.5 * (pred_h - 1.0);
  return pred;
}

static struct bbox generate_anchor(float ratio, float scale, int32_t base_size)
{
  float w, h;
  w = h = (float)base_size;
  float x_ctr = 0.5 * (w - 1.0);
  float y_ctr = 0.5 * (h - 1.0);
  float size = w * h;
  int size_ratios = floor(size / ratio);
  int new_w = floor(sqrt(size_ratios) + 0.5) * scale;
  int new_h = floor((new_w / scale * ratio) + 0.5) * scale;
  struct bbox _bbox;
  _bbox.x1 = x_ctr - 0.5 * (new_w - 1.0);
  _bbox.y1 = y_ctr - 0.5 * (new_h - 1.0);
  _bbox.x2 = x_ctr + 0.5 * (new_w - 1.0);
  _bbox.y2 = y_ctr + 0.5 * (new_h - 1.0);

  return _bbox;
}

static float *predict_bbox(struct csi_tensor *cls_prob_tensor,
                           struct csi_tensor *bbox_pred_tensor,
                           struct csi_tensor *im_info_tensor, float *ratios,
                           int32_t ratios_num, float *scales, int32_t scales_num,
                           int32_t feature_stride, int32_t iou_loss,
                           int32_t rpn_min_size)
{
  int len_scales = scales_num;
  int len_ratios = ratios_num;
  int batch = cls_prob_tensor->dim[0];
  int num_anchors = cls_prob_tensor->dim[1];
  int height = cls_prob_tensor->dim[2];
  int width = cls_prob_tensor->dim[3];
  num_anchors = num_anchors / 2;

  float *cls_prob = cls_prob_tensor->data;
  float *bbox_pred = bbox_pred_tensor->data;
  float *im_info = im_info_tensor->data;

  float *output =
      malloc(batch * height * width * num_anchors * 5 * sizeof(float));

  for (int i = 0; i < batch * height * width; i++) {
    int w = i % width;
    int h = (i / width) % height;
    int b = (i / width) / height;

    for (int k = 0; k < num_anchors; k++) {
      int out_index = i * num_anchors + k;
      float ratio = ratios[k / scales_num];
      float scale = scales[k % scales_num];
      struct bbox anchor = generate_anchor(ratio, scale, feature_stride);
      int im_height = im_info[b * 3];
      int im_width = im_info[b * 3 + 1];
      int x1 = anchor.x1 + w * feature_stride;
      int y1 = anchor.y1 + h * feature_stride;
      int x2 = anchor.x2 + w * feature_stride;
      int y2 = anchor.y2 + h * feature_stride;

      float *delta = malloc(4 * sizeof(float));
      for (int j = 0; j < 4; j++) {
        delta[j] =
            bbox_pred[(((b * num_anchors + k) * 4 + j) * height + h) * width +
                      w];
      }
      struct bbox pred;
      if (iou_loss) {
        pred = reg_iou(x1, y1, x2, y2, delta[0], delta[1], delta[2], delta[3]);
      } else {
        pred = reg_bbox(x1, y1, x2, y2, delta[0], delta[1], delta[2], delta[3]);
      }
      pred.x1 = MAX(MIN(pred.x1, im_width - 1.0), 0.0);
      pred.y1 = MAX(MIN(pred.y1, im_height - 1.0), 0.0);
      pred.x2 = MAX(MIN(pred.x2, im_width - 1.0), 0.0);
      pred.y2 = MAX(MIN(pred.y2, im_height - 1.0), 0.0);

      int real_height = im_height / feature_stride;
      int real_width = im_width / feature_stride;

      float bbox_w = pred.x2 - pred.x1 + 1.0;
      float bbox_h = pred.y2 - pred.y1 + 1.0;
      int min_size = im_info[b * 3 + 2] * rpn_min_size;

      float pred_score = cls_prob[(
          int)(((b * num_anchors * 2 + num_anchors + k) * height + h) * width +
               w)];
      if ((h >= real_height) || (w >= real_width)) {
        pred_score = -1;
      }
      output[out_index * 5 + 0] = pred.x1;
      output[out_index * 5 + 1] = pred.y1;
      output[out_index * 5 + 2] = pred.x2;
      output[out_index * 5 + 3] = pred.y2;
      output[out_index * 5 + 4] = pred_score;
      if ((bbox_w < min_size) || (bbox_h < min_size)) {
        output[out_index * 5 + 0] = output[out_index * 5 + 0] - min_size / 2.0;
        output[out_index * 5 + 1] = output[out_index * 5 + 1] - min_size / 2.0;
        output[out_index * 5 + 2] = output[out_index * 5 + 2] + min_size / 2.0;
        output[out_index * 5 + 3] = output[out_index * 5 + 3] + min_size / 2.0;
        output[out_index * 5 + 4] = -1.0;
      }
    }
  }
  return output;
}

typedef struct {
  int index;
  float data;
} index_value;

static int argsort(const void *a, const void *b)
{
  return ((((index_value *)a)->data - ((index_value *)b)->data > 0) ? -1 : 1);
}

static float calculate_overlap(float *out_tensor, int box_a_idx, int box_b_idx)
{
  float w =
      MAX(0.0, MIN(out_tensor[box_a_idx + 2], out_tensor[box_b_idx + 2]) -
                   MAX(out_tensor[box_a_idx], out_tensor[box_b_idx]) + 1.0);
  float h = MAX(
      0.0, MIN(out_tensor[box_a_idx + 3], out_tensor[box_b_idx + 3]) -
               MAX(out_tensor[box_a_idx + 1], out_tensor[box_b_idx + 1]) + 1.0);
  float i = w * h;
  float u = (out_tensor[box_a_idx + 2] - out_tensor[box_a_idx] + 1.0) *
                (out_tensor[box_a_idx + 3] - out_tensor[box_a_idx + 1] + 1.0) +
            (out_tensor[box_b_idx + 2] - out_tensor[box_b_idx] + 1.0) *
                (out_tensor[box_b_idx + 3] - out_tensor[box_b_idx + 1] + 1.0) -
            i;
  return i / u;
}

static float *compute_nms(int batch, int num_bbox, float *sorted_bbox,
                          float threshold)
{
  float *out = malloc(batch * num_bbox * sizeof(float));
  for (int b = 0; b < batch; b++) {
    int base_idx = b * num_bbox;
    for (int i = 0; i < num_bbox; i++) {
      out[base_idx + i] = 0;
    }
    for (int l = 0; l < num_bbox - 1; l++) {
      for (int i = 0; i < num_bbox; i++) {
        if ((i < num_bbox) && (i > l)) {
          if (out[base_idx + l] == 0) {
            float iou = calculate_overlap(sorted_bbox, (base_idx + l) * 5,
                                          (base_idx + i) * 5);
            if (iou > threshold) {
              out[base_idx + i] = 1;
            }
          }
        }
      }
    }
  }
  return out;
}

static float *prepare_output(float *sorted_bbox, float *remove_mask, int batch,
                             int num_bbox, int rpn_post_nms_top_n)
{
  int *i = malloc(batch * sizeof(int));
  int *nkeep = malloc(batch * sizeof(int));
  float *output = malloc(batch * rpn_post_nms_top_n * 5 * sizeof(int));

  for (int b = 0; b < batch; b++) {
    nkeep[b] = 0;
    i[b] = 0;
  }
  for (int j = 0; j < num_bbox; j++) {
    for (int b = 0; b < batch; b++) {
      if (remove_mask[b * num_bbox + j] == 0) {
        nkeep[b] = nkeep[b] + 1;
      }
    }
  }
  for (int b = 0; b < batch; b++) {
    if (nkeep[b] > 0) {
      int ceil_idx = ceil((float)rpn_post_nms_top_n / nkeep[b]);
      for (int m = 0; m < ceil_idx; m++) {
        for (int j = 0; j < num_bbox; j++) {
          int offset_j = (b * num_bbox + j) * 5;
          int offset_i = (b * rpn_post_nms_top_n + i[b]) * 5;
          if ((i[b] < rpn_post_nms_top_n) &&
              (remove_mask[(b * num_bbox + j)] == 0)) {
            output[offset_i] = b;
            for (int k = 0; k < 4; k++) {
              output[offset_i + k + 1] = sorted_bbox[offset_j + k];
            }
            i[b] = i[b] + 1;
          }
        }
      }
    }
  }
  return output;
}

int csi_proposal_f32(struct csi_tensor *cls_prob,
                     struct csi_tensor *bbox_pred,
                     struct csi_tensor *im_info,
                     struct csi_tensor *output,
                     struct proposal_params *params)
{
  float *output_data = output->data;

  int num_anchors = params->scales_num * params->ratios_num;

  int batch = cls_prob->dim[0];
  int height = cls_prob->dim[2];
  int width = cls_prob->dim[3];

  int num_bbox = height * width * num_anchors;
  params->rpn_pre_nms_top_n =
      params->rpn_pre_nms_top_n > 0 ? MIN(params->rpn_pre_nms_top_n, num_bbox) : num_bbox;

  float *bbox =
      predict_bbox(cls_prob, bbox_pred, im_info, params->ratios, params->ratios_num, params->scales,
                   params->scales_num, params->feature_stride, params->iou_loss, params->rpn_min_size);
  index_value *score = malloc(batch * num_bbox * sizeof(index_value));
  for (int i = 0; i < batch; i++) {
    for (int j = 0; j < num_bbox; j++) {
      int id = j + i * num_bbox;
      int lid = j * 5 + 4 + i * num_bbox;
      score[id].index = id;
      score[id].data = bbox[lid];
    }
  }

  qsort(score, batch * num_bbox, sizeof(index_value), argsort);

  float *sorted_bbox = malloc(batch * params->rpn_pre_nms_top_n * 5 * sizeof(float));
  for (int b = 0; b < batch; b++) {
    for (int i = 0; i < params->rpn_pre_nms_top_n; i++) {
      int sorted_index = score[b * params->rpn_pre_nms_top_n + i].index;
      for (int j = 0; j < 5; j++) {
        int bbox_index = b * params->rpn_pre_nms_top_n + sorted_index * 5 + j;
        int id = b * params->rpn_pre_nms_top_n + i * 5 + j;
        sorted_bbox[id] = bbox[bbox_index];
      }
    }
  }

  float *nms_remove_mask =
      compute_nms(batch, params->rpn_pre_nms_top_n, sorted_bbox, params->threshold);
  float *nms_out = prepare_output(sorted_bbox, nms_remove_mask, batch,
                                  params->rpn_pre_nms_top_n, params->rpn_post_nms_top_n);

  for (int i = 0; i < batch * params->rpn_post_nms_top_n * 5; i++) {
    output_data[i] = nms_out[i];
  }

  return CSINN_TRUE;
}

int csi_proposal_u8(struct csi_tensor *cls_prob,
                    struct csi_tensor *bbox_pred,
                    struct csi_tensor *im_info,
                    struct csi_tensor *output,
                    struct proposal_params *params)
{

    float *scales = (float *)malloc(params->scales_num * sizeof(float));
    for(int i = 0; i < params->scales_num; i++){
      scales[i] = csi_get_scale(params->scale_multipliers[i],params->scale_shifts[i]);
    }

    float *ratios = (float *)malloc(params->scales_num * sizeof(float));
    for(int i = 0; i < params->ratios_num; i++){
      ratios[i] = csi_get_scale(params->ratio_multipliers[i],params->ratio_shifts[i]);
    }
    float threshold = csi_get_scale(params->threshold_multiplier,params->threshold_shift);

    float *float_output_data;
    struct csi_tensor float_output;
    int64_t out_size = 1;
    for (int i = 0; i < output->dim_count; ++i) {
        out_size *= output->dim[i];
    }
    uint8_t *output_data = output->data;
    memcpy(&float_output, output, sizeof(struct csi_tensor));
    float_output_data = malloc(out_size * sizeof(float));
    float_output.data = float_output_data;
    float *f_cls_data;
    float *f_bbox_data;
    struct csi_tensor f_cls;
    struct csi_tensor f_bbox;
    uint8_t *cls_data = cls_prob->data;
    uint8_t *bbox_data = bbox_pred->data;

    int c_size = 1;
    for (int i = 0; i < cls_prob->dim_count; i++) {
        c_size *= cls_prob->dim[i];
    }

    int b_size = 1;
    for (int i = 0; i < bbox_pred->dim_count; i++) {
        b_size *= bbox_pred->dim[i];
    }

    memcpy(&f_cls, cls_prob, sizeof(struct csi_tensor));
    memcpy(&f_bbox, bbox_pred, sizeof(struct csi_tensor));
    f_cls_data = malloc(c_size * sizeof(float));
    f_bbox_data = malloc(b_size * sizeof(float));
    f_cls.data = f_cls_data;
    f_bbox.data = f_bbox_data;

    for (int i = 0; i < c_size; i++) {
        f_cls_data[i] = csi_dequantize_u8_to_f32(cls_data[i], cls_prob->zero_point,
                                                 cls_prob->multiplier, cls_prob->shift);
    }

    for (int i = 0; i < b_size; i++) {
        f_bbox_data[i] = csi_dequantize_u8_to_f32(bbox_data[i], bbox_pred->zero_point,
                                                 bbox_pred->multiplier, bbox_pred->shift);
    }

    params->ratios = ratios;
    params->scales = scales;
    params->threshold = threshold;
    csi_proposal_f32(&f_cls, &f_bbox, im_info, &float_output, params);

    for (int i = 0; i < out_size; i++) {
        output_data[i] = csi_quantize_f32_to_u8(float_output_data[i], output->zero_point,
                                          output->multiplier, output->shift);
    }
    free(float_output_data);
    free(f_cls_data);
    free(f_bbox_data);
    return CSINN_TRUE;
}

int csi_proposal_init(struct csi_tensor *cls_prob,
                      struct csi_tensor *bbox_pred,
                      struct csi_tensor *im_info,
                      struct csi_tensor *output,
                      struct proposal_params *params)
{
    params->bc = csi_bc_map(params->api, CSINN_OP_PROPOSAL, output->dtype);
    if (params->bc == NULL) {
        return CSINN_UNSUPPORT_DTYPE;
    }
    return CSINN_TRUE;
}

int csi_proposal(struct csi_tensor *cls_prob,
                 struct csi_tensor *bbox_pred,
                 struct csi_tensor *im_info,
                 struct csi_tensor *output,
                 struct proposal_params *params)
{
    if (params->bc != NULL) {
        params->bc(cls_prob, bbox_pred, im_info, output, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}
