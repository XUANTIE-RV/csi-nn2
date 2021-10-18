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
#include <assert.h>
#include <math.h>

static float bilinear_sample_nchw(const float *input, int32_t batch,
                                  int32_t channel, int32_t height,
                                  int32_t width, int32_t i, int32_t c, float y,
                                  float x, const int32_t max_y,
                                  const int32_t max_x) {
  float in_y = y;
  float yf = floor(in_y);
  int32_t yc = ceil(in_y);

  int32_t y0 = floor(in_y);
  int32_t y1 = yc > max_y ? max_y : yc;
  float y_lerp = in_y - yf;

  float in_x = x;
  float xf = floor(in_x);
  int32_t xc = ceil(in_x);

  int32_t x0 = floor(in_x);
  int32_t x1 = xc > max_x ? max_x : xc;
  float x_lerp = in_x - xf;

  int32_t s1 = channel * height * width;
  int32_t s2 = height * width;
  int32_t s3 = width;

  float A = input[i * s1 + c * s2 + y0 * s3 + x0];
  float B = input[i * s1 + c * s2 + y0 * s3 + x1];
  float C = input[i * s1 + c * s2 + y1 * s3 + x0];
  float D = input[i * s1 + c * s2 + y1 * s3 + x1];

  return A * (1 - x_lerp) * (1 - y_lerp) + B * x_lerp * (1 - y_lerp) +
         C * (1 - x_lerp) * y_lerp + D * x_lerp * y_lerp;
}

static float _bilinear(const float *data, int32_t batch, int32_t channel,
                       int32_t height, int32_t width, int32_t i, int32_t c,
                       float y, float x) {
  bool outside = (y < -1.0 || x < -1.0 || y > height || x > width);

  y = fmax(y, 0.0);
  x = fmax(x, 0.0);

  float val = bilinear_sample_nchw(data, batch, channel, height, width, i, c, y,
                                   x, height - 1, width - 1);

  if (outside)
    return 0.0;
  else
    return val;
}

int csi_roi_align_f32(struct csi_tensor *data,
                       struct csi_tensor *rois,
                       struct csi_tensor *output,
                       struct roi_align_params *params)
{

  assert(0);

  float *output_data = output->data;

  int32_t i_size = output->dim[0];
  int32_t c_size = output->dim[1];
  int32_t ph_size = output->dim[2];
  int32_t pw_size = output->dim[3];

  int32_t s1 = c_size * ph_size * pw_size;
  int32_t s2 = ph_size * pw_size;
  int32_t s3 = pw_size;


  for (int32_t i = 0; i < i_size; i++) {
    for (int32_t c = 0; c < c_size; c++) {
      for (int32_t ph = 0; ph < ph_size; ph++) {
        for (int32_t pw = 0; pw < pw_size; pw++) {
          float *roi = rois->data;
          int32_t *roi_int = rois->data;
          int32_t batch_index = roi_int[i * 5 + 0];
          float roi_start_w = roi[i * 5 + 1];
          float roi_start_h = roi[i * 5 + 2];
          float roi_end_w = roi[i * 5 + 3];
          float roi_end_h = roi[i * 5 + 4];

          roi_start_h *= params->spatial_scale;
          roi_end_h *= params->spatial_scale;
          roi_start_w *= params->spatial_scale;
          roi_end_w *= params->spatial_scale;

          // force malformed ROIs to be 1x1
          float roi_h = fmax(roi_end_h - roi_start_h, 1.0);
          float roi_w = fmax(roi_end_w - roi_start_w, 1.0);

          float bin_h = roi_h / params->pooled_size_h;
          float bin_w = roi_w / params->pooled_size_w;

          int32_t roi_bin_grid_h = 0;
          int32_t roi_bin_grid_w = 0;

          if (params->sample_ratio > 0) {
            roi_bin_grid_h = roi_bin_grid_w = params->sample_ratio;
          } else {
            roi_bin_grid_h = ceil(roi_h / params->pooled_size_h);
            roi_bin_grid_w = ceil(roi_w / params->pooled_size_w);
          }

          int32_t count = roi_bin_grid_h * roi_bin_grid_w;
          int32_t rh_size = roi_bin_grid_h;
          int32_t rw_size = roi_bin_grid_w;
          float result = 0;
          for (int32_t rh = 0; rh < rh_size; rh++) {
            for (int32_t rw = 0; rw < rw_size; rw++) {
              roi_start_h += ph * bin_h;
              roi_start_w += pw * bin_w;
              float _bv =
                  _bilinear(data->data, data->dim[0], data->dim[1],
                            data->dim[2], data->dim[3], batch_index, c,
                            roi_start_h + (rh + 0.5) * bin_h / roi_bin_grid_h,
                            roi_start_w + (rw + 0.5) * bin_w / roi_bin_grid_w);
              result += _bv / count;
            }
          }

          output_data[i * s1 + c * s2 + ph * s3 + pw] = result;
        }
      }
    }
  }
  return CSINN_TRUE;
}

int csi_roi_align_init(struct csi_tensor *data,
                       struct csi_tensor *rois,
                       struct csi_tensor *output,
                       struct roi_align_params *params)
{
    params->bc = csi_bc_map(params->api, CSINN_OP_ROIALIGN, data->dtype);
    if (params->bc == NULL) {
        return CSINN_UNSUPPORT_DTYPE;
    }
    return CSINN_TRUE;
}

int csi_roi_align(struct csi_tensor *data,
                  struct csi_tensor *rois,
                  struct csi_tensor *output,
                  struct roi_align_params *params)
{
    if (params->bc != NULL) {
        params->bc(data, rois, output, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}
