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

int32_t csi_max_internal_s32(int32_t a, int32_t b)
{
    if (a >= b) {
        return a;
    } else {
        return b;
    }
}

int32_t csi_min_internal_s32(int32_t a, int32_t b)
{
    if (a <= b) {
        return a;
    } else {
        return b;
    }
}

uint8_t csi_saturate_u8(int32_t input)
{
    csi_min_internal_s32(255, csi_max_internal_s32(0, input));
}

int32_t csi_get_index(int32_t *dim, int32_t index0, int32_t index1, int32_t index2, int32_t index3)
{
    return ((index0 * dim[1] + index1) * dim[2] + index2) * dim[3] + index3;
}

int32_t csi_get_index_5(int32_t *dim, int32_t index0, int32_t index1, int32_t index2, int32_t index3, int32_t index4)
{
    return dim[4] * (dim[3] * (dim[2] * (dim[1] * index0 + index1) + index2) + index3) + index4;
}

int32_t csi_get_index_6(int32_t *dim, int32_t index0, int32_t index1, int32_t index2, int32_t index3, int32_t index4, int32_t index5)
{
    return dim[5] * (dim[4] * (dim[3] * (dim[2] * (dim[1] * index0 + index1) + index2) + index3) + index4) + index5;
}

static int32_t mask_non_zero(int32_t a)
{
    int32_t zero = 0;
    return a ? (~zero) : zero;
}

static int32_t round_div_pot(int32_t x, int32_t exponent)
{
    assert(exponent >= 0);
    assert(exponent <= 31);
    int32_t mask = (1ll << exponent) - 1;
    int32_t zero = 0;
    int32_t one = 1;
    int32_t remainder = x & mask;
    int32_t threshold = (mask >> 1) + (mask_non_zero(x < zero) & one);
    return (x >> exponent) + (mask_non_zero(remainder > threshold) & one);
}

static int32_t high_mul_sat_round_double(int32_t a, int32_t b)
{
    int overflow = a == b && a == INT32_MIN;
    int64_t a_64 = a;
    int64_t b_64 = b;
    int64_t ab_64 = a_64 * b_64;
    int32_t nudge = ab_64 >= 0 ? (1 << 30) : (1 - (1 << 30));
    int32_t ab_x2_high32 = (int32_t)((ab_64 + nudge) / (1ll << 31));
    return overflow ? INT32_MAX : ab_x2_high32;
}

int32_t quantized_multiply(int32_t x, int32_t quantized_multiplier, int shift)
{
    int left_shift = shift > 0 ? shift : 0;
    int right_shift = shift > 0 ? 0 : -shift;
    return round_div_pot(high_mul_sat_round_double(x * (1 << left_shift), quantized_multiplier), right_shift);
}

int32_t quantized_multiply_s(int32_t x, int32_t quantized_multiplier, int left_shift)
{
    return round_div_pot(high_mul_sat_round_double(x, quantized_multiplier), -left_shift);
}

float csi_get_scale(int32_t multiplier, int32_t shift)
{
    float scale = multiplier / pow(2, 31) * pow(2, shift);

    return scale;
}

int32_t csi_dequantize_u8(uint8_t input, int32_t offset, int32_t multiplier, int32_t shift)
{
    int32_t x = input + offset;
    float scale = csi_get_scale(multiplier, shift);
    return x * scale;
}

uint8_t csi_quantize_u8(int32_t input, int32_t offset, int32_t multiplier, int32_t shift)
{
    int left_shift = shift > 0 ? shift : 0;
    int right_shift = shift > 0 ? 0 : -shift;
    int32_t output = round_div_pot(high_mul_sat_round_double(input * (1 << left_shift), multiplier), right_shift);
    output += offset;
    return csi_min_internal_s32(255, csi_max_internal_s32(0, output));
}

float csi_dequantize_f32(uint8_t input, int32_t offset, int32_t multiplier, int32_t shift)
{
    float x = input;
    x += offset;
    float scale = csi_get_scale(multiplier, shift);
    return x * scale;
}

uint8_t csi_quantize_f32(float input, int32_t offset, int32_t multiplier, int32_t shift)
{
    float scale = csi_get_scale(multiplier, shift);
    float output = round(input / scale + offset);
    return fmin(255, fmax(0, output));
}


uint8_t csi_requantize_u8(uint8_t input, int32_t input_offset, int32_t input_multiplier,
                          int32_t input_shift, int32_t output_offset, int32_t output_multiplier,
                          int32_t output_shift)
{
    float val = csi_dequantize_f32(input, input_offset, input_multiplier, input_shift);
    return csi_quantize_f32(val, output_offset, output_multiplier, output_shift);
}

struct csi_tensor *csi_deconv_kernel_nchw_to_nhwc_u8(struct csi_tensor *t, int32_t *permute)
{
    struct csi_tensor *nt = malloc(sizeof(struct csi_tensor));

    assert(t->dim_count < 5);

    int size = 1;
    for (int i = 0; i < t->dim_count; i++) {
        size = size * t->dim[i];
    }

    for (int i = t->dim_count; i < 4; i++) {
        t->dim[i] = 1;
    }

    int t_dim = t->dim_count;
    t->dim_count = 4;
    memcpy(nt, t, sizeof(struct csi_tensor));
    nt->dim[0] = t->dim[permute[0]];
    nt->dim[1] = t->dim[permute[1]];
    nt->dim[2] = t->dim[permute[2]];
    nt->dim[3] = t->dim[permute[3]];

    nt->data = malloc(size);

    struct transpose_params tparams;
    tparams.permute = permute;
    csi_transpose_init(t, nt, &tparams);
    csi_transpose(t, nt, &tparams);
    t->dim_count = t_dim;
    return nt;
}

struct csi_tensor *csi_nchw_to_nhwc_u8(struct csi_tensor *t)
{
    struct csi_tensor *nt = malloc(sizeof(struct csi_tensor));

    assert(t->dim_count < 5);

    int size = 1;
    for (int i = 0; i < t->dim_count; i++) {
        size = size * t->dim[i];
    }

    for (int i = t->dim_count; i < 4; i++) {
        t->dim[i] = 1;
    }

    int t_dim = t->dim_count;
    t->dim_count = 4;
    memcpy(nt, t, sizeof(struct csi_tensor));
    nt->dim[1] = t->dim[2];
    nt->dim[2] = t->dim[3];
    nt->dim[3] = t->dim[1];

    nt->data = malloc(size);
    int32_t permute[4] = {0, 2, 3, 1};

    struct transpose_params tparams;
    tparams.permute = permute;
    csi_transpose_init(t, nt, &tparams);
    csi_transpose(t, nt, &tparams);
    t->dim_count = t_dim;
    return nt;
}

void csi_nhwc_to_nchw_u8(struct csi_tensor *nt, struct csi_tensor *t)
{
    nt->dim[1] = t->dim[3];
    nt->dim[2] = t->dim[1];
    nt->dim[3] = t->dim[2];

    int nt_dim = nt->dim_count;
    nt->dim_count = 4;

    int32_t permute[4] = {0, 3, 1, 2};

    struct transpose_params tparams;
    tparams.permute = permute;
    csi_transpose_init(t, nt, &tparams);
    csi_transpose(t, nt, &tparams);

    nt->dim_count = nt_dim;

    free(t->data);
    free(t);
}

struct csi_tensor *csi_nchw_to_nhwc_f32(struct csi_tensor *t)
{
    struct csi_tensor *nt = malloc(sizeof(struct csi_tensor));

    assert(t->dim_count < 5);

    int size = 1;
    for (int i = 0; i < t->dim_count; i++) {
        size = size * t->dim[i];
    }

    for (int i = t->dim_count; i < 4; i++) {
        t->dim[i] = 1;
    }

    int t_dim = t->dim_count;
    t->dim_count = 4;
    memcpy(nt, t, sizeof(struct csi_tensor));
    nt->dim[1] = t->dim[2];
    nt->dim[2] = t->dim[3];
    nt->dim[3] = t->dim[1];

    nt->data = malloc(size * 4);
    int32_t permute[4] = {0, 2, 3, 1};

    struct transpose_params tparams;
    tparams.permute = permute;
    csi_transpose_init(t, nt, &tparams);
    csi_transpose(t, nt, &tparams);
    t->dim_count = t_dim;
    return nt;
}

void csi_nhwc_to_nchw_f32(struct csi_tensor *nt, struct csi_tensor *t)
{
    nt->dim[1] = t->dim[3];
    nt->dim[2] = t->dim[1];
    nt->dim[3] = t->dim[2];

    int nt_dim = nt->dim_count;
    nt->dim_count = 4;

    int32_t permute[4] = {0, 3, 1, 2};

    struct transpose_params tparams;
    tparams.permute = permute;
    csi_transpose_init(t, nt, &tparams);
    csi_transpose(t, nt, &tparams);

    nt->dim_count = nt_dim;

    free(t->data);
    free(t);
}

int32_t get_reduction_index(int32_t k, const int32_t *strides,
                         const int32_t *extents, int32_t n) {

  int32_t index = 0;
  for (int32_t i = 0; i < n; i++) {
    int32_t div = 1;
    for (int32_t j = i + 1; j < n; j++) {
      div *= extents[j];
    }
    int32_t mod = div * extents[i];

    index += ((k % mod) / div * strides[i]);
  }

  return index;
}