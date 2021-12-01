/*
 * Copyright (C) 2016-2021 C-SKY Limited. All rights reserved.
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

#include "csi_ref.h"
#include "csi_utils.h"
#include <time.h>

int32_t csi_ref_max_internal_s32(int32_t a, int32_t b)
{
    if (a >= b) {
        return a;
    } else {
        return b;
    }
}

int32_t csi_ref_min_internal_s32(int32_t a, int32_t b)
{
    if (a <= b) {
        return a;
    } else {
        return b;
    }
}

int32_t csi_ref_get_index(int32_t *dim, int32_t index0, int32_t index1, int32_t index2, int32_t index3)
{
    return ((index0 * dim[1] + index1) * dim[2] + index2) * dim[3] + index3;
}

int32_t csi_ref_get_index_5(int32_t *dim, int32_t index0, int32_t index1, int32_t index2, int32_t index3, int32_t index4)
{
    return dim[4] * (dim[3] * (dim[2] * (dim[1] * index0 + index1) + index2) + index3) + index4;
}

/* iteration to calculate index */
int32_t csi_ref_get_index_iter(int32_t *dim, int dim_idx, int32_t *index)
{
    int32_t ret;
    if (dim_idx > 0) {
        ret = csi_ref_get_index_iter(dim, dim_idx - 1, index) * dim[dim_idx] + index[dim_idx];
    } else {
        ret = index[dim_idx];
    }

    return ret;
}

int32_t csi_ref_get_broadcast_index_iter(int32_t *dim, int dim_idx, int32_t *index)
{
    int32_t ret;
    if (dim_idx > 0) {
        if (dim[dim_idx] != 1) {
            ret = csi_ref_get_broadcast_index_iter(dim, dim_idx - 1, index) * dim[dim_idx] + index[dim_idx];
        } else {
            ret = csi_ref_get_broadcast_index_iter(dim, dim_idx - 1, index);
        }
    } else {
        if (dim[dim_idx] != 1) {
            ret = index[dim_idx];
        } else {
            ret = 0;
        }
    }

    return ret;
}

int32_t *csi_ref_get_input_dim(struct csi_tensor *input, int dim_count, int32_t *axis, int axis_size)
{
    int8_t alloc_size = dim_count * sizeof(int32_t *);
    int32_t *ret = malloc(alloc_size);

    for (int i = 0; i < dim_count; i++) {
        ret[i] = 1;
    }

    for (int i = 0; i < axis_size; i++) {
        ret[axis[i]] = input->dim[axis[i]];
    }

    return ret;
}

void csi_ref_diso_dim_iter(int32_t *dim, int dim_idx, int32_t *index, struct csi_ref_diso_callback *cb)
{
    for (index[dim_idx] = 0; index[dim_idx] < dim[dim_idx]; index[dim_idx]++) {
        if (dim_idx == 0) {
            int input1_idx = csi_ref_get_broadcast_index_iter(cb->input_dim, cb->output->dim_count - 1, index);
            int output_idx = csi_ref_get_index_iter(cb->output->dim, cb->output->dim_count - 1, index);
            cb->bc(cb->input0->data, cb->input1->data, cb->output->data, input1_idx, output_idx);
        } else {
            csi_ref_diso_dim_iter(dim, dim_idx - 1, index, cb);
        }
    }
}

int csi_check_rhs_shape(struct csi_tensor *input)
{
    int axis = -1;
    int in_size = csi_tensor_size(input);
    for (int i = 0; i < input->dim_count; i++)
    {
        if (input->dim[i] == in_size){axis = i;}
    }
    return axis;
}

int csi_ref_diso_broadcast_base(struct csi_tensor *input0,
                                struct csi_tensor *input1,
                                struct csi_tensor *output,
                                struct diso_params *params,
                                struct csi_ref_diso_callback *cb)
{
    float *input0_data = input0->data;
    float *input1_data = input1->data;
    float *output_data = output->data;

    cb->output = output;

    int32_t idx[output->dim_count];

    int size0 = csi_tensor_size(input0);
    int size1 = csi_tensor_size(input1);

    if (size0 == size1) {
        for (int i = 0; i < size0; i++) {
            cb->bc(input0_data, input1_data, output_data, i, i);
        }
    } else {
        if (size0 > size1) {
            cb->input0 = input0;
            cb->input1 = input1;
        } else {
            cb->input0 = input1;
            cb->input1 = input0;
        }
        /* FIXME: other axis */
        int axis_size = 1;
        int axis[axis_size];

        axis[0] = csi_check_rhs_shape(cb->input1);
        struct csi_tensor new_input1;
        if (axis[0] != -1){
            memcpy(&new_input1, cb->input1, sizeof(struct csi_tensor));
            new_input1.dim_count = 1;
            new_input1.dim[0] = cb->input1->dim[axis[0]];
            cb->input1 = &new_input1;
        }

        if (cb->input1->dim_count == 1) {

            for (int i = 0; i < output->dim_count; i++) {
                if (cb->input1->dim[0] == output->dim[i]){
                    axis[0] = i;
                }
            }
            cb->input_dim = csi_ref_get_input_dim(input1, output->dim_count, axis, axis_size);
        } else if (cb->input0->dim_count == cb->input1->dim_count) {
            cb->input_dim = cb->input1->dim;
        } else {
            for (int i = 0; i < cb->input1->dim_count; i++) {
                if (cb->input1->dim[cb->input1->dim_count - i - 1] !=
                    cb->input0->dim[cb->input0->dim_count - i - 1]) {
                    return CSINN_FALSE;
                }
            }
        }

        csi_ref_diso_dim_iter(output->dim, output->dim_count - 1, idx, cb);
    }

    return CSINN_TRUE;
}

float csi_ref_get_scale(int32_t multiplier, int32_t shift)
{
    float scale = multiplier / pow(2, 31) * pow(2, shift);

    return scale;
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

uint8_t csi_ref_quantize_channel_u8(int32_t data, struct csi_tensor* input, struct csi_tensor* output, float wscale)
{
    float out = data * input->qinfo->scale * wscale;
    return csi_ref_quantize_f32_to_u8(out, output->qinfo);
}

int8_t csi_ref_quantize_channel_i8(int32_t data, struct csi_tensor* input, struct csi_tensor* output, float wscale)
{
    float out = data * input->qinfo->scale * wscale;
    return csi_ref_quantize_f32_to_i8(out, output->qinfo);
}

float csi_ref_dequantize_u8_to_f32(uint8_t input, struct csi_quant_info *qinfo)
{
    float x = input;
    x -= qinfo->zero_point;
    float scale = csi_ref_get_scale(qinfo->multiplier, qinfo->shift);
    return x * scale;
}

float csi_ref_dequantize_i8_to_f32(int8_t input, struct csi_quant_info *qinfo)
{
    float x = input;
    x -= qinfo->zero_point;
    float scale = csi_ref_get_scale(qinfo->multiplier, qinfo->shift);
    return x * scale;
}

uint8_t csi_ref_quantize_f32_to_u8(float input, struct csi_quant_info *qinfo)
{
    float scale = csi_ref_get_scale(qinfo->multiplier, qinfo->shift);
    float output = round(input / scale + qinfo->zero_point);
    return fmin(255, fmax(0, output));
}

int8_t csi_ref_quantize_f32_to_i8(float input, struct csi_quant_info *qinfo)
{
    float scale = csi_ref_get_scale(qinfo->multiplier, qinfo->shift);
    float output = round(input / scale + qinfo->zero_point);
    return fmin(127, fmax(-127, output));
}

struct csi_tensor *csi_ref_deconv_kernel_nchw_to_nhwc_f32(struct csi_tensor *t, int32_t *permute)
{
    struct csi_tensor *nt = csi_alloc_tensor(NULL);

    assert(t->dim_count < 5);

    int size = csi_tensor_byte_size(t);

    for (int i = t->dim_count; i < 4; i++) {
        t->dim[i] = 1;
    }

    int t_dim = t->dim_count;
    t->dim_count = 4;
    t->quant_channel = 0;
    csi_tensor_copy(nt, t);
    nt->dim[0] = t->dim[permute[0]];
    nt->dim[1] = t->dim[permute[1]];
    nt->dim[2] = t->dim[permute[2]];
    nt->dim[3] = t->dim[permute[3]];

    nt->data = malloc(size);

    struct transpose_params tparams;
    tparams.permute = permute;
    tparams.base.api = CSINN_REF;
    tparams.base.name = "internal_transpose";
    csi_transpose_init(t, nt, &tparams);
    csi_transpose(t, nt, &tparams);
    t->dim_count = t_dim;
    return nt;
}

struct csi_tensor *csi_ref_nchw_to_nhwc_8(struct csi_tensor *t)
{
    struct csi_tensor *nt = csi_alloc_tensor(NULL);

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
    csi_tensor_copy(nt, t);
    nt->dim[1] = t->dim[2];
    nt->dim[2] = t->dim[3];
    nt->dim[3] = t->dim[1];

    nt->data = malloc(size);
    int32_t permute[4] = {0, 2, 3, 1};

    struct transpose_params tparams;
    tparams.permute = permute;
    tparams.base.api = CSINN_REF;
    tparams.base.name = "internal_transpose";
    csi_transpose_init(t, nt, &tparams);
    csi_transpose(t, nt, &tparams);
    t->dim_count = t_dim;
    return nt;
}

void csi_ref_nhwc_to_nchw_8(struct csi_tensor *nt, struct csi_tensor *t)
{
    nt->dim[1] = t->dim[3];
    nt->dim[2] = t->dim[1];
    nt->dim[3] = t->dim[2];

    int nt_dim = nt->dim_count;
    nt->dim_count = 4;

    int32_t permute[4] = {0, 3, 1, 2};

    struct transpose_params tparams;
    tparams.permute = permute;
    tparams.base.api = CSINN_REF;
    tparams.base.name = "internal_transpose";
    csi_transpose_init(t, nt, &tparams);
    csi_transpose(t, nt, &tparams);

    nt->dim_count = nt_dim;

    free(t->data);
    free(t);
}

struct csi_tensor *csi_ref_nchw_to_nhwc_f32(struct csi_tensor *t)
{
    struct csi_tensor *nt = csi_alloc_tensor(NULL);

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
    t->quant_channel = 0;
    csi_tensor_copy(nt, t);
    nt->dim[1] = t->dim[2];
    nt->dim[2] = t->dim[3];
    nt->dim[3] = t->dim[1];

    nt->data = malloc(size * 4);
    int32_t permute[4] = {0, 2, 3, 1};

    struct transpose_params tparams;
    tparams.permute = permute;
    tparams.base.api = CSINN_REF;
    tparams.base.name = "internal_transpose";
    csi_transpose_init(t, nt, &tparams);
    csi_transpose(t, nt, &tparams);
    t->dim_count = t_dim;
    return nt;
}

void csi_ref_nhwc_to_nchw_f32(struct csi_tensor *nt, struct csi_tensor *t)
{
    nt->dim[1] = t->dim[3];
    nt->dim[2] = t->dim[1];
    nt->dim[3] = t->dim[2];

    int nt_dim = nt->dim_count;
    nt->dim_count = 4;

    int32_t permute[4] = {0, 3, 1, 2};

    struct transpose_params tparams;
    tparams.permute = permute;
    tparams.base.api = CSINN_REF;
    tparams.base.name = "internal_transpose";
    csi_transpose_init(t, nt, &tparams);
    csi_transpose(t, nt, &tparams);

    nt->dim_count = nt_dim;

    free(t->data);
    free(t);
}

int32_t csi_ref_get_reduction_index(int32_t k, const int32_t *strides,
                            const int32_t *extents, int32_t n)
{
    int32_t index = 0;
    for (int32_t i = 0; i < n; i++)
    {
        int32_t div = 1;
        for (int32_t j = i + 1; j < n; j++)
        {
            div *= extents[j];
        }
        int32_t mod = div * extents[i];

        index += ((k % mod) / div * strides[i]);
    }

    return index;
}

float csi_ref_uint8_to_float(uint8_t i, struct csi_tensor *t)
{
    return ((float)i - t->qinfo->zero_point) * t->qinfo->scale;
}

float csi_ref_int8_to_float(int8_t i, struct csi_tensor *t)
{
    return ((float)i - t->qinfo->zero_point) * t->qinfo->scale;
}

struct csi_tensor *csi_ref_alloc_float_tensor(struct csi_tensor *src)
{
    struct csi_tensor *ret = csi_alloc_tensor(NULL);
    csi_tensor_copy(ret, src);
    ret->dtype = CSINN_DTYPE_FLOAT32;
    int size = csi_tensor_byte_size(ret);
    float *data = malloc(size);
    ret->data = data;
    return ret;
}

void csi_ref_free_float_tensor(struct csi_tensor *src)
{
    free(src->data);
    csi_free_tensor(src);
}

struct csi_tensor *csi_ref_convert_float_tensor(struct csi_tensor *src)
{
    struct csi_tensor *ret = csi_ref_alloc_float_tensor(src);
    int size = csi_tensor_size(src);
    float *float_data = ret->data;

    if (src->dtype == CSINN_DTYPE_UINT8) {
        uint8_t *input_data = src->data;
        for (int i = 0; i < size; i++) {
            float_data[i] = csi_ref_uint8_to_float(input_data[i], src);
        }
    } else if (src->dtype == CSINN_DTYPE_INT8) {
        int8_t *input_data = src->data;
        for (int i = 0; i < size; i++) {
            float_data[i] = csi_ref_int8_to_float(input_data[i], src);
        }
    } else {
        return NULL;
    }

    return ret;
}

void csi_ref_conv_free_float_tensor(struct csi_tensor *input,
    struct csi_tensor *output, struct csi_tensor *kernel,
    struct csi_tensor *bias)
{
    csi_ref_free_float_tensor(input);
    csi_ref_free_float_tensor(output);
    csi_ref_free_float_tensor(kernel);
    csi_ref_free_float_tensor(bias);
}

struct csi_tensor *csi_ref_tensor_transform_f32(struct csi_tensor *input)
{
    struct csi_tensor *ret = csi_alloc_tensor(NULL);
    csi_tensor_copy(ret, input);
    ret->dtype = CSINN_DTYPE_FLOAT32;
    ret->data = malloc(csi_tensor_size(input) * 4);
    if (csi_tensor_data_convert(ret, input) == CSINN_TRUE) {
        return ret;
    }
    return NULL;
}

int csi_ref_tensor_transform_free_f32(struct csi_tensor *input)
{
    free(input->data);
    csi_free_tensor(input);
    return CSINN_TRUE;
}

int csi_ref_siso_callback_base(struct csi_tensor *input,
                               struct csi_tensor *output,
                               void *params,
                               void *cb)
{
    int (*callback)() = cb;
    int ret;
    struct csi_tensor *finput = csi_ref_tensor_transform_f32(input);
    struct csi_tensor *foutput = csi_ref_tensor_transform_f32(output);
    ret = callback(finput, foutput, params);
    csi_tensor_data_convert(output, foutput);
    csi_ref_tensor_transform_free_f32(finput);
    csi_ref_tensor_transform_free_f32(foutput);
    return ret;
}

int csi_ref_diso_callback_base(struct csi_tensor *input0,
                               struct csi_tensor *input1,
                               struct csi_tensor *output,
                               void *params,
                               void *cb)
{
    int (*callback)() = cb;
    int ret;
    struct csi_tensor *finput0 = csi_ref_tensor_transform_f32(input0);
    struct csi_tensor *finput1 = csi_ref_tensor_transform_f32(input1);
    struct csi_tensor *foutput = csi_ref_tensor_transform_f32(output);
    ret = callback(finput0, finput1, foutput, params);
    csi_tensor_data_convert(output, foutput);
    csi_ref_tensor_transform_free_f32(finput0);
    csi_ref_tensor_transform_free_f32(finput1);
    csi_ref_tensor_transform_free_f32(foutput);
    return ret;
}

int csi_ref_conv_callback_base(struct csi_tensor *input,
                               struct csi_tensor *output,
                               struct csi_tensor *kernel,
                               struct csi_tensor *bias,
                               void *params,
                               void *cb)
{
    int (*callback)() = cb;
    struct csi_tensor *float_input = csi_ref_tensor_transform_f32(input);
    struct csi_tensor *float_kernel = csi_ref_tensor_transform_f32(kernel);
    struct csi_tensor *float_bias = csi_ref_tensor_transform_f32(bias);
    struct csi_tensor *float_output = csi_ref_tensor_transform_f32(output);
    float *float_bias_data = float_bias->data;
    int32_t *bias_data = bias->data;
    int bias_size = csi_tensor_size(bias);
    for (int i = 0; i < bias_size; i++) {
        float_bias_data[i] = bias_data[i] * kernel->qinfo->scale * input->qinfo->scale;
    }
    int ret = callback(float_input, float_output, float_kernel, float_bias, params);
    csi_tensor_data_convert(output, float_output);
    csi_ref_tensor_transform_free_f32(float_input);
    csi_ref_tensor_transform_free_f32(float_output);
    csi_ref_tensor_transform_free_f32(float_kernel);
    csi_ref_tensor_transform_free_f32(float_bias);
    return ret;
}
