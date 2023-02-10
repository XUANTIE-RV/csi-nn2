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

#include <time.h>

#include "shl_ref.h"

int32_t shl_ref_max_internal_s32(int32_t a, int32_t b)
{
    if (a >= b) {
        return a;
    } else {
        return b;
    }
}

int32_t shl_ref_min_internal_s32(int32_t a, int32_t b)
{
    if (a <= b) {
        return a;
    } else {
        return b;
    }
}

int32_t shl_ref_get_index(int32_t *dim, int32_t index0, int32_t index1, int32_t index2,
                          int32_t index3)
{
    return ((index0 * dim[1] + index1) * dim[2] + index2) * dim[3] + index3;
}

int32_t shl_ref_get_index_5(int32_t *dim, int32_t index0, int32_t index1, int32_t index2,
                            int32_t index3, int32_t index4)
{
    return dim[4] * (dim[3] * (dim[2] * (dim[1] * index0 + index1) + index2) + index3) + index4;
}

/* iteration to calculate index */
int32_t shl_ref_get_index_iter(int32_t *dim, int dim_idx, int32_t *index)
{
    int32_t ret;
    if (dim_idx > 0) {
        ret = shl_ref_get_index_iter(dim, dim_idx - 1, index) * dim[dim_idx] + index[dim_idx];
    } else {
        ret = index[dim_idx];
    }

    return ret;
}

int32_t *shl_ref_get_input_dim(struct csinn_tensor *input, int dim_count, int32_t *axis,
                               int axis_size)
{
    int8_t alloc_size = dim_count * sizeof(int32_t *);
    int32_t *ret = shl_mem_alloc(alloc_size);

    for (int i = 0; i < dim_count; i++) {
        ret[i] = 1;
    }

    for (int i = 0; i < axis_size; i++) {
        ret[axis[i]] = input->dim[axis[i]];
    }

    return ret;
}

int shl_ref_diso_broadcast_base(struct csinn_tensor *input0, struct csinn_tensor *input1,
                                struct csinn_tensor *output, struct csinn_diso_params *params,
                                struct shl_ref_diso_callback *cb)
{
    float *input0_data = input0->data;
    float *input1_data = input1->data;
    float *output_data = output->data;

    cb->output = output;

    int out_size = csinn_tensor_size(output);
    float *in0_data_b = shl_mem_alloc(out_size * sizeof(float));
    float *in1_data_b = shl_mem_alloc(out_size * sizeof(float));

    struct csinn_tensor *b_input0 = csinn_alloc_tensor(NULL);
    struct csinn_tensor *b_input1 = csinn_alloc_tensor(NULL);
    csinn_tensor_copy(b_input0, output);
    csinn_tensor_copy(b_input1, output);
    b_input0->data = in0_data_b;
    b_input1->data = in1_data_b;

    if (shl_ref_broadcast_to_shape(input0, b_input0, output->dim, output->dim_count) ==
        CSINN_FALSE) {
        SHL_DEBUG_CALL(shl_debug_info("%s: broadcast input0 failed.\n", __func__));
        return CSINN_FALSE;
    };
    if (shl_ref_broadcast_to_shape(input1, b_input1, output->dim, output->dim_count) ==
        CSINN_FALSE) {
        SHL_DEBUG_CALL(shl_debug_info("%s: broadcast input1 failed.\n", __func__));
        return CSINN_FALSE;
    };

    int size0 = csinn_tensor_size(b_input0);
    int size1 = csinn_tensor_size(b_input1);

    if (size0 == size1) {
        for (int i = 0; i < size0; i++) {
            cb->bc(in0_data_b, in1_data_b, output_data, i, i);
        }
    } else {
        return CSINN_FALSE;
    }
    shl_mem_free(in0_data_b);
    shl_mem_free(in1_data_b);
    csinn_free_tensor(b_input0);
    csinn_free_tensor(b_input1);
    return CSINN_TRUE;
}

float shl_ref_get_scale(int32_t multiplier, int32_t shift)
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

uint8_t shl_ref_quantize_channel_u8(int32_t data, struct csinn_tensor *input,
                                    struct csinn_tensor *output, float wscale)
{
    float out = data * input->qinfo->scale * wscale;
    return shl_ref_quantize_f32_to_u8(out, output->qinfo);
}

int8_t shl_ref_quantize_channel_i8(int32_t data, struct csinn_tensor *input,
                                   struct csinn_tensor *output, float wscale)
{
    float out = data * input->qinfo->scale * wscale;
    return shl_ref_quantize_f32_to_i8(out, output->qinfo);
}

float shl_ref_dequantize_u8_to_f32(uint8_t input, struct csinn_quant_info *qinfo)
{
    float x = input;
    x -= qinfo->zero_point;
    float scale = shl_ref_get_scale(qinfo->multiplier, qinfo->shift);
    return x * scale;
}

float shl_ref_dequantize_i8_to_f32(int8_t input, struct csinn_quant_info *qinfo)
{
    float x = input;
    x -= qinfo->zero_point;
    float scale = shl_ref_get_scale(qinfo->multiplier, qinfo->shift);
    return x * scale;
}

uint8_t shl_ref_quantize_f32_to_u8(float input, struct csinn_quant_info *qinfo)
{
    float scale = shl_ref_get_scale(qinfo->multiplier, qinfo->shift);
    float output = nearbyint(input / scale + qinfo->zero_point);
    return fmin(255, fmax(0, output));
}

int8_t shl_ref_quantize_f32_to_i8(float input, struct csinn_quant_info *qinfo)
{
    float scale = shl_ref_get_scale(qinfo->multiplier, qinfo->shift);
    float output = nearbyint(input / scale + qinfo->zero_point);
    return fmin(127, fmax(-128, output));
}

struct csinn_tensor *shl_ref_deconv_kernel_nchw_to_nhwc_f32(struct csinn_tensor *t,
                                                            int32_t *permute)
{
    struct csinn_tensor *nt = csinn_alloc_tensor(NULL);

    assert(t->dim_count < 5);

    int size = csinn_tensor_byte_size(t);

    for (int i = t->dim_count; i < 4; i++) {
        t->dim[i] = 1;
    }

    int t_dim = t->dim_count;
    t->dim_count = 4;
    t->quant_channel = 0;
    csinn_tensor_copy(nt, t);
    nt->dim[0] = t->dim[permute[0]];
    nt->dim[1] = t->dim[permute[1]];
    nt->dim[2] = t->dim[permute[2]];
    nt->dim[3] = t->dim[permute[3]];

    nt->data = shl_mem_alloc(size);

    struct csinn_transpose_params tparams;
    tparams.permute = permute;
    tparams.base.api = CSINN_REF;
    tparams.base.name = "internal_transpose";
    shl_ref_transpose(t, nt, &tparams);
    t->dim_count = t_dim;
    return nt;
}

struct csinn_tensor *shl_ref_nchw_to_nhwc_8(struct csinn_tensor *t)
{
    struct csinn_tensor *nt = csinn_alloc_tensor(NULL);

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
    csinn_tensor_copy(nt, t);
    nt->dim[1] = t->dim[2];
    nt->dim[2] = t->dim[3];
    nt->dim[3] = t->dim[1];

    nt->data = shl_mem_alloc(size);
    int32_t permute[4] = {0, 2, 3, 1};

    struct csinn_transpose_params tparams;
    tparams.permute = permute;
    tparams.base.api = CSINN_REF;
    tparams.base.name = "internal_transpose";
    shl_ref_transpose(t, nt, &tparams);
    t->dim_count = t_dim;
    return nt;
}

void shl_ref_nhwc_to_nchw_8(struct csinn_tensor *nt, struct csinn_tensor *t)
{
    nt->dim[1] = t->dim[3];
    nt->dim[2] = t->dim[1];
    nt->dim[3] = t->dim[2];

    int nt_dim = nt->dim_count;
    nt->dim_count = 4;

    int32_t permute[4] = {0, 3, 1, 2};

    struct csinn_transpose_params tparams;
    tparams.permute = permute;
    tparams.base.api = CSINN_REF;
    tparams.base.name = "internal_transpose";
    shl_ref_transpose(t, nt, &tparams);

    nt->dim_count = nt_dim;

    shl_mem_free(t->data);
    shl_mem_free(t);
}

struct csinn_tensor *shl_ref_nchw_to_nhwc_f32(struct csinn_tensor *t)
{
    struct csinn_tensor *nt = csinn_alloc_tensor(NULL);

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
    csinn_tensor_copy(nt, t);
    nt->dim[1] = t->dim[2];
    nt->dim[2] = t->dim[3];
    nt->dim[3] = t->dim[1];

    nt->data = shl_mem_alloc(size * sizeof(float));
    int32_t permute[4] = {0, 2, 3, 1};

    struct csinn_transpose_params tparams;
    tparams.permute = permute;
    tparams.permute_num = 4;
    tparams.base.api = CSINN_REF;
    tparams.base.name = "internal_transpose";
    shl_ref_transpose(t, nt, &tparams);
    t->dim_count = t_dim;
    return nt;
}

void shl_ref_nhwc_to_nchw_f32(struct csinn_tensor *nt, struct csinn_tensor *t)
{
    nt->dim[1] = t->dim[3];
    nt->dim[2] = t->dim[1];
    nt->dim[3] = t->dim[2];

    int nt_dim = nt->dim_count;
    nt->dim_count = 4;

    int32_t permute[4] = {0, 3, 1, 2};

    struct csinn_transpose_params tparams;
    tparams.permute = permute;
    tparams.permute_num = 4;
    tparams.base.api = CSINN_REF;
    tparams.base.name = "internal_transpose";
    shl_ref_transpose(t, nt, &tparams);

    nt->dim_count = nt_dim;

    if (t->qinfo != NULL) {
        shl_mem_free(t->qinfo);
        t->qinfo = NULL;
    }
    shl_mem_free(t->data);
    shl_mem_free(t);
}

int32_t shl_ref_get_reduction_index(int32_t k, const int32_t *strides, const int32_t *extents,
                                    int32_t n)
{
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

float shl_ref_uint8_to_float(uint8_t i, struct csinn_tensor *t)
{
    return ((float)i - t->qinfo->zero_point) * t->qinfo->scale;
}

float shl_ref_int8_to_float(int8_t i, struct csinn_tensor *t)
{
    return ((float)i - t->qinfo->zero_point) * t->qinfo->scale;
}

int16_t shl_ref_float32_to_float16(float value)
{
    int16_t ret;
    if (value > -6.1e-5 && value < 6.1e-5) {
        /* to small for f16, ignore to 0 */
        return 0;
    }
    if (value > 65504) {
        shl_debug_error("too large f32 to f16\n");
        /* saturate to f16 max value: 65504 */
        value = 65504;
    }
    int32_t org_format = *(int32_t *)&value;
    int16_t sign = (org_format & 0x80000000) >> 16;
    int16_t frac = (org_format & 0x7fffff) >> 13;
    int16_t exp = (((((org_format >> 23) & 0xff) - 128) + 16) & 0x1f) << 10;
    ret = sign | frac | exp;
    return ret;
}

float shl_ref_float16_to_float32(int16_t value)
{
    float ret;
    if (value == 0) {
        return 0;
    }
    int32_t ret_format = 0;
    int32_t sign = (value & 0x8000) << 16;
    int32_t frac = (value & 0x3ff) << 13;
    int32_t exp = (((((value >> 10) & 0x1f) - 16) + 128) & 0xff) << 23;
    ret_format = sign | frac | exp;
    ret = *(float *)&ret_format;
    return ret;
}

int16_t shl_ref_float32_to_bfloat16(float value)
{
    int16_t ret;
    int32_t org_format = *(int32_t *)&value;
    ret = (org_format & 0xffff0000) >> 16;
    return ret;
}

float shl_ref_bfloat16_to_float32(int16_t value)
{
    float ret;
    int32_t ret_format = value << 16;
    ;
    ret = *(float *)&ret_format;
    return ret;
}

struct csinn_tensor *shl_ref_alloc_float_tensor(struct csinn_tensor *src)
{
    struct csinn_tensor *ret = csinn_alloc_tensor(NULL);
    csinn_tensor_copy(ret, src);
    ret->dtype = CSINN_DTYPE_FLOAT32;
    int size = csinn_tensor_byte_size(ret);
    float *data = shl_mem_alloc(size);
    ret->data = data;
    return ret;
}

void shl_ref_free_float_tensor(struct csinn_tensor *src)
{
    shl_mem_free(src->data);
    csinn_free_tensor(src);
}

struct csinn_tensor *shl_ref_convert_float_tensor(struct csinn_tensor *src)
{
    struct csinn_tensor *ret = shl_ref_alloc_float_tensor(src);
    int size = csinn_tensor_size(src);
    float *float_data = ret->data;

    if (src->dtype == CSINN_DTYPE_UINT8) {
        uint8_t *input_data = src->data;
        for (int i = 0; i < size; i++) {
            float_data[i] = shl_ref_uint8_to_float(input_data[i], src);
        }
    } else if (src->dtype == CSINN_DTYPE_INT8) {
        int8_t *input_data = src->data;
        for (int i = 0; i < size; i++) {
            float_data[i] = shl_ref_int8_to_float(input_data[i], src);
        }
    } else {
        return NULL;
    }

    return ret;
}

void shl_ref_conv_free_float_tensor(struct csinn_tensor *input, struct csinn_tensor *output,
                                    struct csinn_tensor *kernel, struct csinn_tensor *bias)
{
    shl_ref_free_float_tensor(input);
    shl_ref_free_float_tensor(output);
    shl_ref_free_float_tensor(kernel);
    shl_ref_free_float_tensor(bias);
}

struct csinn_tensor *shl_ref_tensor_transform_base(struct csinn_tensor *input, int csinn_dtype,
                                                   uint8_t num_bits)
{
    struct csinn_tensor *ret = csinn_alloc_tensor(NULL);
    csinn_tensor_copy(ret, input);
    if (ret->qinfo != NULL) {
        shl_mem_free(ret->qinfo);
        ret->qinfo = NULL;
    }
    ret->quant_channel = 0;
    ret->dtype = csinn_dtype;
    switch (input->layout) {
        case CSINN_LAYOUT_NC1DHWC0:
            ret->layout = CSINN_LAYOUT_NCDHW;
            ret->dim[1] *= input->dim[5];
            ret->dim[5] = 0;
            ret->dim_count = 5;
            break;
        case CSINN_LAYOUT_NC1HWC0:
            ret->layout = CSINN_LAYOUT_NCHW;
            ret->dim[1] *= input->dim[4];
            ret->dim[4] = 0;
            ret->dim_count = 4;
            break;
        case CSINN_LAYOUT_NC1WC0:
            ret->layout = CSINN_LAYOUT_NCW;
            ret->dim[1] *= input->dim[3];
            ret->dim[3] = 0;
            ret->dim_count = 3;
            break;
        case CSINN_LAYOUT_NC1C0:
            ret->layout = CSINN_LAYOUT_NC;
            ret->dim[1] *= input->dim[2];
            ret->dim[2] = 0;
            ret->dim_count = 2;
            break;
        default:
            break;
    }
    if (ret->dim_count == 0) {
        return ret;
    }
    int input_size = csinn_tensor_size(input);
    if (input_size == 0) {
        return ret;
    }
    ret->data = shl_mem_alloc(input_size * num_bits);
    if (csinn_tensor_data_convert(ret, input) == CSINN_TRUE) {
        return ret;
    }
    return NULL;
}

struct csinn_tensor *shl_ref_tensor_transform_f32(struct csinn_tensor *input)
{
    return shl_ref_tensor_transform_base(input, CSINN_DTYPE_FLOAT32, sizeof(float));
}

struct csinn_tensor *shl_ref_tensor_transform_int64(struct csinn_tensor *input)
{
    return shl_ref_tensor_transform_base(input, CSINN_DTYPE_INT64, sizeof(int64_t));
}

int shl_ref_tensor_transform_free_int64(struct csinn_tensor *input)
{
    int size = csinn_tensor_size(input);
    if (size != 0) {
        shl_mem_free(input->data);
    }
    csinn_free_tensor(input);
    return CSINN_TRUE;
}

int shl_ref_tensor_transform_free_f32(struct csinn_tensor *input)
{
    int size = csinn_tensor_size(input);
    if (size != 0) {
        shl_mem_free(input->data);
    }
    csinn_free_tensor(input);
    return CSINN_TRUE;
}

int shl_ref_siso_callback_base(struct csinn_tensor *input, struct csinn_tensor *output,
                               void *params, void *cb)
{
    int (*callback)() = cb;
    int ret;
    struct csinn_tensor *finput = shl_ref_tensor_transform_f32(input);
    struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(output);
    ret = callback(finput, foutput, params);
    csinn_tensor_data_convert(output, foutput);
    shl_ref_tensor_transform_free_f32(finput);
    shl_ref_tensor_transform_free_f32(foutput);
    return ret;
}

int shl_ref_diso_callback_base(struct csinn_tensor *input0, struct csinn_tensor *input1,
                               struct csinn_tensor *output, void *params, void *cb)
{
    int (*callback)() = cb;
    int ret;
    struct csinn_tensor *finput0 = shl_ref_tensor_transform_f32(input0);
    struct csinn_tensor *finput1 = shl_ref_tensor_transform_f32(input1);
    struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(output);
    ret = callback(finput0, finput1, foutput, params);
    csinn_tensor_data_convert(output, foutput);
    shl_ref_tensor_transform_free_f32(finput0);
    shl_ref_tensor_transform_free_f32(finput1);
    shl_ref_tensor_transform_free_f32(foutput);
    return ret;
}

int shl_ref_conv_callback_base(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_tensor *kernel, struct csinn_tensor *bias, void *params,
                               void *cb)
{
    int (*callback)() = cb;
    struct csinn_tensor *float_input = shl_ref_tensor_transform_f32(input);
    struct csinn_tensor *float_kernel = shl_ref_tensor_transform_f32(kernel);
    struct csinn_tensor *float_bias = shl_ref_tensor_transform_f32(bias);
    struct csinn_tensor *float_output = shl_ref_tensor_transform_f32(output);
    int ret = callback(float_input, float_output, float_kernel, float_bias, params);
    csinn_tensor_data_convert(output, float_output);
    shl_ref_tensor_transform_free_f32(float_input);
    shl_ref_tensor_transform_free_f32(float_output);
    shl_ref_tensor_transform_free_f32(float_kernel);
    shl_ref_tensor_transform_free_f32(float_bias);
    return ret;
}

uint8_t *shl_ref_f32_to_input_dtype(uint32_t index, float *data, struct csinn_session *sess)
{
    struct csinn_tensor *ftmp = csinn_alloc_tensor(NULL);
    csinn_tensor_copy(ftmp, sess->input[index]);
    ftmp->data = data;
    ftmp->dtype = CSINN_DTYPE_FLOAT32;
    /* The img preprocess only accepts nchw layout input */
    if (sess->input[index]->layout == CSINN_LAYOUT_NHWC) {
        ftmp->layout = CSINN_LAYOUT_NCHW;
        ftmp->dim[1] = sess->input[index]->dim[3];
        ftmp->dim[2] = sess->input[index]->dim[1];
        ftmp->dim[3] = sess->input[index]->dim[2];
    }
    struct csinn_tensor *ret = csinn_alloc_tensor(NULL);
    csinn_tensor_copy(ret, sess->input[index]);
    ret->data = shl_mem_alloc(csinn_tensor_byte_size(ret));
    csinn_tensor_data_convert(ret, ftmp);
    uint8_t *ret_data = ret->data;
    csinn_free_tensor(ret);
    csinn_free_tensor(ftmp);
    return ret_data;
}

int shl_ref_broadcast_to_shape(struct csinn_tensor *input, struct csinn_tensor *output,
                               int32_t *shape, int32_t shape_count)
{
    int ret;
    if (input->dtype != CSINN_DTYPE_FLOAT32) {
        ret = shl_ref_broadcast_to_shape_quant(input, output, shape, shape_count);
    } else {
        ret = shl_ref_broadcast_to_shape_f32(input, output, shape, shape_count);
    }
    return ret;
}

int shl_ref_broadcast_to_shape_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                                   int32_t *shape, int32_t shape_count)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    int32_t *target_shape = shape;
    int32_t *in_shape = input->dim;
    int32_t in_shape_rank = input->dim_count;
    int32_t target_shape_rank = shape_count;

    // check for broadcast rule
    if (target_shape_rank < in_shape_rank) {
        return CSINN_FALSE;
    }
    for (int i = 0; i < in_shape_rank; i++) {
        if ((in_shape[in_shape_rank - i - 1] != target_shape[target_shape_rank - i - 1]) &&
            (in_shape[in_shape_rank - i - 1] != 1)) {
            shl_debug_error("The shapes of input and target do not meet the rules of broadcast!");
            return CSINN_FALSE;
        }
    }
    int data_size = csinn_tensor_size(input);
    int out_size = csinn_tensor_size(output);

    if (data_size == out_size) {
        memcpy(output_data, input_data, data_size * 4);
        return CSINN_TRUE;
    }

    if (data_size == 1) {
        for (int i = 0; i < out_size; i++) {
            output_data[i] = input_data[0];
        }
        return CSINN_TRUE;
    }

    // full in_shape
    int32_t new_shape[target_shape_rank];
    memcpy(new_shape, in_shape, in_shape_rank * 4);
    if (target_shape_rank > in_shape_rank) {
        for (int i = 0; i < target_shape_rank - in_shape_rank; i++) {
            new_shape[i] = 1;
        }
        for (int i = 0; i < in_shape_rank; i++) {
            int index = target_shape_rank - in_shape_rank + i;
            new_shape[index] = in_shape[i];
        }
    }
    in_shape = new_shape;

    float *output_data_t = shl_mem_alloc(out_size * sizeof(float));
    memcpy(output_data_t, input_data, data_size * 4);
    memcpy(output_data, input_data, data_size * 4);

    for (int i = 0; i < target_shape_rank; i++) {
        int origin_dim = in_shape[target_shape_rank - i - 1];
        int target_dim = target_shape[target_shape_rank - i - 1];

        if (origin_dim != target_dim) {
            data_size = 1;
            for (int i = 0; i < target_shape_rank; i++) {
                data_size *= in_shape[i];
            }
            int inner_size = 1;
            for (int j = target_shape_rank - i - 1; j < target_shape_rank; j++) {
                inner_size *= in_shape[j];
            }
            int target_inner_size = 1;
            for (int j = target_shape_rank - i - 1; j < target_shape_rank; j++) {
                target_inner_size *= target_shape[j];
            }

            float *tmp_arr = (float *)shl_mem_alloc(inner_size * sizeof(float));
            for (int idx = 0; idx < data_size; idx++) {
                // at first output equal to input, then tmp data be saved in output
                tmp_arr[idx % inner_size] = output_data_t[idx];
                if ((idx + 1) % inner_size == 0) {
                    int out_index = ((idx + 1) / inner_size - 1) * target_inner_size;
                    for (int cp_num = 0; cp_num < target_dim; cp_num++) {
                        for (int elem_id = 0; elem_id < inner_size; elem_id++) {
                            output_data[out_index + cp_num * inner_size + elem_id] =
                                tmp_arr[elem_id];
                        }
                    }
                }
            }
            shl_mem_free(tmp_arr);
            in_shape[target_shape_rank - i - 1] = target_shape[target_shape_rank - i - 1];
            memcpy(output_data_t, output_data, out_size * 4);
        }
    }
    shl_mem_free(output_data_t);
    return CSINN_TRUE;
}

int shl_ref_broadcast_to_shape_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                                     int32_t *shape, int32_t shape_count)
{
    struct csinn_tensor *finput = shl_ref_tensor_transform_f32(input);
    struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(output);
    int ret = shl_ref_broadcast_to_shape_f32(finput, foutput, shape, shape_count);
    csinn_tensor_data_convert(output, foutput);
    shl_ref_tensor_transform_free_f32(finput);
    shl_ref_tensor_transform_free_f32(foutput);
    return ret;
}

bool shl_is_first_layer_input(struct csinn_tensor *input, struct csinn_session *sess)
{
    for (int i = 0; i < sess->input_num; i++) {
        if (input == sess->input[i]) {
            return true;
        }
    }
    return false;
}
