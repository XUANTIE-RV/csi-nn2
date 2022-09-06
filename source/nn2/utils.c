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

#include <time.h>

#include "csi_nn.h"
#include "shl_utils.h"

/* https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/internal/quantization_util.cc
 */
static int64_t integer_from_exp(double input, int32_t *shift)
{
    uint64_t kSignMask = 0x8000000000000000LL;
    uint64_t kExponentMask = 0x7ff0000000000000LL;
    int32_t kExponentShift = 52;
    int32_t kExponentBias = 1023;
    uint32_t kExponentIsBadNum = 0x7ff;
    uint64_t kFractionMask = 0x000fffffffc00000LL;
    uint32_t kFractionShift = 22;
    uint32_t kFractionRoundingMask = 0x003fffff;
    uint32_t kFractionRoundingThreshold = 0x00200000;

    // We want to access the bits of the input double value directly, which is
    // tricky to do safely, so use a union to handle the casting.
    union {
        double double_value;
        uint64_t double_as_uint;
    } cast_union;

    cast_union.double_value = input;
    const uint64_t u = cast_union.double_as_uint;

    // If the bitfield is all zeros apart from the sign bit, this is a normalized
    // zero value, so return standard values for this special case.
    if ((u & ~kSignMask) == 0) {
        *shift = 0;
        return 0;
    }

    // Deal with NaNs and Infs, which are always indicated with a fixed pattern in
    // the exponent, and distinguished by whether the fractions are zero or
    // non-zero.
    const uint32_t exponent_part = ((u & kExponentMask) >> kExponentShift);
    if (exponent_part == kExponentIsBadNum) {
        *shift = 0x7fffffff;
        if (u & kFractionMask) {
            // NaN, so just return zero (with the exponent set to INT_MAX).
            return 0;
        } else {
            // Infinity, so return +/- INT_MAX.
            if (u & kSignMask) {
                return 0x8000000000000000;
            } else {
                return 0x7fffffffffffffff;
            }
        }
    }

    // The shift is fairly easy to extract from the high bits of the double value,
    // just by masking it out and applying a bias. The std::frexp() implementation
    // always returns values between 0.5 and 1.0 though, whereas the exponent
    // assumes 1.0 to 2.0 is the standard range, so I add on one to match that
    // interface.
    *shift = (exponent_part - kExponentBias) + 1;

    // There's an implicit high bit in the double format definition, so make sure
    // we include that at the top, and then reconstruct the rest of the fractional
    // value from the remaining fragments.
    int64_t fraction = 0x40000000 + ((u & kFractionMask) >> kFractionShift);

    // We're cutting off some bits at the bottom, so to exactly match the standard
    // frexp implementation here we'll apply rounding by adding one to the least
    // significant bit of the result if the discarded portion is over half of the
    // maximum.
    if ((u & kFractionRoundingMask) > kFractionRoundingThreshold) {
        fraction += 1;
    }
    // Negate the fraction if the sign bit was set.
    if (u & kSignMask) {
        fraction *= -1;
    }

    return fraction;
}

void shl_quantize_multiplier(double double_multiplier, int32_t *quantized_multiplier,
                             int32_t *shift)
{
    if (double_multiplier == 0.) {
        *quantized_multiplier = 0;
        *shift = 0;
        return;
    }

    // If we're trying to avoid the use of floating-point instructions (for
    // example on microcontrollers) then use an alternative implementation
    // that only requires integer and bitwise operations. To enable this, you
    // need to set the define during the build process for your platform.
    int64_t q_fixed = integer_from_exp(double_multiplier, shift);

    if (q_fixed == (1ll << 31)) {
        q_fixed /= 2;
        ++*shift;
    }
    // A shift amount smaller than -31 would cause all bits to be shifted out
    // and thus all results would be zero. We implement that instead with
    // q_fixed==0, so as to avoid hitting issues with right-shift
    // operations with shift amounts greater than 31. Note that this happens
    // roughly when abs(double_multiplier) < 2^-31 and the present handling means
    // that we're effectively flushing tiny double_multiplier's to zero.
    // We could conceivably handle values in the range (roughly) [32, 63]
    // as 'denormals' i.e. (shift==0, q_fixed < 2^30). In that point of view
    // the present handling is just doing 'flush denormals to zero'. We could
    // reconsider and actually generate nonzero denormals if a need arises.
    if (*shift < -31) {
        *shift = 0;
        q_fixed = 0;
    }
    *quantized_multiplier = (int32_t)(q_fixed);
}

void shl_statistical_mean_std(float *data, int sz)
{
    int i = 0;
    float max_value = data[0];
    float min_value = data[0];
    double std = 0.0;
    double sum = 0.0;
    for (i = 0; i < sz; i++) {
        sum += data[i];
        if (data[i] > max_value) {
            max_value = data[i];
        }
        if (data[i] < min_value) {
            min_value = data[i];
        }
    }
    double mean = sum / sz;
    sum = 0.0;
    for (i = 0; i < sz; i++) {
        sum += ((data[i] - mean) * (data[i] - mean));
    }
    std = sum / sz;
    printf("The max_value of output: %lf\n", max_value);
    printf("The min_value of output: %lf\n", min_value);
    printf("The mean_value of output: %lf\n", mean);
    printf("The std_value of output: %lf\n", std);
}

void shl_get_top5(float *buf, uint32_t size, float *prob, uint32_t *class)
{
    uint32_t i, j, k;

    memset(prob, 0xfe, sizeof(float) * 5);
    memset(class, 0xff, sizeof(uint32_t) * 5);

    for (j = 0; j < 5; j++) {
        for (i = 0; i < size; i++) {
            for (k = 0; k < 5; k++) {
                if (i == class[k]) {
                    break;
                }
            }

            if (k != 5) {
                continue;
            }

            if (buf[i] > prob[j]) {
                prob[j] = buf[i];
                class[j] = i;
            }
        }
    }
}

void shl_show_top5(struct csinn_tensor *output, struct csinn_session *sess)
{
    uint32_t i, size;
    uint32_t class[5];
    float prob[5];

    if (output->data == NULL) {
        return;
    }

    size = 1;
    for (i = 0; i < output->dim_count; i++) {
        size *= output->dim[i];
    }

    // #ifdef SHL_DEBUG
    shl_statistical_mean_std(output->data, size);
    // #endif

    shl_get_top5(output->data, size, prob, class);

    printf(" ============ top5: ===========\n");
    size = size > 5 ? 5 : size;
    for (i = 0; i < size; i++) {
        printf("%3d: %8.6f\n", class[i], prob[i]);
    }
}

int csinn_tensor_size(struct csinn_tensor *tensor)
{
    if (tensor->dim_count == 0) {
        return 0;
    }
    int size = 1;
    if (tensor->layout == CSINN_LAYOUT_O32I32) {
        size = tensor->dim[1] * ((tensor->dim[0] + 31) / 32) * 32;
    } else if (tensor->layout == CSINN_LAYOUT_O32HWI32) {
        size = tensor->dim[1] * tensor->dim[2] * tensor->dim[3] * ((tensor->dim[0] + 31) / 32) * 32;
    } else if (tensor->layout == CSINN_LAYOUT_1HW32O32) {
        size = tensor->dim[1] * tensor->dim[2] * ((tensor->dim[3] + 31) / 32) * 32;
    } else {
        for (int i = 0; i < tensor->dim_count; i++) {
            size *= tensor->dim[i];
        }
    }
    return size;
}

int csinn_tensor_byte_size(struct csinn_tensor *tensor)
{
    int size = csinn_tensor_size(tensor);
    switch (tensor->dtype) {
        case CSINN_DTYPE_INT4:
            /* FIXME: round to byte */
            size = (size + 1) / 2;
            break;
        case CSINN_DTYPE_INT16:
        case CSINN_DTYPE_UINT16:
        case CSINN_DTYPE_FLOAT16:
        case CSINN_DTYPE_BFLOAT16:
            size *= 2;
            break;
        case CSINN_DTYPE_INT32:
        case CSINN_DTYPE_UINT32:
        case CSINN_DTYPE_FLOAT32:
            size *= 4;
            break;
        case CSINN_DTYPE_FLOAT64:
            size *= 8;
            break;
        default:
            break;
    }
    return size;
}

struct csinn_tensor *csinn_alloc_tensor(struct csinn_session *session)
{
    struct csinn_tensor *ret = shl_mem_alloc(sizeof(struct csinn_tensor));
    if (session != NULL) {
        ret->dtype = session->base_dtype;
        ret->layout = session->base_layout;
        ret->sess = session;
    }
    ret->quant_channel = 1;
    ret->qinfo = shl_mem_alloc(sizeof(struct csinn_quant_info));
    return ret;
}

void csinn_realloc_quant_info(struct csinn_tensor *tensor, int quant_info_num)
{
    tensor->quant_channel = quant_info_num;
    tensor->qinfo =
        shl_mem_realloc(tensor->qinfo, quant_info_num * sizeof(struct csinn_quant_info));
}

void csinn_tensor_copy(struct csinn_tensor *dest, struct csinn_tensor *src)
{
    dest->data = src->data;
    dest->dtype = src->dtype;
    memcpy(dest->dim, src->dim, MAX_DIM * 4);
    dest->dim_count = src->dim_count;
    dest->name = src->name;
    dest->layout = src->layout;
    if (src->quant_channel != dest->quant_channel && src->quant_channel != 0) {
        csinn_realloc_quant_info(dest, src->quant_channel);
    }
    memcpy(dest->qinfo, src->qinfo, sizeof(struct csinn_quant_info) * src->quant_channel);
    dest->sess = src->sess;
    dest->is_const = src->is_const;
}

void csinn_free_tensor(struct csinn_tensor *tensor)
{
    if (tensor->qinfo != NULL) {
        shl_mem_free(tensor->qinfo);
    }
    shl_mem_free(tensor);
}

void *csinn_alloc_params(int params_size, struct csinn_session *session)
{
    struct csinn_params_base *params = shl_mem_alloc(params_size);
    if (session != NULL) {
        params->api = session->base_api;
        params->layout = session->base_layout;
        params->sess = session;
    }
    params->cb = shl_mem_alloc(sizeof(struct csinn_callback));
    return params;
}

void csinn_free_params(void *params) { shl_mem_free(params); }

static float int4_to_float_base(int8_t i, struct csinn_tensor *t, int index)
{
    return ((float)i - t->qinfo[index].zero_point) * t->qinfo[index].scale;
}

static float uint8_to_float_base(uint8_t i, struct csinn_tensor *t, int index)
{
    return ((float)i - t->qinfo[index].zero_point) * t->qinfo[index].scale;
}

static float int8_to_float_base(int8_t i, struct csinn_tensor *t, int index)
{
    return ((float)i - t->qinfo[index].zero_point) * t->qinfo[index].scale;
}

static float int16_to_float_base(int16_t i, struct csinn_tensor *t, int index)
{
    return ((float)i - t->qinfo[index].zero_point) * t->qinfo[index].scale;
}

static float int32_to_float_base(int32_t i, struct csinn_tensor *t, int index)
{
    return (float)i * t->qinfo[index].scale;
}

static int8_t float_to_int4_base(float i, struct csinn_tensor *t, int index)
{
    float ret = round(i / t->qinfo[index].scale) + t->qinfo[index].zero_point;
    if (ret > 7) {
        return 7;
    } else if (ret < -8) {
        return -8;
    } else {
        return ret;
    }
}

static uint8_t float_to_uint8_base(float i, struct csinn_tensor *t, int index)
{
    float ret = round(i / t->qinfo[index].scale) + t->qinfo[index].zero_point;
    if (ret > 255) {
        return 255;
    } else if (ret < 0) {
        return 0;
    } else {
        return ret;
    }
}

static int8_t float_to_int8_base(float i, struct csinn_tensor *t, int index)
{
    float ret = round(i / t->qinfo[index].scale) + t->qinfo[index].zero_point;
    if (ret > 127) {
        return 127;
    } else if (ret < -128) {
        return -128;
    } else {
        return ret;
    }
}

static int16_t float_to_int16_base(float i, struct csinn_tensor *t, int index)
{
    float ret = round(i / t->qinfo[index].scale) + t->qinfo[index].zero_point;
    if (ret > 32767) {
        return 32767;
    } else if (ret < -32768) {
        return -32768;
    } else {
        return ret;
    }
}

static int16_t float32_to_float16_base(float value)
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

static float float16_to_float32_base(int16_t value)
{
    float ret;
    if (value == 0 || value == 0x8000) {
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

static int16_t float32_to_bfloat16_base(float value)
{
    int16_t ret;
    int32_t org_format = *(int32_t *)&value;
    ret = (org_format & 0xffff0000) >> 16;
    return ret;
}

static float bfloat16_to_float32_base(int16_t value)
{
    float ret;
    int32_t ret_format = value << 16;
    ret = *(float *)&ret_format;
    return ret;
}

/* Only for CSINN_LAYOUT_OHWI, HWI's size align */
static void axis0_int4_to_float_alignHWI(struct csinn_tensor *dest, struct csinn_tensor *src,
                                         int inner_size)
{
    int8_t *src_data = src->data;
    float *dest_data = dest->data;
    int32_t q_size = src->quant_channel;
    for (int i = 0; i < q_size; i++) {
        for (int j = 0; j < inner_size; j++) {
            int index = i * inner_size + j;
            int in_index = i * ((inner_size + 1) / 2) + j / 2;
            float ret = 0;
            int8_t src_tmp = 0;
            /* int4 little endian */
            if (j % 2) {
                src_tmp = src_data[in_index] & 0xf0;
                ret = int4_to_float_base(src_tmp >> 4, src, i);
            } else {
                src_tmp = (src_data[in_index] & 0xf) << 4;
                ret = int4_to_float_base(src_tmp >> 4, src, i);
            }
            dest_data[index] = ret;
        }
    }
}

/* Only for CSINN_LAYOUT_OHWI, HWI's size align */
static void axis0_float_to_int4_alignHWI(struct csinn_tensor *dest, struct csinn_tensor *src,
                                         int inner_size)
{
    float *src_data = src->data;
    int8_t *dest_data = dest->data;
    int32_t q_size = dest->quant_channel;
    for (int i = 0; i < q_size; i++) {
        for (int j = 0; j < inner_size; j++) {
            int index = i * inner_size + j;
            int input_val = float_to_int4_base(src_data[index], dest, i);
            int out_index = i * ((inner_size + 1) / 2) + j / 2;
            /* int4 little endian */
            if (j % 2) {
                dest_data[out_index] = (dest_data[out_index] & 0xf) | (input_val << 4);
            } else {
                /* init as 0 at first access half of byte */
                dest_data[out_index] = 0;
                dest_data[out_index] = (dest_data[out_index] & 0xf0) | (input_val & 0xf);
            }
        }
    }
}

static void nchw_int4_to_float(struct csinn_tensor *dest, struct csinn_tensor *src, int n,
                               int inner_size)
{
    int8_t *src_data = src->data;
    float *dest_data = dest->data;
    int32_t q_size = src->quant_channel;
    for (int i = 0; i < q_size; i++) {
        for (int j = 0; j < inner_size; j++) {
            int index = n * q_size * inner_size + i * inner_size + j;
            int in_index = index / 2;
            float ret = 0;
            int8_t src_tmp = 0;
            /* int4 little endian */
            if (index % 2) {
                src_tmp = src_data[in_index] & 0xf0;
                ret = int4_to_float_base(src_tmp >> 4, src, i);
            } else {
                src_tmp = (src_data[in_index] & 0xf) << 4;
                ret = int4_to_float_base(src_tmp >> 4, src, i);
            }
            dest_data[index] = ret;
        }
    }
}

static void nhwc_int4_to_float(struct csinn_tensor *dest, struct csinn_tensor *src, int n,
                               int inner_size)
{
    int8_t *src_data = src->data;
    float *dest_data = dest->data;
    int32_t q_size = src->quant_channel;
    for (int j = 0; j < inner_size; j++) {
        for (int i = 0; i < q_size; i++) {
            int index = n * q_size * inner_size + j * q_size + i;
            int in_index = index / 2;
            float ret = 0;
            int8_t src_tmp = 0;
            /* int4 little endian */
            if (index % 2) {
                src_tmp = src_data[in_index] & 0xf0;
                ret = int4_to_float_base(src_tmp >> 4, src, i);
            } else {
                src_tmp = (src_data[in_index] & 0xf) << 4;
                ret = int4_to_float_base(src_tmp >> 4, src, i);
            }
            dest_data[index] = ret;
        }
    }
}

static void nchw_float_to_int4(struct csinn_tensor *dest, struct csinn_tensor *src, int n,
                               int inner_size)
{
    float *src_data = src->data;
    int8_t *dest_data = dest->data;
    int32_t q_size = dest->quant_channel;
    for (int i = 0; i < q_size; i++) {
        for (int j = 0; j < inner_size; j++) {
            int index = n * q_size * inner_size + i * inner_size + j;
            int input_val = float_to_int4_base(src_data[index], dest, i);
            int out_index = index / 2;
            /* int4 little endian */
            if (index % 2) {
                dest_data[out_index] = (dest_data[out_index] & 0xf) | (input_val << 4);
            } else {
                dest_data[out_index] = (dest_data[out_index] & 0xf0) | (input_val & 0xf);
            }
        }
    }
}

static void nhwc_float_to_int4(struct csinn_tensor *dest, struct csinn_tensor *src, int n,
                               int inner_size)
{
    float *src_data = src->data;
    int8_t *dest_data = dest->data;
    int32_t q_size = dest->quant_channel;
    for (int j = 0; j < inner_size; j++) {
        for (int i = 0; i < q_size; i++) {
            int index = n * q_size * inner_size + j * q_size + i;
            int input_val = float_to_int4_base(src_data[index], dest, i);
            int out_index = index / 2;
            /* int4 little endian */
            if (index % 2) {
                dest_data[out_index] = (dest_data[out_index] & 0xf) | (input_val << 4);
            } else {
                dest_data[out_index] = (dest_data[out_index] & 0xf0) | input_val;
            }
        }
    }
}

static void nchw_uint8_to_float(struct csinn_tensor *dest, struct csinn_tensor *src, int n,
                                int inner_size)
{
    uint8_t *src_data = src->data;
    float *dest_data = dest->data;
    int32_t q_size = src->quant_channel;
    for (int i = 0; i < q_size; i++) {
        for (int j = 0; j < inner_size; j++) {
            int index = n * q_size * inner_size + i * inner_size + j;
            dest_data[index] = uint8_to_float_base(src_data[index], src, i);
        }
    }
}

static void nhwc_uint8_to_float(struct csinn_tensor *dest, struct csinn_tensor *src, int n,
                                int inner_size)
{
    uint8_t *src_data = src->data;
    float *dest_data = dest->data;
    int32_t q_size = src->quant_channel;
    for (int j = 0; j < inner_size; j++) {
        for (int i = 0; i < q_size; i++) {
            int index = n * q_size * inner_size + j * q_size + i;
            dest_data[index] = uint8_to_float_base(src_data[index], src, i);
        }
    }
}

static void nchw_float_to_uint8(struct csinn_tensor *dest, struct csinn_tensor *src, int n,
                                int inner_size)
{
    float *src_data = src->data;
    uint8_t *dest_data = dest->data;
    int32_t q_size = dest->quant_channel;
    for (int i = 0; i < q_size; i++) {
        for (int j = 0; j < inner_size; j++) {
            int index = n * q_size * inner_size + i * inner_size + j;
            dest_data[index] = float_to_uint8_base(src_data[index], dest, i);
        }
    }
}
static void nhwc_float_to_uint8(struct csinn_tensor *dest, struct csinn_tensor *src, int n,
                                int inner_size)
{
    float *src_data = src->data;
    uint8_t *dest_data = dest->data;
    int32_t q_size = dest->quant_channel;
    for (int j = 0; j < inner_size; j++) {
        for (int i = 0; i < q_size; i++) {
            int index = n * q_size * inner_size + j * q_size + i;
            dest_data[index] = float_to_uint8_base(src_data[index], dest, i);
        }
    }
}

static void nchw_int8_to_float(struct csinn_tensor *dest, struct csinn_tensor *src, int n,
                               int inner_size)
{
    int8_t *src_data = src->data;
    float *dest_data = dest->data;
    int32_t q_size = src->quant_channel;
    for (int i = 0; i < q_size; i++) {
        for (int j = 0; j < inner_size; j++) {
            int index = n * q_size * inner_size + i * inner_size + j;
            dest_data[index] = int8_to_float_base(src_data[index], src, i);
        }
    }
}
static void nhwc_int8_to_float(struct csinn_tensor *dest, struct csinn_tensor *src, int n,
                               int inner_size)
{
    int8_t *src_data = src->data;
    float *dest_data = dest->data;
    int32_t q_size = src->quant_channel;
    for (int j = 0; j < inner_size; j++) {
        for (int i = 0; i < q_size; i++) {
            int index = n * q_size * inner_size + j * q_size + i;
            dest_data[index] = int8_to_float_base(src_data[index], src, i);
        }
    }
}

static void nchw_float_to_int8(struct csinn_tensor *dest, struct csinn_tensor *src, int n,
                               int inner_size)
{
    float *src_data = src->data;
    int8_t *dest_data = dest->data;
    int32_t q_size = dest->quant_channel;
    for (int i = 0; i < q_size; i++) {
        for (int j = 0; j < inner_size; j++) {
            int index = n * q_size * inner_size + i * inner_size + j;
            dest_data[index] = float_to_int8_base(src_data[index], dest, i);
        }
    }
}

static void nhwc_float_to_int8(struct csinn_tensor *dest, struct csinn_tensor *src, int n,
                               int inner_size)
{
    float *src_data = src->data;
    int8_t *dest_data = dest->data;
    int32_t q_size = dest->quant_channel;
    for (int j = 0; j < inner_size; j++) {
        for (int i = 0; i < q_size; i++) {
            int index = n * q_size * inner_size + j * q_size + i;
            dest_data[index] = float_to_int8_base(src_data[index], dest, i);
        }
    }
}

static void nchw_int16_to_float(struct csinn_tensor *dest, struct csinn_tensor *src, int n,
                                int inner_size)
{
    int16_t *src_data = src->data;
    float *dest_data = dest->data;
    int32_t q_size = src->quant_channel;
    for (int i = 0; i < q_size; i++) {
        for (int j = 0; j < inner_size; j++) {
            int index = n * q_size * inner_size + i * inner_size + j;
            dest_data[index] = int16_to_float_base(src_data[index], src, i);
        }
    }
}

static void nhwc_int16_to_float(struct csinn_tensor *dest, struct csinn_tensor *src, int n,
                                int inner_size)
{
    int16_t *src_data = src->data;
    float *dest_data = dest->data;
    int32_t q_size = src->quant_channel;
    for (int j = 0; j < inner_size; j++) {
        for (int i = 0; i < q_size; i++) {
            int index = n * q_size * inner_size + j * q_size + i;
            dest_data[index] = int16_to_float_base(src_data[index], src, i);
        }
    }
}

static void nchw_float_to_int16(struct csinn_tensor *dest, struct csinn_tensor *src, int n,
                                int inner_size)
{
    float *src_data = src->data;
    int16_t *dest_data = dest->data;
    int32_t q_size = dest->quant_channel;
    for (int i = 0; i < q_size; i++) {
        for (int j = 0; j < inner_size; j++) {
            int index = n * q_size * inner_size + i * inner_size + j;
            dest_data[index] = float_to_int16_base(src_data[index], dest, i);
        }
    }
}

static void nhwc_float_to_int16(struct csinn_tensor *dest, struct csinn_tensor *src, int n,
                                int inner_size)
{
    float *src_data = src->data;
    int16_t *dest_data = dest->data;
    int32_t q_size = dest->quant_channel;
    for (int j = 0; j < inner_size; j++) {
        for (int i = 0; i < q_size; i++) {
            int index = n * q_size * inner_size + j * q_size + i;
            dest_data[index] = float_to_int16_base(src_data[index], dest, i);
        }
    }
}

static void nchw_int32_to_float(struct csinn_tensor *dest, struct csinn_tensor *src, int n,
                                int inner_size)
{
    int32_t *src_data = src->data;
    float *dest_data = dest->data;
    int32_t q_size = src->quant_channel;
    for (int i = 0; i < q_size; i++) {
        for (int j = 0; j < inner_size; j++) {
            int index = n * q_size * inner_size + i * inner_size + j;
            dest_data[index] = int32_to_float_base(src_data[index], src, i);
        }
    }
}

static void nhwc_int32_to_float(struct csinn_tensor *dest, struct csinn_tensor *src, int n,
                                int inner_size)
{
    int32_t *src_data = src->data;
    float *dest_data = dest->data;
    int32_t q_size = src->quant_channel;
    for (int j = 0; j < inner_size; j++) {
        for (int i = 0; i < q_size; i++) {
            int index = n * q_size * inner_size + j * q_size + i;
            dest_data[index] = int32_to_float_base(src_data[index], src, i);
        }
    }
}

static void csinn_f16_to_float(struct csinn_tensor *dest, struct csinn_tensor *src)
{
    int16_t *src_data = src->data;
    float *dest_data = dest->data;
    int32_t size = csinn_tensor_size(src);
    for (int j = 0; j < size; j++) {
        dest_data[j] = float16_to_float32_base(src_data[j]);
    }
}

static void csinn_float_to_f16(struct csinn_tensor *dest, struct csinn_tensor *src)
{
    float *src_data = src->data;
    int16_t *dest_data = dest->data;
    int32_t size = csinn_tensor_size(src);
    for (int i = 0; i < size; i++) {
        dest_data[i] = float32_to_float16_base(src_data[i]);
    }
}

static void bf16_to_float(struct csinn_tensor *dest, struct csinn_tensor *src)
{
    int16_t *src_data = src->data;
    float *dest_data = dest->data;
    int32_t size = csinn_tensor_size(src);
    for (int j = 0; j < size; j++) {
        dest_data[j] = bfloat16_to_float32_base(src_data[j]);
    }
}

static void float_to_bf16(struct csinn_tensor *dest, struct csinn_tensor *src)
{
    float *src_data = src->data;
    int16_t *dest_data = dest->data;
    int32_t size = csinn_tensor_size(src);
    for (int i = 0; i < size; i++) {
        dest_data[i] = float32_to_bfloat16_base(src_data[i]);
    }
}

static int tensor_data_convert_weight(struct csinn_tensor *dest, struct csinn_tensor *src)
{
    int size = csinn_tensor_size(src);
    int inner_size = src->quant_channel == 0 ? size : size / src->quant_channel;
    if (dest->dtype == CSINN_DTYPE_FLOAT32 && src->dtype == CSINN_DTYPE_INT4) {
        switch (src->layout) {
            case CSINN_LAYOUT_O:
            case CSINN_LAYOUT_OI:
            case CSINN_LAYOUT_OIW:
            case CSINN_LAYOUT_OIHW:
            case CSINN_LAYOUT_OIDHW:
            case CSINN_LAYOUT_O1HW:
            case CSINN_LAYOUT_OWI:
            case CSINN_LAYOUT_ODHWI:
                nchw_int4_to_float(dest, src, 0, inner_size);
                break;
            case CSINN_LAYOUT_OHWI:
                axis0_int4_to_float_alignHWI(dest, src, inner_size);
                break;
            case CSINN_LAYOUT_1HWO:
                nhwc_int4_to_float(dest, src, 0, inner_size);
                break;
            default:
                break;
        }
    } else if (dest->dtype == CSINN_DTYPE_INT4 && src->dtype == CSINN_DTYPE_FLOAT32) {
        switch (src->layout) {
            case CSINN_LAYOUT_O:
            case CSINN_LAYOUT_OI:
            case CSINN_LAYOUT_OIW:
            case CSINN_LAYOUT_OIHW:
            case CSINN_LAYOUT_OIDHW:
            case CSINN_LAYOUT_O1HW:
            case CSINN_LAYOUT_OWI:
            case CSINN_LAYOUT_ODHWI:
                nchw_float_to_int4(dest, src, 0, inner_size);
                break;
            case CSINN_LAYOUT_OHWI:
                axis0_float_to_int4_alignHWI(dest, src, inner_size);
            case CSINN_LAYOUT_1HWO:
                nhwc_float_to_int4(dest, src, 0, inner_size);
                break;
            default:
                break;
        }
    } else if (dest->dtype == CSINN_DTYPE_FLOAT32 && src->dtype == CSINN_DTYPE_UINT8) {
        switch (src->layout) {
            case CSINN_LAYOUT_O:
            case CSINN_LAYOUT_OI:
            case CSINN_LAYOUT_OIW:
            case CSINN_LAYOUT_OIHW:
            case CSINN_LAYOUT_OIDHW:
            case CSINN_LAYOUT_O1HW:
            case CSINN_LAYOUT_OWI:
            case CSINN_LAYOUT_OHWI:
            case CSINN_LAYOUT_ODHWI:
                nchw_uint8_to_float(dest, src, 0, inner_size);
                break;
            case CSINN_LAYOUT_1HWO:
                nhwc_uint8_to_float(dest, src, 0, inner_size);
                break;
            default:
                break;
        }
    } else if (dest->dtype == CSINN_DTYPE_UINT8 && src->dtype == CSINN_DTYPE_FLOAT32) {
        switch (src->layout) {
            case CSINN_LAYOUT_O:
            case CSINN_LAYOUT_OI:
            case CSINN_LAYOUT_OIW:
            case CSINN_LAYOUT_OIHW:
            case CSINN_LAYOUT_OIDHW:
            case CSINN_LAYOUT_O1HW:
            case CSINN_LAYOUT_OWI:
            case CSINN_LAYOUT_OHWI:
            case CSINN_LAYOUT_ODHWI:
                nchw_float_to_uint8(dest, src, 0, inner_size);
                break;
            case CSINN_LAYOUT_1HWO:
                nhwc_float_to_uint8(dest, src, 0, inner_size);
                break;
            default:
                break;
        }
    } else if (dest->dtype == CSINN_DTYPE_FLOAT32 && src->dtype == CSINN_DTYPE_INT8) {
        switch (src->layout) {
            case CSINN_LAYOUT_O:
            case CSINN_LAYOUT_OI:
            case CSINN_LAYOUT_OIW:
            case CSINN_LAYOUT_OIHW:
            case CSINN_LAYOUT_OIDHW:
            case CSINN_LAYOUT_O1HW:
            case CSINN_LAYOUT_OWI:
            case CSINN_LAYOUT_OHWI:
            case CSINN_LAYOUT_ODHWI:
                nchw_int8_to_float(dest, src, 0, inner_size);
                break;
            case CSINN_LAYOUT_1HWO:
                nhwc_int8_to_float(dest, src, 0, inner_size);
                break;
            default:
                break;
        }
    } else if (dest->dtype == CSINN_DTYPE_INT8 && src->dtype == CSINN_DTYPE_FLOAT32) {
        switch (src->layout) {
            case CSINN_LAYOUT_O:
            case CSINN_LAYOUT_OI:
            case CSINN_LAYOUT_OIW:
            case CSINN_LAYOUT_OIHW:
            case CSINN_LAYOUT_OIDHW:
            case CSINN_LAYOUT_O1HW:
            case CSINN_LAYOUT_OWI:
            case CSINN_LAYOUT_OHWI:
            case CSINN_LAYOUT_ODHWI:
                nchw_float_to_int8(dest, src, 0, inner_size);
                break;
            case CSINN_LAYOUT_1HWO:
                nhwc_float_to_int8(dest, src, 0, inner_size);
                break;
            default:
                break;
        }
    } else if (dest->dtype == CSINN_DTYPE_FLOAT32 && src->dtype == CSINN_DTYPE_INT16) {
        switch (src->layout) {
            case CSINN_LAYOUT_O:
            case CSINN_LAYOUT_OI:
            case CSINN_LAYOUT_OIW:
            case CSINN_LAYOUT_OIHW:
            case CSINN_LAYOUT_OIDHW:
            case CSINN_LAYOUT_O1HW:
            case CSINN_LAYOUT_OWI:
            case CSINN_LAYOUT_OHWI:
            case CSINN_LAYOUT_ODHWI:
                nchw_int16_to_float(dest, src, 0, inner_size);
                break;
            case CSINN_LAYOUT_1HWO:
                nhwc_int16_to_float(dest, src, 0, inner_size);
                break;
            default:
                break;
        }
    } else if (dest->dtype == CSINN_DTYPE_INT16 && src->dtype == CSINN_DTYPE_FLOAT32) {
        switch (src->layout) {
            case CSINN_LAYOUT_O:
            case CSINN_LAYOUT_OI:
            case CSINN_LAYOUT_OIW:
            case CSINN_LAYOUT_OIHW:
            case CSINN_LAYOUT_OIDHW:
            case CSINN_LAYOUT_O1HW:
            case CSINN_LAYOUT_OWI:
            case CSINN_LAYOUT_OHWI:
            case CSINN_LAYOUT_ODHWI:
                nchw_float_to_int16(dest, src, 0, inner_size);
                break;
            case CSINN_LAYOUT_1HWO:
                nhwc_float_to_int16(dest, src, 0, inner_size);
                break;
            default:
                break;
        }
    } else if (dest->dtype == CSINN_DTYPE_FLOAT32 && src->dtype == CSINN_DTYPE_INT32) {
        switch (src->layout) {
            case CSINN_LAYOUT_O:
            case CSINN_LAYOUT_OI:
            case CSINN_LAYOUT_OIW:
            case CSINN_LAYOUT_OIHW:
            case CSINN_LAYOUT_OIDHW:
            case CSINN_LAYOUT_O1HW:
            case CSINN_LAYOUT_OWI:
            case CSINN_LAYOUT_OHWI:
            case CSINN_LAYOUT_ODHWI:
                nchw_int32_to_float(dest, src, 0, inner_size);
                break;
            case CSINN_LAYOUT_1HWO:
                nhwc_int32_to_float(dest, src, 0, inner_size);
                break;
            default:
                break;
        }
    } else if (dest->dtype == CSINN_DTYPE_FLOAT16 && src->dtype == CSINN_DTYPE_FLOAT32) {
        csinn_float_to_f16(dest, src);
    } else if (dest->dtype == CSINN_DTYPE_FLOAT32 && src->dtype == CSINN_DTYPE_FLOAT16) {
        csinn_f16_to_float(dest, src);
    } else if (dest->dtype == CSINN_DTYPE_BFLOAT16 && src->dtype == CSINN_DTYPE_FLOAT32) {
        float_to_bf16(dest, src);
    } else if (dest->dtype == CSINN_DTYPE_FLOAT32 && src->dtype == CSINN_DTYPE_BFLOAT16) {
        bf16_to_float(dest, src);
    } else if (dest->dtype == src->dtype) {
        memcpy(dest->data, src->data, csinn_tensor_byte_size(src));
    } else {
        return CSINN_FALSE;
    }
    return CSINN_TRUE;
}

int tensor_data_convert_activation(struct csinn_tensor *dest, struct csinn_tensor *src)
{
    int size = csinn_tensor_size(src);
    int32_t q_size = src->quant_channel != 0 ? src->quant_channel : dest->quant_channel;
    if (q_size == 0) {
        q_size = 1;
    }
    int inner_size = size / q_size / src->dim[0];
    if (dest->dtype == CSINN_DTYPE_FLOAT32 && src->dtype == CSINN_DTYPE_INT4) {
        for (int n = 0; n < src->dim[0]; n++) {
            if (src->layout >= CSINN_LAYOUT_N && src->layout <= CSINN_LAYOUT_NCDHW) {
                nchw_int4_to_float(dest, src, n, inner_size);
            } else if (src->layout >= CSINN_LAYOUT_NWC && src->layout <= CSINN_LAYOUT_NDHWC) {
                nhwc_int4_to_float(dest, src, n, inner_size);
            }
        }
    } else if (dest->dtype == CSINN_DTYPE_INT4 && src->dtype == CSINN_DTYPE_FLOAT32) {
        for (int n = 0; n < src->dim[0]; n++) {
            if (src->layout >= CSINN_LAYOUT_N && src->layout <= CSINN_LAYOUT_NCDHW) {
                nchw_float_to_int4(dest, src, n, inner_size);
            } else if (src->layout >= CSINN_LAYOUT_NWC && src->layout <= CSINN_LAYOUT_NDHWC) {
                nhwc_float_to_int4(dest, src, n, inner_size);
            }
        }
    } else if (dest->dtype == CSINN_DTYPE_FLOAT32 && src->dtype == CSINN_DTYPE_UINT8) {
        for (int n = 0; n < src->dim[0]; n++) {
            if (src->layout >= CSINN_LAYOUT_N && src->layout <= CSINN_LAYOUT_NCDHW) {
                nchw_uint8_to_float(dest, src, n, inner_size);
            } else if (src->layout >= CSINN_LAYOUT_NWC && src->layout <= CSINN_LAYOUT_NDHWC) {
                nhwc_uint8_to_float(dest, src, n, inner_size);
            }
        }
    } else if (dest->dtype == CSINN_DTYPE_UINT8 && src->dtype == CSINN_DTYPE_FLOAT32) {
        for (int n = 0; n < src->dim[0]; n++) {
            if (src->layout >= CSINN_LAYOUT_N && src->layout <= CSINN_LAYOUT_NCDHW) {
                nchw_float_to_uint8(dest, src, n, inner_size);
            } else if (src->layout >= CSINN_LAYOUT_NWC && src->layout <= CSINN_LAYOUT_NDHWC) {
                nhwc_float_to_uint8(dest, src, n, inner_size);
            }
        }
    } else if (dest->dtype == CSINN_DTYPE_FLOAT32 && src->dtype == CSINN_DTYPE_INT8) {
        for (int n = 0; n < src->dim[0]; n++) {
            if (src->layout >= CSINN_LAYOUT_N && src->layout <= CSINN_LAYOUT_NCDHW) {
                nchw_int8_to_float(dest, src, n, inner_size);
            } else if (src->layout >= CSINN_LAYOUT_NWC && src->layout <= CSINN_LAYOUT_NDHWC) {
                nhwc_int8_to_float(dest, src, n, inner_size);
            }
        }
    } else if (dest->dtype == CSINN_DTYPE_INT8 && src->dtype == CSINN_DTYPE_FLOAT32) {
        for (int n = 0; n < src->dim[0]; n++) {
            if (src->layout >= CSINN_LAYOUT_N && src->layout <= CSINN_LAYOUT_NCDHW) {
                nchw_float_to_int8(dest, src, n, inner_size);
            } else if (src->layout >= CSINN_LAYOUT_NWC && src->layout <= CSINN_LAYOUT_NDHWC) {
                nhwc_float_to_int8(dest, src, n, inner_size);
            }
        }
    } else if (dest->dtype == CSINN_DTYPE_FLOAT32 && src->dtype == CSINN_DTYPE_INT16) {
        for (int n = 0; n < src->dim[0]; n++) {
            if (src->layout >= CSINN_LAYOUT_N && src->layout <= CSINN_LAYOUT_NCDHW) {
                nchw_int16_to_float(dest, src, n, inner_size);
            } else if (src->layout >= CSINN_LAYOUT_NWC && src->layout <= CSINN_LAYOUT_NDHWC) {
                nhwc_int16_to_float(dest, src, n, inner_size);
            }
        }
    } else if (dest->dtype == CSINN_DTYPE_INT16 && src->dtype == CSINN_DTYPE_FLOAT32) {
        for (int n = 0; n < src->dim[0]; n++) {
            if (src->layout >= CSINN_LAYOUT_N && src->layout <= CSINN_LAYOUT_NCDHW) {
                nchw_float_to_int16(dest, src, n, inner_size);
            } else if (src->layout >= CSINN_LAYOUT_NWC && src->layout <= CSINN_LAYOUT_NDHWC) {
                nhwc_float_to_int16(dest, src, n, inner_size);
            }
        }
    } else if (dest->dtype == CSINN_DTYPE_FLOAT32 && src->dtype == CSINN_DTYPE_INT32) {
        for (int n = 0; n < src->dim[0]; n++) {
            if (src->layout >= CSINN_LAYOUT_N && src->layout <= CSINN_LAYOUT_NCDHW) {
                nchw_int32_to_float(dest, src, n, inner_size);
            } else if (src->layout >= CSINN_LAYOUT_NWC && src->layout <= CSINN_LAYOUT_NDHWC) {
                nhwc_int32_to_float(dest, src, n, inner_size);
            }
        }
    } else if (dest->dtype == CSINN_DTYPE_FLOAT16 && src->dtype == CSINN_DTYPE_FLOAT32) {
        csinn_float_to_f16(dest, src);
    } else if (dest->dtype == CSINN_DTYPE_FLOAT32 && src->dtype == CSINN_DTYPE_FLOAT16) {
        csinn_f16_to_float(dest, src);
    } else if (dest->dtype == CSINN_DTYPE_BFLOAT16 && src->dtype == CSINN_DTYPE_FLOAT32) {
        float_to_bf16(dest, src);
    } else if (dest->dtype == CSINN_DTYPE_FLOAT32 && src->dtype == CSINN_DTYPE_BFLOAT16) {
        bf16_to_float(dest, src);
    } else if (dest->dtype == src->dtype) {
        memcpy(dest->data, src->data, csinn_tensor_byte_size(src));
    } else {
        return CSINN_FALSE;
    }
    return CSINN_TRUE;
}

int csinn_tensor_data_convert(struct csinn_tensor *dest, struct csinn_tensor *src)
{
    if (src->layout != dest->layout) return CSINN_FALSE;

    switch (src->layout) {
        case CSINN_LAYOUT_NULL:
            return CSINN_TRUE;
        case CSINN_LAYOUT_N:
        case CSINN_LAYOUT_NC:
        case CSINN_LAYOUT_NCW:
        case CSINN_LAYOUT_NCHW:
        case CSINN_LAYOUT_NHWC:
        case CSINN_LAYOUT_NWC:
        case CSINN_LAYOUT_NCDHW:
        case CSINN_LAYOUT_NDHWC:
            return tensor_data_convert_activation(dest, src);
        case CSINN_LAYOUT_O:
        case CSINN_LAYOUT_OI:
        case CSINN_LAYOUT_OIW:
        case CSINN_LAYOUT_OWI:
        case CSINN_LAYOUT_OIHW:
        case CSINN_LAYOUT_OHWI:
        case CSINN_LAYOUT_OIDHW:
        case CSINN_LAYOUT_ODHWI:
        case CSINN_LAYOUT_O1HW:
        case CSINN_LAYOUT_1HWO:
            return tensor_data_convert_weight(dest, src);
        default:
            return CSINN_FALSE;
    }
}

static int layout_1HWO_to_1HW32O32(struct csinn_tensor *dest, struct csinn_tensor *src)
{
    if (src->dtype != CSINN_DTYPE_INT8 && src->dtype != CSINN_DTYPE_UINT8) {
        return CSINN_FALSE;
    }
    int a_len = 32;
    int b_len = a_len * src->dim[1] * src->dim[2];

    void *src_addr = src->data;
    void *dest_addr = dest->data;
    /* read in src order, write stride */
    for (int i = 0; i < src->dim[1] * src->dim[2]; i++) {
        for (int j = 0; j < src->dim[3] / a_len; j++) {
            dest_addr = dest->data + j * b_len + i * a_len;
            memcpy(dest_addr, src_addr, a_len);
            src_addr += a_len;
        }
        if (src->dim[3] % a_len) {
            dest_addr = dest->data + (src->dim[3] / a_len) * b_len + i * a_len;
            memcpy(dest_addr, src_addr, src->dim[3] % a_len);
            src_addr += src->dim[3] % a_len;
        }
    }
    return CSINN_TRUE;
}

static int layout_OI_to_O32I32(struct csinn_tensor *dest, struct csinn_tensor *src)
{
    if (src->dtype != CSINN_DTYPE_INT8 && src->dtype != CSINN_DTYPE_UINT8) {
        return CSINN_FALSE;
    }
    int a_len = 32;

    int8_t *src_addr = src->data;
    int8_t *dest_addr = dest->data;
    int src_idx = 0;
    int idx_base = 0;
    int dest_idx = 0;
    /* read src stride, write in order */
    for (int i = 0; i < src->dim[0] / a_len; i++) {
        idx_base = i * a_len * src->dim[1];
        dest_idx = idx_base;
        for (int j = 0; j < src->dim[1]; j++) {
            for (int k = 0; k < a_len; k++) {
                src_idx = idx_base + k * src->dim[1] + j;
                dest_addr[dest_idx] = src_addr[src_idx];
                dest_idx++;
            }
        }
    }
    idx_base = (src->dim[0] / a_len) * a_len * src->dim[1];
    dest_idx = idx_base;
    for (int j = 0; j < src->dim[1]; j++) {
        for (int k = 0; k < src->dim[0] % a_len; k++) {
            src_idx = idx_base + k * src->dim[1] + j;
            dest_idx = idx_base + k + a_len * j;
            dest_addr[dest_idx] = src_addr[src_idx];
        }
    }
}

static int layout_OHWI_to_O32HWI32(struct csinn_tensor *dest, struct csinn_tensor *src)
{
    if (src->dtype != CSINN_DTYPE_INT8 && src->dtype != CSINN_DTYPE_UINT8) {
        return CSINN_FALSE;
    }
    int a_len = 32;
    int b_len = src->dim[1] * src->dim[2] * src->dim[3];

    int8_t *src_addr = src->data;
    int8_t *dest_addr = dest->data;
    int src_idx = 0;
    int idx_base = 0;
    int dest_idx = 0;
    /* read src stride, write in order */
    for (int i = 0; i < src->dim[0] / a_len; i++) {
        idx_base = i * a_len * b_len;
        dest_idx = idx_base;
        for (int j = 0; j < b_len; j++) {
            for (int k = 0; k < a_len; k++) {
                src_idx = idx_base + k * b_len + j;
                dest_addr[dest_idx] = src_addr[src_idx];
                dest_idx++;
            }
        }
    }
    idx_base = (src->dim[0] / a_len) * a_len * b_len;
    dest_idx = idx_base;
    for (int j = 0; j < b_len; j++) {
        for (int k = 0; k < src->dim[0] % a_len; k++) {
            src_idx = idx_base + k * b_len + j;
            dest_idx = idx_base + k + a_len * j;
            dest_addr[dest_idx] = src_addr[src_idx];
        }
    }
}

int csinn_tensor_layout_convert(struct csinn_tensor *dest, struct csinn_tensor *src)
{
    int ret = CSINN_FALSE;
    if (src->layout == CSINN_LAYOUT_1HWO && dest->layout == CSINN_LAYOUT_1HW32O32) {
        ret = layout_1HWO_to_1HW32O32(dest, src);
    } else if (src->layout == CSINN_LAYOUT_OI && dest->layout == CSINN_LAYOUT_O32I32) {
        ret = layout_OI_to_O32I32(dest, src);
    } else if (src->layout == CSINN_LAYOUT_OHWI && dest->layout == CSINN_LAYOUT_O32HWI32) {
        ret = layout_OHWI_to_O32HWI32(dest, src);
    }

    return ret;
}

enum csinn_rmode_enum shl_get_run_mode(struct csinn_params_base *base)
{
    if (base->sess == NULL) {
        return CSINN_RM_LAYER;
    } else {
        return base->sess->base_run_mode;
    }
}

struct shl_cb_op_list *shl_cb_list_end(struct shl_cb_op_list *list)
{
    struct shl_cb_op_list *l = list;
    while (l->next) {
        l = l->next;
    }
    return l;
}

struct csinn_callback *shl_cb_list_match(struct shl_cb_op_list *list, enum csinn_dtype_enum dtype,
                                         enum csinn_op_enum op_name)
{
    struct csinn_callback *ret = NULL;
    struct shl_cb_op_list *l = list;
    while (l) {
        if (l->dtype == dtype && l->op_name == op_name) {
            ret = l->cb;
            break;
        }
        l = l->next;
    }
    return ret;
}

void *shl_get_init_cb(struct csinn_params_base *base)
{
    struct csinn_callback *cb = base->cb;
    if (base->sess && ((base->sess->base_run_mode == CSINN_RM_CPU_GRAPH) ||
                       (base->sess->base_run_mode == CSINN_RM_NPU_GRAPH))) {
        return NULL;
    }
    if (cb->init) {
        return cb->init;
    }

    return NULL;
}

/* establish graph or compute directly, get higher priority one */
void *shl_get_p0_cb(struct csinn_params_base *base)
{
    struct csinn_callback *cb = base->cb;
    if ((cb->est == NULL) && (cb->exec == NULL)) {
        shl_debug_error("OP have not register\n");
    }
    if (base->sess->base_run_mode == CSINN_RM_LAYER) {
        if (cb->exec) {
            return cb->exec;
        }
    } else {
        if (cb->est) {
            return cb->est;
        }
        if (cb->exec) {
            return cb->exec;
        }
    }

    return NULL;
}

#ifdef SHL_BUILD_RTOS
uint64_t shl_get_timespec() { return 0; }

void shl_print_time_interval(uint64_t start, uint64_t end, const char *msg) { return; }
#else
#define BILLION 1000000000
uint64_t shl_get_timespec()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)((uint64_t)ts.tv_nsec + (uint64_t)ts.tv_sec * BILLION);
}

void shl_print_time_interval(uint64_t start, uint64_t end, const char *msg)
{
    printf("Run %s time: %.5fms, FPS=%.2f\n", msg, ((double)(end - start)) / 1000000,
           1000000000.0 / ((double)(end - start)));
}
#endif

int csinn_version(char *vstr)
{
    int major = VERSION_MAJOR;
    int minor = VERSION_MINOR;
    int patch = VERSION_PATCH;
    if (vstr) {
        sprintf(vstr, "%d.%d.%d", major, minor, patch);
    }
    return (major << (VERSION_SHIFT * 2)) | (minor << VERSION_SHIFT) | patch;
}
