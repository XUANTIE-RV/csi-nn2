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

/* CSI-NN2 version 1.12.x */

#include <time.h>

#include "csi_nn.h"
#include "csi_ref.h"

/* https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/internal/quantization_util.cc
 */
static int64_t integer_from_exp(double input, int *shift)
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

void csi_quantize_multiplier(double double_multiplier, int32_t *quantized_multiplier, int *shift)
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

void csi_statistical_mean_std(float *data, int sz)
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

void csi_get_top5(float *buf, uint32_t size, float *prob, uint32_t *class)
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

void csi_show_top5(struct csi_tensor *output, struct csi_session *sess)
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

    // #ifdef CSI_DEBUG
    csi_statistical_mean_std(output->data, size);
    // #endif

    csi_get_top5(output->data, size, prob, class);

    printf(" ============ top5: ===========\n");
    size = size > 5 ? 5 : size;
    for (i = 0; i < size; i++) {
        printf("%3d: %8.6f\n", class[i], prob[i]);
    }
}

int csi_tensor_size(struct csi_tensor *tensor)
{
    if (tensor->dim_count == 0) {
        return 0;
    }
    int size = 1;
    for (int i = 0; i < tensor->dim_count; i++) {
        size *= tensor->dim[i];
    }
    return size;
}

int csi_tensor_byte_size(struct csi_tensor *tensor)
{
    int size = csi_tensor_size(tensor);
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

struct csi_tensor *csi_alloc_tensor(struct csi_session *session)
{
    struct csi_tensor *ret = csi_mem_alloc(sizeof(struct csi_tensor));
    if (session != NULL) {
        ret->dtype = session->base_dtype;
        ret->layout = session->base_layout;
        ret->sess = session;
    }
    ret->quant_channel = 1;
    ret->qinfo = csi_mem_alloc(sizeof(struct csi_quant_info));
    return ret;
}

void csi_realloc_quant_info(struct csi_tensor *tensor, int quant_info_num)
{
    tensor->quant_channel = quant_info_num;
    tensor->qinfo = csi_mem_realloc(tensor->qinfo, quant_info_num * sizeof(struct csi_quant_info));
}

void csi_tensor_copy(struct csi_tensor *dest, struct csi_tensor *src)
{
    dest->data = src->data;
    dest->dtype = src->dtype;
    memcpy(dest->dim, src->dim, MAX_DIM * 4);
    dest->dim_count = src->dim_count;
    dest->name = src->name;
    dest->layout = src->layout;
    if (src->quant_channel != dest->quant_channel && src->quant_channel != 0) {
        csi_realloc_quant_info(dest, src->quant_channel);
    }
    memcpy(dest->qinfo, src->qinfo, sizeof(struct csi_quant_info) * src->quant_channel);
    dest->sess = src->sess;
    dest->is_const = src->is_const;
}

void csi_free_tensor(struct csi_tensor *tensor)
{
    if (tensor->qinfo != NULL) {
        csi_mem_free(tensor->qinfo);
    }
    csi_mem_free(tensor);
}

void *csi_alloc_params(int params_size, struct csi_session *session)
{
    struct csi_params_base *params = csi_mem_alloc(params_size);
    if (session != NULL) {
        params->api = session->base_api;
        params->layout = session->base_layout;
        params->run_mode = session->base_run_mode;
        params->sess = session;
    }
    return params;
}

void csi_free_params(void *params) { csi_mem_free(params); }

static float csi_int4_to_float_base(int8_t i, struct csi_tensor *t, int index)
{
    return ((float)i - t->qinfo[index].zero_point) * t->qinfo[index].scale;
}

static float csi_uint8_to_float_base(uint8_t i, struct csi_tensor *t, int index)
{
    return ((float)i - t->qinfo[index].zero_point) * t->qinfo[index].scale;
}

static float csi_int8_to_float_base(int8_t i, struct csi_tensor *t, int index)
{
    return ((float)i - t->qinfo[index].zero_point) * t->qinfo[index].scale;
}

static float csi_int16_to_float_base(int16_t i, struct csi_tensor *t, int index)
{
    return ((float)i - t->qinfo[index].zero_point) * t->qinfo[index].scale;
}

static float csi_int32_to_float_base(int32_t i, struct csi_tensor *t, int index)
{
    return (float)i * t->qinfo[index].scale;
}

static int8_t csi_float_to_int4_base(float i, struct csi_tensor *t, int index)
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

static uint8_t csi_float_to_uint8_base(float i, struct csi_tensor *t, int index)
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

static int8_t csi_float_to_int8_base(float i, struct csi_tensor *t, int index)
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

static int16_t csi_float_to_int16_base(float i, struct csi_tensor *t, int index)
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

/* Only for CSINN_LAYOUT_OHWI, HWI's size align */
static void csi_axis0_int4_to_float_alignHWI(struct csi_tensor *dest, struct csi_tensor *src,
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
                ret = csi_int4_to_float_base(src_tmp >> 4, src, i);
            } else {
                src_tmp = (src_data[in_index] & 0xf) << 4;
                ret = csi_int4_to_float_base(src_tmp >> 4, src, i);
            }
            dest_data[index] = ret;
        }
    }
}

/* Only for CSINN_LAYOUT_OHWI, HWI's size align */
static void csi_axis0_float_to_int4_alignHWI(struct csi_tensor *dest, struct csi_tensor *src,
                                             int inner_size)
{
    float *src_data = src->data;
    int8_t *dest_data = dest->data;
    int32_t q_size = dest->quant_channel;
    for (int i = 0; i < q_size; i++) {
        for (int j = 0; j < inner_size; j++) {
            int index = i * inner_size + j;
            int input_val = csi_float_to_int4_base(src_data[index], dest, i);
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

static void csi_nchw_int4_to_float(struct csi_tensor *dest, struct csi_tensor *src, int n,
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
                ret = csi_int4_to_float_base(src_tmp >> 4, src, i);
            } else {
                src_tmp = (src_data[in_index] & 0xf) << 4;
                ret = csi_int4_to_float_base(src_tmp >> 4, src, i);
            }
            dest_data[index] = ret;
        }
    }
}

static void csi_nhwc_int4_to_float(struct csi_tensor *dest, struct csi_tensor *src, int n,
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
                ret = csi_int4_to_float_base(src_tmp >> 4, src, i);
            } else {
                src_tmp = (src_data[in_index] & 0xf) << 4;
                ret = csi_int4_to_float_base(src_tmp >> 4, src, i);
            }
            dest_data[index] = ret;
        }
    }
}

static void csi_nchw_float_to_int4(struct csi_tensor *dest, struct csi_tensor *src, int n,
                                   int inner_size)
{
    float *src_data = src->data;
    int8_t *dest_data = dest->data;
    int32_t q_size = dest->quant_channel;
    for (int i = 0; i < q_size; i++) {
        for (int j = 0; j < inner_size; j++) {
            int index = n * q_size * inner_size + i * inner_size + j;
            int input_val = csi_float_to_int4_base(src_data[index], dest, i);
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

static void csi_nhwc_float_to_int4(struct csi_tensor *dest, struct csi_tensor *src, int n,
                                   int inner_size)
{
    float *src_data = src->data;
    int8_t *dest_data = dest->data;
    int32_t q_size = dest->quant_channel;
    for (int j = 0; j < inner_size; j++) {
        for (int i = 0; i < q_size; i++) {
            int index = n * q_size * inner_size + j * q_size + i;
            int input_val = csi_float_to_int4_base(src_data[index], dest, i);
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

static void csi_nchw_uint8_to_float(struct csi_tensor *dest, struct csi_tensor *src, int n,
                                    int inner_size)
{
    uint8_t *src_data = src->data;
    float *dest_data = dest->data;
    int32_t q_size = src->quant_channel;
    for (int i = 0; i < q_size; i++) {
        for (int j = 0; j < inner_size; j++) {
            int index = n * q_size * inner_size + i * inner_size + j;
            dest_data[index] = csi_uint8_to_float_base(src_data[index], src, i);
        }
    }
}

static void csi_nhwc_uint8_to_float(struct csi_tensor *dest, struct csi_tensor *src, int n,
                                    int inner_size)
{
    uint8_t *src_data = src->data;
    float *dest_data = dest->data;
    int32_t q_size = src->quant_channel;
    for (int j = 0; j < inner_size; j++) {
        for (int i = 0; i < q_size; i++) {
            int index = n * q_size * inner_size + j * q_size + i;
            dest_data[index] = csi_uint8_to_float_base(src_data[index], src, i);
        }
    }
}

static void csi_nchw_float_to_uint8(struct csi_tensor *dest, struct csi_tensor *src, int n,
                                    int inner_size)
{
    float *src_data = src->data;
    uint8_t *dest_data = dest->data;
    int32_t q_size = dest->quant_channel;
    for (int i = 0; i < q_size; i++) {
        for (int j = 0; j < inner_size; j++) {
            int index = n * q_size * inner_size + i * inner_size + j;
            dest_data[index] = csi_float_to_uint8_base(src_data[index], dest, i);
        }
    }
}
static void csi_nhwc_float_to_uint8(struct csi_tensor *dest, struct csi_tensor *src, int n,
                                    int inner_size)
{
    float *src_data = src->data;
    uint8_t *dest_data = dest->data;
    int32_t q_size = dest->quant_channel;
    for (int j = 0; j < inner_size; j++) {
        for (int i = 0; i < q_size; i++) {
            int index = n * q_size * inner_size + j * q_size + i;
            dest_data[index] = csi_float_to_uint8_base(src_data[index], dest, i);
        }
    }
}

static void csi_nchw_int8_to_float(struct csi_tensor *dest, struct csi_tensor *src, int n,
                                   int inner_size)
{
    int8_t *src_data = src->data;
    float *dest_data = dest->data;
    int32_t q_size = src->quant_channel;
    for (int i = 0; i < q_size; i++) {
        for (int j = 0; j < inner_size; j++) {
            int index = n * q_size * inner_size + i * inner_size + j;
            dest_data[index] = csi_int8_to_float_base(src_data[index], src, i);
        }
    }
}
static void csi_nhwc_int8_to_float(struct csi_tensor *dest, struct csi_tensor *src, int n,
                                   int inner_size)
{
    int8_t *src_data = src->data;
    float *dest_data = dest->data;
    int32_t q_size = src->quant_channel;
    for (int j = 0; j < inner_size; j++) {
        for (int i = 0; i < q_size; i++) {
            int index = n * q_size * inner_size + j * q_size + i;
            dest_data[index] = csi_int8_to_float_base(src_data[index], src, i);
        }
    }
}

static void csi_nchw_float_to_int8(struct csi_tensor *dest, struct csi_tensor *src, int n,
                                   int inner_size)
{
    float *src_data = src->data;
    int8_t *dest_data = dest->data;
    int32_t q_size = dest->quant_channel;
    for (int i = 0; i < q_size; i++) {
        for (int j = 0; j < inner_size; j++) {
            int index = n * q_size * inner_size + i * inner_size + j;
            dest_data[index] = csi_float_to_int8_base(src_data[index], dest, i);
        }
    }
}

static void csi_nhwc_float_to_int8(struct csi_tensor *dest, struct csi_tensor *src, int n,
                                   int inner_size)
{
    float *src_data = src->data;
    int8_t *dest_data = dest->data;
    int32_t q_size = dest->quant_channel;
    for (int j = 0; j < inner_size; j++) {
        for (int i = 0; i < q_size; i++) {
            int index = n * q_size * inner_size + j * q_size + i;
            dest_data[index] = csi_float_to_int8_base(src_data[index], dest, i);
        }
    }
}

static void csi_nchw_int16_to_float(struct csi_tensor *dest, struct csi_tensor *src, int n,
                                    int inner_size)
{
    int16_t *src_data = src->data;
    float *dest_data = dest->data;
    int32_t q_size = src->quant_channel;
    for (int i = 0; i < q_size; i++) {
        for (int j = 0; j < inner_size; j++) {
            int index = n * q_size * inner_size + i * inner_size + j;
            dest_data[index] = csi_int16_to_float_base(src_data[index], src, i);
        }
    }
}

static void csi_nhwc_int16_to_float(struct csi_tensor *dest, struct csi_tensor *src, int n,
                                    int inner_size)
{
    int16_t *src_data = src->data;
    float *dest_data = dest->data;
    int32_t q_size = src->quant_channel;
    for (int j = 0; j < inner_size; j++) {
        for (int i = 0; i < q_size; i++) {
            int index = n * q_size * inner_size + j * q_size + i;
            dest_data[index] = csi_int16_to_float_base(src_data[index], src, i);
        }
    }
}

static void csi_nchw_float_to_int16(struct csi_tensor *dest, struct csi_tensor *src, int n,
                                    int inner_size)
{
    float *src_data = src->data;
    int16_t *dest_data = dest->data;
    int32_t q_size = dest->quant_channel;
    for (int i = 0; i < q_size; i++) {
        for (int j = 0; j < inner_size; j++) {
            int index = n * q_size * inner_size + i * inner_size + j;
            dest_data[index] = csi_float_to_int16_base(src_data[index], dest, i);
        }
    }
}

static void csi_nhwc_float_to_int16(struct csi_tensor *dest, struct csi_tensor *src, int n,
                                    int inner_size)
{
    float *src_data = src->data;
    int16_t *dest_data = dest->data;
    int32_t q_size = dest->quant_channel;
    for (int j = 0; j < inner_size; j++) {
        for (int i = 0; i < q_size; i++) {
            int index = n * q_size * inner_size + j * q_size + i;
            dest_data[index] = csi_float_to_int16_base(src_data[index], dest, i);
        }
    }
}

static void csi_nchw_int32_to_float(struct csi_tensor *dest, struct csi_tensor *src, int n,
                                    int inner_size)
{
    int32_t *src_data = src->data;
    float *dest_data = dest->data;
    int32_t q_size = src->quant_channel;
    for (int i = 0; i < q_size; i++) {
        for (int j = 0; j < inner_size; j++) {
            int index = n * q_size * inner_size + i * inner_size + j;
            dest_data[index] = csi_int32_to_float_base(src_data[index], src, i);
        }
    }
}

static void csi_nhwc_int32_to_float(struct csi_tensor *dest, struct csi_tensor *src, int n,
                                    int inner_size)
{
    int32_t *src_data = src->data;
    float *dest_data = dest->data;
    int32_t q_size = src->quant_channel;
    for (int j = 0; j < inner_size; j++) {
        for (int i = 0; i < q_size; i++) {
            int index = n * q_size * inner_size + j * q_size + i;
            dest_data[index] = csi_int32_to_float_base(src_data[index], src, i);
        }
    }
}

static void csi_f16_to_float(struct csi_tensor *dest, struct csi_tensor *src)
{
    int16_t *src_data = src->data;
    float *dest_data = dest->data;
    int32_t size = csi_tensor_size(src);
    for (int j = 0; j < size; j++) {
        dest_data[j] = csi_ref_float16_to_float32(src_data[j]);
    }
}

static void csi_float_to_f16(struct csi_tensor *dest, struct csi_tensor *src)
{
    float *src_data = src->data;
    int16_t *dest_data = dest->data;
    int32_t size = csi_tensor_size(src);
    for (int i = 0; i < size; i++) {
        dest_data[i] = csi_ref_float32_to_float16(src_data[i]);
    }
}

static void csi_bf16_to_float(struct csi_tensor *dest, struct csi_tensor *src)
{
    int16_t *src_data = src->data;
    float *dest_data = dest->data;
    int32_t size = csi_tensor_size(src);
    for (int j = 0; j < size; j++) {
        dest_data[j] = csi_ref_bfloat16_to_float32(src_data[j]);
    }
}

static void csi_float_to_bf16(struct csi_tensor *dest, struct csi_tensor *src)
{
    float *src_data = src->data;
    int16_t *dest_data = dest->data;
    int32_t size = csi_tensor_size(src);
    for (int i = 0; i < size; i++) {
        dest_data[i] = csi_ref_float32_to_bfloat16(src_data[i]);
    }
}

int csi_tensor_data_convert_weight(struct csi_tensor *dest, struct csi_tensor *src)
{
    int size = csi_tensor_size(src);
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
                csi_nchw_int4_to_float(dest, src, 0, inner_size);
                break;
            case CSINN_LAYOUT_OHWI:
                csi_axis0_int4_to_float_alignHWI(dest, src, inner_size);
                break;
            case CSINN_LAYOUT_1HWO:
                csi_nhwc_int4_to_float(dest, src, 0, inner_size);
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
                csi_nchw_float_to_int4(dest, src, 0, inner_size);
                break;
            case CSINN_LAYOUT_OHWI:
                csi_axis0_float_to_int4_alignHWI(dest, src, inner_size);
            case CSINN_LAYOUT_1HWO:
                csi_nhwc_float_to_int4(dest, src, 0, inner_size);
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
                csi_nchw_uint8_to_float(dest, src, 0, inner_size);
                break;
            case CSINN_LAYOUT_1HWO:
                csi_nhwc_uint8_to_float(dest, src, 0, inner_size);
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
                csi_nchw_float_to_uint8(dest, src, 0, inner_size);
                break;
            case CSINN_LAYOUT_1HWO:
                csi_nhwc_float_to_uint8(dest, src, 0, inner_size);
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
                csi_nchw_int8_to_float(dest, src, 0, inner_size);
                break;
            case CSINN_LAYOUT_1HWO:
                csi_nhwc_int8_to_float(dest, src, 0, inner_size);
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
                csi_nchw_float_to_int8(dest, src, 0, inner_size);
                break;
            case CSINN_LAYOUT_1HWO:
                csi_nhwc_float_to_int8(dest, src, 0, inner_size);
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
                csi_nchw_int16_to_float(dest, src, 0, inner_size);
                break;
            case CSINN_LAYOUT_1HWO:
                csi_nhwc_int16_to_float(dest, src, 0, inner_size);
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
                csi_nchw_float_to_int16(dest, src, 0, inner_size);
                break;
            case CSINN_LAYOUT_1HWO:
                csi_nhwc_float_to_int16(dest, src, 0, inner_size);
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
                csi_nchw_int32_to_float(dest, src, 0, inner_size);
                break;
            case CSINN_LAYOUT_1HWO:
                csi_nhwc_int32_to_float(dest, src, 0, inner_size);
                break;
            default:
                break;
        }
    } else if (dest->dtype == CSINN_DTYPE_FLOAT16 && src->dtype == CSINN_DTYPE_FLOAT32) {
        csi_float_to_f16(dest, src);
    } else if (dest->dtype == CSINN_DTYPE_FLOAT32 && src->dtype == CSINN_DTYPE_FLOAT16) {
        csi_f16_to_float(dest, src);
    } else if (dest->dtype == CSINN_DTYPE_BFLOAT16 && src->dtype == CSINN_DTYPE_FLOAT32) {
        csi_float_to_bf16(dest, src);
    } else if (dest->dtype == CSINN_DTYPE_FLOAT32 && src->dtype == CSINN_DTYPE_BFLOAT16) {
        csi_bf16_to_float(dest, src);
    } else if (dest->dtype == src->dtype) {
        memcpy(dest->data, src->data, csi_tensor_byte_size(src));
    } else {
        return CSINN_FALSE;
    }
    return CSINN_TRUE;
}

int csi_tensor_data_convert_activation(struct csi_tensor *dest, struct csi_tensor *src)
{
    int size = csi_tensor_size(src);
    int32_t q_size = src->quant_channel != 0 ? src->quant_channel : dest->quant_channel;
    if (q_size == 0) {
        q_size = 1;
    }
    int inner_size = size / q_size / src->dim[0];
    if (dest->dtype == CSINN_DTYPE_FLOAT32 && src->dtype == CSINN_DTYPE_INT4) {
        for (int n = 0; n < src->dim[0]; n++) {
            if (src->layout >= CSINN_LAYOUT_N && src->layout <= CSINN_LAYOUT_NCDHW) {
                csi_nchw_int4_to_float(dest, src, n, inner_size);
            } else if (src->layout >= CSINN_LAYOUT_NWC && src->layout <= CSINN_LAYOUT_NDHWC) {
                csi_nhwc_int4_to_float(dest, src, n, inner_size);
            }
        }
    } else if (dest->dtype == CSINN_DTYPE_INT4 && src->dtype == CSINN_DTYPE_FLOAT32) {
        for (int n = 0; n < src->dim[0]; n++) {
            if (src->layout >= CSINN_LAYOUT_N && src->layout <= CSINN_LAYOUT_NCDHW) {
                csi_nchw_float_to_int4(dest, src, n, inner_size);
            } else if (src->layout >= CSINN_LAYOUT_NWC && src->layout <= CSINN_LAYOUT_NDHWC) {
                csi_nhwc_float_to_int4(dest, src, n, inner_size);
            }
        }
    } else if (dest->dtype == CSINN_DTYPE_FLOAT32 && src->dtype == CSINN_DTYPE_UINT8) {
        for (int n = 0; n < src->dim[0]; n++) {
            if (src->layout >= CSINN_LAYOUT_N && src->layout <= CSINN_LAYOUT_NCDHW) {
                csi_nchw_uint8_to_float(dest, src, n, inner_size);
            } else if (src->layout >= CSINN_LAYOUT_NWC && src->layout <= CSINN_LAYOUT_NDHWC) {
                csi_nhwc_uint8_to_float(dest, src, n, inner_size);
            }
        }
    } else if (dest->dtype == CSINN_DTYPE_UINT8 && src->dtype == CSINN_DTYPE_FLOAT32) {
        for (int n = 0; n < src->dim[0]; n++) {
            if (src->layout >= CSINN_LAYOUT_N && src->layout <= CSINN_LAYOUT_NCDHW) {
                csi_nchw_float_to_uint8(dest, src, n, inner_size);
            } else if (src->layout >= CSINN_LAYOUT_NWC && src->layout <= CSINN_LAYOUT_NDHWC) {
                csi_nhwc_float_to_uint8(dest, src, n, inner_size);
            }
        }
    } else if (dest->dtype == CSINN_DTYPE_FLOAT32 && src->dtype == CSINN_DTYPE_INT8) {
        for (int n = 0; n < src->dim[0]; n++) {
            if (src->layout >= CSINN_LAYOUT_N && src->layout <= CSINN_LAYOUT_NCDHW) {
                csi_nchw_int8_to_float(dest, src, n, inner_size);
            } else if (src->layout >= CSINN_LAYOUT_NWC && src->layout <= CSINN_LAYOUT_NDHWC) {
                csi_nhwc_int8_to_float(dest, src, n, inner_size);
            }
        }
    } else if (dest->dtype == CSINN_DTYPE_INT8 && src->dtype == CSINN_DTYPE_FLOAT32) {
        for (int n = 0; n < src->dim[0]; n++) {
            if (src->layout >= CSINN_LAYOUT_N && src->layout <= CSINN_LAYOUT_NCDHW) {
                csi_nchw_float_to_int8(dest, src, n, inner_size);
            } else if (src->layout >= CSINN_LAYOUT_NWC && src->layout <= CSINN_LAYOUT_NDHWC) {
                csi_nhwc_float_to_int8(dest, src, n, inner_size);
            }
        }
    } else if (dest->dtype == CSINN_DTYPE_FLOAT32 && src->dtype == CSINN_DTYPE_INT16) {
        for (int n = 0; n < src->dim[0]; n++) {
            if (src->layout >= CSINN_LAYOUT_N && src->layout <= CSINN_LAYOUT_NCDHW) {
                csi_nchw_int16_to_float(dest, src, n, inner_size);
            } else if (src->layout >= CSINN_LAYOUT_NWC && src->layout <= CSINN_LAYOUT_NDHWC) {
                csi_nhwc_int16_to_float(dest, src, n, inner_size);
            }
        }
    } else if (dest->dtype == CSINN_DTYPE_INT16 && src->dtype == CSINN_DTYPE_FLOAT32) {
        for (int n = 0; n < src->dim[0]; n++) {
            if (src->layout >= CSINN_LAYOUT_N && src->layout <= CSINN_LAYOUT_NCDHW) {
                csi_nchw_float_to_int16(dest, src, n, inner_size);
            } else if (src->layout >= CSINN_LAYOUT_NWC && src->layout <= CSINN_LAYOUT_NDHWC) {
                csi_nhwc_float_to_int16(dest, src, n, inner_size);
            }
        }
    } else if (dest->dtype == CSINN_DTYPE_FLOAT32 && src->dtype == CSINN_DTYPE_INT32) {
        for (int n = 0; n < src->dim[0]; n++) {
            if (src->layout >= CSINN_LAYOUT_N && src->layout <= CSINN_LAYOUT_NCDHW) {
                csi_nchw_int32_to_float(dest, src, n, inner_size);
            } else if (src->layout >= CSINN_LAYOUT_NWC && src->layout <= CSINN_LAYOUT_NDHWC) {
                csi_nhwc_int32_to_float(dest, src, n, inner_size);
            }
        }
    } else if (dest->dtype == CSINN_DTYPE_FLOAT16 && src->dtype == CSINN_DTYPE_FLOAT32) {
        csi_float_to_f16(dest, src);
    } else if (dest->dtype == CSINN_DTYPE_FLOAT32 && src->dtype == CSINN_DTYPE_FLOAT16) {
        csi_f16_to_float(dest, src);
    } else if (dest->dtype == CSINN_DTYPE_BFLOAT16 && src->dtype == CSINN_DTYPE_FLOAT32) {
        csi_float_to_bf16(dest, src);
    } else if (dest->dtype == CSINN_DTYPE_FLOAT32 && src->dtype == CSINN_DTYPE_BFLOAT16) {
        csi_bf16_to_float(dest, src);
    } else if (dest->dtype == src->dtype) {
        memcpy(dest->data, src->data, csi_tensor_byte_size(src));
    } else {
        return CSINN_FALSE;
    }
    return CSINN_TRUE;
}

int csi_tensor_data_convert(struct csi_tensor *dest, struct csi_tensor *src)
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
            return csi_tensor_data_convert_activation(dest, src);
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
            return csi_tensor_data_convert_weight(dest, src);
        default:
            return CSINN_FALSE;
    }
}

#ifdef CSI_BUILD_RTOS
uint64_t csi_get_timespec() { return 0; }

void csi_print_time_interval(uint64_t start, uint64_t end, const char *msg) { return; }
#else
#define BILLION 1000000000
uint64_t csi_get_timespec()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)((uint64_t)ts.tv_nsec + (uint64_t)ts.tv_sec * BILLION);
}

void csi_print_time_interval(uint64_t start, uint64_t end, const char *msg)
{
    printf("Run %s time: %.5fms, FPS=%.2f\n", msg, ((double)(end - start)) / 1000000,
           1000000000.0 / ((double)(end - start)));
}
#endif
