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

/* CSI-NN2 version 1.8.x */

#include "stdint.h"
#include "stdio.h"
#include "math.h"
#include "float.h"
#include "math_snr.h"
#include "test_utils.h"

int test_number = 0;
int failures = 0;

int done_testing(void)
{
    if (0 < failures) {
		printf("Failed %d tests\n", failures);
		exit(EXIT_FAILURE);
	} else {
		printf("All functions tested sucessfully\n");
		exit(EXIT_SUCCESS);
	}
	return failures;
}

void init_testsuite(const char* testname)
{
	printf("%s", testname);
	test_number = 0;
	failures = 0;
}

uint64_t kSignMask = 0x8000000000000000LL;
uint64_t kExponentMask = 0x7ff0000000000000LL;
int32_t  kExponentShift = 52;
int32_t  kExponentBias = 1023;
uint32_t kExponentIsBadNum = 0x7ff;
uint64_t kFractionMask = 0x000fffffffc00000LL;
uint32_t kFractionShift = 22;
uint32_t kFractionRoundingMask = 0x003fffff;
uint32_t kFractionRoundingThreshold = 0x00200000;

int *read_input_data_f32(char *path)
{
    FILE *fp = fopen(path, "rb");
    if(fp == NULL) {
        printf("Invalid input file: %s\n", path);
        return NULL;
    }

    int size;
    fread(&size, 4, 1, fp);

    int *buffer = malloc(size* sizeof(int));
    if(buffer == NULL) {
        printf("Malloc fail.\n");
        return NULL;
    }

    fread(buffer, 4, size, fp);

    fclose(fp);
    return buffer;
}

// calculate Kullback-Leibler divergence
float compute_kl(float *p, float *q, uint32_t size)
{
    float p_sum = 0.0f, q_sum = 0.0f, ret = 0.0f;

    for (int i = 0; i < size; i++) {
        p_sum += fabs(p[i]);
        q_sum += fabs(q[i]);
    }
    float p_tmp = 0.0f, q_tmp = 0.0f;
    for (int i = 0; i < size; i++) {
        p_tmp = fabs(p[i]);
        q_tmp = fabs(q[i]);
        p_tmp = p_tmp / p_sum;
        q_tmp = q_tmp / q_sum;
        if (fabs(p_tmp) <= 1e-9) {
            p_tmp += 1e-9;
        }
        if (fabs(q_tmp) <= 1e-9) {
            q_tmp += 1e-9;
        }
        if (!isnan(p_tmp) && !isnan(q_tmp)) {
            ret += p_tmp * log(p_tmp / q_tmp);
        }
    }
    return ret;
}

// calculate cosine similarity
float compute_cs(float *a, float *b, uint32_t size)
{
    float dot_sum = 0.0;
    float a_norm = 0.0;
    float b_norm = 0.0;
    float res = 0.0;

    for (int i = 0; i < size; i++) {
        dot_sum += (a[i] * b[i]);
        a_norm += (a[i] * a[i]);
        b_norm += (b[i] * b[i]);
    }
    res = dot_sum / (sqrt(a_norm * b_norm));
    return res;
}

void result_verify_int32(int *reference, int *output, int *input, float gap, int size, bool save)
{
    int i;
    float error, snr;

    for (i = 0; i < size; i++) {
        error = abs(reference[i] - output[i]);

        test_number++;
        if (error > gap) {
            failures++;
        }
#ifdef BASIC_DEBUG
        if (error > gap)
        {
          printf("i = %d :%d, %d, %d\n", i, reference[i], output[i], input[i]);
        }
#endif
    }
}

void result_verify_f32(float *reference, float *output, float *input, float gap, int size, bool save)
{
    int i;
    float error, snr;
    float max_error = 0;

    for (i = 0; i < size; i++) {
        if(isinf(reference[i]) && isinf(output[i]) || isnan(reference[i]) && isnan(output[i])){
            error = 0;
        } else {
            error = fabs(reference[i] - output[i]);
            if(error > gap) {
                error = fabs(reference[i] - output[i])/fabs(reference[i] + 1e-9);
            }
        }
        if(error > max_error) {
            max_error = error;
        }

        test_number++;
        if (error > gap) {
            failures++;
        }
#ifdef BASIC_DEBUG
        if (error > gap)
        {
          printf("i = %d :%.6f, %.6f, %.6f\n", i, reference[i], output[i], input[i]);
        }
#endif
    }
    printf("The max error is %f\n", max_error);

    float kl = compute_kl(output, reference, size);
    printf("The kl diver is %f.\n", kl);

    float cs = compute_cs(output, reference, size);
    printf("The cos sim is %f.\n", cs);
}


void result_verify_bool(bool *reference, bool *output, float *input, float gap, int size, bool save)
{
    int i;
    float error, snr;

    for (i = 0; i < size; i++) {
        error = fabs(reference[i] - output[i]);
        if(error > gap) {
            error = fabs(reference[i] - output[i])/fabs(reference[i] + 1e-9);
        }

        test_number++;
        if (error > gap) {
            failures++;
        }
#ifdef BASIC_DEBUG
        if (error > gap)
        {
          printf("i = %d, %d, %.6f\n", i, reference[i], output[i], input[i]);
        }
#endif
    }
}

void result_verify_8(float *reference, struct csi_tensor *output, int8_t *input, float gap, int size, bool save)
{
    int i;
    float error;
    float *output_tmp = malloc(size * sizeof(float));
    void *output_data = output->data;

    float max_error = 0;

    for (i = 0; i < size; i++) {
        if (output->dtype == CSINN_DTYPE_UINT8) {
            output_tmp[i] = csi_ref_dequantize_u8_to_f32(*((uint8_t *)output_data + i), output->qinfo);
        } else if (output->dtype == CSINN_DTYPE_INT8) {
            output_tmp[i] = csi_ref_dequantize_i8_to_f32(*((int8_t *)output_data + i), output->qinfo);
        }
        if(isinf(reference[i]) || isnan(reference[i])){
            error = 0;
        } else {
            error = fabs(reference[i] - output_tmp[i]);
            if(error > gap) {
                error = fabs(reference[i] - output_tmp[i])/fabs(reference[i] + 1e-9);
            }
        }
        if(error > max_error) {
            max_error = error;
        }

        test_number++;
        if (error > gap) {
            failures++;
        }
#ifdef BASIC_DEBUG
        if (error > gap)
        {
            printf("i = %d :%.6f, %.6f, %.6f\n", i, reference[i], output_tmp[i], input[i]);
        }
#endif
    }
    printf("The max error is %f\n", max_error);

    float kl = compute_kl(output_tmp, reference, size);
    printf("The kl diver is %f.\n", kl);

    float cs = compute_cs(output_tmp, reference, size);
    printf("The cos sim is %f.\n", cs);
    free(output_tmp);
}

/* https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/internal/quantization_util.cc */

int64_t integer_from_exp(double input, int* shift)
{

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

void quantize_multiplier(double double_multiplier, int32_t* quantized_multiplier, int* shift)
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

void get_scale_and_zp(float max_value, float min_value, float *scale, int *zp)
{
    int valid_range = 255;
    float scale_tmp, zp_tmp;

    max_value = max_value > 0.0f ? max_value : 0.0f;
    min_value = min_value < 0.0f ? min_value : 0.0f;

    scale_tmp = (max_value - min_value) / (float)valid_range;

    if (scale_tmp){
        zp_tmp = 0 - min_value / scale_tmp;
    } else {
        scale_tmp = 1;
        zp_tmp   = max_value;
    }
    zp_tmp = zp_tmp > 255 ? 255 : zp_tmp;
    zp_tmp = zp_tmp < 0 ? 0 : zp_tmp;

    *zp = (int)round(zp_tmp);
    *scale = scale_tmp;
}

void get_scale_and_zp_i8(float max_value, float min_value, float *scale, int *zp)
{
    int valid_range = 255;
    float scale_tmp, zp_tmp, max_tmp;

    if (fabs(max_value) >= fabs(min_value)){
        max_tmp = fabs(max_value);
    } else{
        max_tmp = fabs(min_value);
    }
    scale_tmp = 2 * max_tmp / (float)valid_range;
    zp_tmp = 0;

    if (scale_tmp == 0){
        scale_tmp = 1;
    }

    *zp = (int)zp_tmp;
    *scale = scale_tmp;
}

void find_min_max(float *input, float *max_value, float *min_value, int size)
{
    int i;
    float max_tmp = -FLT_MAX;
    float min_tmp = FLT_MAX;

    for(i = 0; i < size; i++) {
        if(input[i] != -FLT_MAX && input[i] != FLT_MAX) {
            if(input[i] > max_tmp) {
                max_tmp = input[i];
            }
            if(input[i] < min_tmp) {
                min_tmp = input[i];
            }
        }
    }

    *max_value = max_tmp;
    *min_value = min_tmp;
}

void get_quant_info(struct csi_tensor *tensor)
{
    float max, min, scale;
    int zp, quantized_multiplier, shift;
    if (tensor->qinfo == NULL) {
        tensor->qinfo = malloc(sizeof(struct csi_quant_info));
    }
    int size = csi_tensor_size(tensor);
    find_min_max(tensor->data, &max, &min, size);
    if (tensor->dtype == CSINN_DTYPE_UINT8) {
        get_scale_and_zp(max, min, &scale, &zp);
    } else if (tensor->dtype == CSINN_DTYPE_INT8) {
        get_scale_and_zp_i8(max, min, &scale, &zp);
    }
    quantize_multiplier(scale, &quantized_multiplier, &shift);
    tensor->qinfo->max = max;
    tensor->qinfo->min = min;
    tensor->qinfo->scale = scale;
    tensor->qinfo->zero_point = zp;
    tensor->qinfo->multiplier = quantized_multiplier;
    tensor->qinfo->shift = shift;
}
