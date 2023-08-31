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

#include "rvv/rvv.h"

/*************************************************************************************
 * reorder kernel_data inplace, means the origin kernel_data be destoried.
 * The reason to do this is that the packaging process must not consume more memory.
 **************************************************************************************/
void shl_rvv_conv1d_im2col_gemm_reorder_kernel_fp16(struct csinn_tensor *kernel,
                                                    struct csinn_conv1d_params *params)
{
    __fp16 *kernel_data = (__fp16 *)kernel->data;
    int group = params->group;

    int m = kernel->dim[0] / group;
    int k = kernel->dim[1] * kernel->dim[2];

    __fp16 *pa_reorder = (__fp16 *)shl_mem_alloc(group * m * k * sizeof(__fp16));
    for (int g = 0; g < group; g++) {
        shl_rvv_reorder_kernel_n8_fp16(kernel_data + g * m * k, pa_reorder + g * m * k, m, k, k);
    }
    memcpy(kernel_data, pa_reorder, group * m * k * sizeof(__fp16));
    shl_mem_free(pa_reorder);
}

/*************************************************************************************
 * reorder kernel_data inplace, means the origin kernel_data be destoried.
 * The reason to do this is that the packaging process must not consume more memory.
 **************************************************************************************/
void shl_rvv_conv1d_im2col_gemm_reorder_kernel_fp16_w_int8(struct csinn_tensor *kernel,
                                                           struct csinn_conv1d_params *params)
{
    int8_t *kernel_data = (int8_t *)kernel->data;
    int group = params->group;

    int m = kernel->dim[0] / group;
    int k = kernel->dim[1] * kernel->dim[2];

    int8_t *pa_reorder = (int8_t *)shl_mem_alloc(group * m * k * sizeof(int8_t));
    for (int g = 0; g < group; g++) {
        shl_rvv_reorder_kernel_n8_fp16_w_int8(kernel_data + g * m * k, pa_reorder + g * m * k, m, k,
                                              k);
    }
    memcpy(kernel_data, pa_reorder, group * m * k * sizeof(int8_t));
    shl_mem_free(pa_reorder);
}

/*************************************************************************************
 * Per-channel dequantize int8 -> fp16
 ************************************************************************************/
void shl_rvv_conv1d_im2col_gemm_dequantize_per_channel_i8_to_f16(struct csinn_tensor *kernel,
                                                                 struct csinn_conv1d_params *params,
                                                                 __fp16 *kernel_fp16)
{
    int8_t *kernel_int8 = (int8_t *)kernel->data;
    const int group = params->group;
    const int m = kernel->dim[0] / group;
    const int k = kernel->dim[1] * kernel->dim[2];
    for (int g = 0; g < group; g++) {
        int8_t *ksrc = kernel_int8 + g * m * k;
        __fp16 *kdst = kernel_fp16 + g * m * k;
        int i = 0;
        int vl = 8;
        for (; i + 7 < m; i += 8) {
            int oc = g * m + i;
            vint32m4_t _z32 = vlse32_v_i32m4(&(kernel->qinfo[oc].zero_point),
                                             sizeof(struct csinn_quant_info), vl);
            vint16m2_t _z16 = vnclip_wx_i16m2(_z32, 0, vl);
            vint8m1_t _z = vnclip_wx_i8m1(_z16, 0, vl);
            vfloat32m4_t _s32 =
                vlse32_v_f32m4(&(kernel->qinfo[oc].scale), sizeof(struct csinn_quant_info), vl);
            vfloat16m2_t _s = vfncvt_f_f_w_f16m2(_s32, vl);
            for (int j = 0; j < k; j++) {
                vint8m1_t _i8 = vle8_v_i8m1(ksrc, vl);
                vfloat16m2_t _f16 = shl_rvv_vdeq_vv_f16m2(_i8, _z, _s, vl);
                vse16_v_f16m2(kdst, _f16, vl);
                ksrc += vl;
                kdst += vl;
            }
        }
        vl = 4;
        for (; i + 3 < m; i += 4) {
            int oc = g * m + i;
            vint32m4_t _z32 = vlse32_v_i32m4(&(kernel->qinfo[oc].zero_point),
                                             sizeof(struct csinn_quant_info), vl);
            vint16m2_t _z16 = vnclip_wx_i16m2(_z32, 0, vl);
            vint8m1_t _z = vnclip_wx_i8m1(_z16, 0, vl);
            vfloat32m4_t _s32 =
                vlse32_v_f32m4(&(kernel->qinfo[oc].scale), sizeof(struct csinn_quant_info), vl);
            vfloat16m2_t _s = vfncvt_f_f_w_f16m2(_s32, vl);
            for (int j = 0; j < k; j++) {
                vint8m1_t _i8 = vle8_v_i8m1(ksrc, vl);
                vfloat16m2_t _f16 = shl_rvv_vdeq_vv_f16m2(_i8, _z, _s, vl);
                vse16_v_f16m2(kdst, _f16, vl);
                ksrc += vl;
                kdst += vl;
            }
        }
        vl = 2;
        for (; i + 1 < m; i += 2) {
            int oc = g * m + i;
            vint32m4_t _z32 = vlse32_v_i32m4(&(kernel->qinfo[oc].zero_point),
                                             sizeof(struct csinn_quant_info), vl);
            vint16m2_t _z16 = vnclip_wx_i16m2(_z32, 0, vl);
            vint8m1_t _z = vnclip_wx_i8m1(_z16, 0, vl);
            vfloat32m4_t _s32 =
                vlse32_v_f32m4(&(kernel->qinfo[oc].scale), sizeof(struct csinn_quant_info), vl);
            vfloat16m2_t _s = vfncvt_f_f_w_f16m2(_s32, vl);
            for (int j = 0; j < k; j++) {
                vint8m1_t _i8 = vle8_v_i8m1(ksrc, vl);
                vfloat16m2_t _f16 = shl_rvv_vdeq_vv_f16m2(_i8, _z, _s, vl);
                vse16_v_f16m2(kdst, _f16, vl);
                ksrc += vl;
                kdst += vl;
            }
        }
        vl = 1;
        for (; i < m; i++) {
            int oc = g * m + i;
            int32_t zp = kernel->qinfo[oc].zero_point;
            float scale = kernel->qinfo[oc].scale;
            shl_rvv_dequantize_i8_to_f16(ksrc, kdst, k, zp, scale);
        }
    }
}

int shl_rvv_conv1d_im2col_gemm_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                    struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                    struct csinn_conv1d_params *params)
{
    if (input->layout == CSINN_LAYOUT_NC1WC0) {
        shl_rvv_tensor_nc1xc0_to_ndarray_replace_fp16(input);
    }
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    __fp16 *kernel_data = NULL;
    __fp16 *bias_data = (__fp16 *)bias->data;

    int32_t group = params->group;
    int32_t batch = input->dim[0];
    int32_t in_ch = input->dim[1];
    int32_t in_width = input->dim[2];

    int32_t out_ch = kernel->dim[0];
    int32_t out_width = output->dim[2];

    int32_t kernel_w = kernel->dim[2];
    int32_t stride_w = params->stride_width;
    int32_t pad_left = params->pad_left;
    int32_t dilation_w = params->dilation_width;

    int32_t m = out_ch / group;
    int32_t k = in_ch / group * kernel_w;
    int32_t n = out_width;

    __fp16 *kernel_fp16 = NULL;
    if (kernel->is_const && kernel->dtype == CSINN_DTYPE_INT8) {
        int size = csinn_tensor_size(kernel);
        kernel_fp16 = (__fp16 *)shl_mem_alloc(size * sizeof(__fp16));
        if (kernel->quant_channel > 1) {
            shl_rvv_conv1d_im2col_gemm_dequantize_per_channel_i8_to_f16(kernel, params,
                                                                        kernel_fp16);
        } else {
            int8_t *kernel_int8 = (int8_t *)kernel->data;
            int32_t zp = kernel->qinfo->zero_point;
            float scale = kernel->qinfo->scale;
            shl_rvv_dequantize_i8_to_f16(kernel_int8, kernel_fp16, size, zp, scale);
        }
        kernel_data = kernel_fp16;
    } else if (kernel->dtype == CSINN_DTYPE_FLOAT16) {
        kernel_data = (__fp16 *)kernel->data;
    } else {
        shl_debug_error("kernel unsupport dtype: %d\n", kernel->dtype);
        return CSINN_FALSE;
    }

    __fp16 *im2col_data = (__fp16 *)shl_mem_alloc(k * n * sizeof(__fp16));
    __fp16 *pb_reorder = (__fp16 *)shl_mem_alloc(k * n * sizeof(__fp16));

    for (int i = 0; i < batch; i++) {
        for (int g = 0; g < group; g++) {
            // im2col
            __fp16 *data_col = im2col_data;
            __fp16 *channel_data = input_data;
            for (int c = 0; c < in_ch / group; c++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    int in_col = -pad_left + kw * dilation_w;
                    for (int ow1 = 0; ow1 < out_width; ow1++) {
                        if (in_col < in_width && in_col >= 0) {
                            *data_col++ = channel_data[in_col];
                        } else {
                            *data_col++ = 0.0f;
                        }
                        in_col += stride_w;
                    }
                }
                channel_data += in_width;
            }

            __fp16 *pa = kernel_data + g * m * k;
            __fp16 *pb = pb_reorder;
            __fp16 *pc = output_data;

            // pack
            shl_rvv_reorder_input_z16_fp16(im2col_data, pb, k, n, n);
            // GEMM
            shl_rvv_gemm_8x16_fp16(pc, pa, pb, bias_data + g * m, m, k, n, n);
            input_data += in_ch / group * in_width;
            output_data += m * n;
        }
    }
    shl_mem_free(pb_reorder);
    shl_mem_free(im2col_data);
    if (kernel->is_const && kernel->dtype == CSINN_DTYPE_INT8) {
        shl_mem_free(kernel_fp16);
        return CSINN_TRUE;
    }
    // requantize
    shl_rvv_sidcso_op_requantize_fp16(input, output, kernel);
    return CSINN_TRUE;
}