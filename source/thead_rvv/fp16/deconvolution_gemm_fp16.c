/*
 * Copyright (C) 2016-2023 C-SKY Microsystems Co., Ltd. All rights reserved.
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

static void transpose_10_fp16(__fp16 *src, __fp16 *dst, int inner_size, int outer_size)
{
    for (int i = 0; i < outer_size; i++) {
        int size = inner_size;
        __fp16 *d_ptr = dst + i;
        while (size > 0) {
            int vl = vsetvl_e16m4(size);
            vfloat16m4_t _in = vle16_v_f16m4(src, vl);
            src += vl;
            vsse16_v_f16m4(d_ptr, outer_size * sizeof(__fp16), _in, vl);
            d_ptr += vl * outer_size;
            size -= vl;
        }
    }
}

static void transpose_10_int8(int8_t *src, int8_t *dst, int inner_size, int outer_size)
{
    for (int i = 0; i < outer_size; i++) {
        int size = inner_size;
        int8_t *d_ptr = dst + i;
        while (size > 0) {
            int vl = vsetvl_e8m4(size);
            vint8m4_t _in = vle8_v_i8m4(src, vl);
            src += vl;
            vsse8_v_i8m4(d_ptr, outer_size * sizeof(int8_t), _in, vl);
            d_ptr += vl * outer_size;
            size -= vl;
        }
    }
}

// Kernel:[IC,OC,KH,KW] --> [OC,KH,KW,IC]
void shl_rvv_deconv2d_gemm_col2im_reorder_kernel_fp16(struct csinn_tensor *kernel,
                                                      struct csinn_conv2d_params *params)
{
    __fp16 *kernel_data = (__fp16 *)kernel->data;
    __fp16 *data_buf = shl_mem_alloc(kernel->dim[0] * kernel->dim[1] * kernel->dim[2] *
                                     kernel->dim[3] * sizeof(__fp16));

    int inner_size = kernel->dim[1] * kernel->dim[2] * kernel->dim[3];
    int outer_size = kernel->dim[0];

    transpose_10_fp16(kernel_data, data_buf, inner_size, outer_size);

    int group = params->group;

    int k = kernel->dim[0];
    int m = kernel->dim[1] * kernel->dim[2] * kernel->dim[3] / group;
    for (int g = 0; g < group; g++) {
        shl_rvv_reorder_kernel_n8_fp16(data_buf + g * m * k, kernel_data + g * m * k, m, k, k);
    }
    shl_mem_free(data_buf);
}

void shl_rvv_deconv2d_gemm_col2im_reorder_kernel_fp16_w_int8(struct csinn_tensor *kernel,
                                                             struct csinn_conv2d_params *params)
{
    int8_t *kernel_data = (int8_t *)kernel->data;
    int8_t *data_buf = shl_mem_alloc(kernel->dim[0] * kernel->dim[1] * kernel->dim[2] *
                                     kernel->dim[3] * sizeof(int8_t));

    int inner_size = kernel->dim[1] * kernel->dim[2] * kernel->dim[3];
    int outer_size = kernel->dim[0];

    transpose_10_int8(kernel_data, data_buf, inner_size, outer_size);

    int group = params->group;

    int k = kernel->dim[0];
    int m = kernel->dim[1] * kernel->dim[2] * kernel->dim[3] / group;
    for (int g = 0; g < group; g++) {
        shl_rvv_reorder_kernel_n8_fp16_w_int8(data_buf + g * m * k, kernel_data + g * m * k, m, k,
                                              k);
    }
    shl_mem_free(data_buf);
}

/*************************************************************************************
 * Per-channel dequantize int8 -> fp16
 ************************************************************************************/
void shl_rvv_deconv2d_gemm_col2im_dequantize_per_channel_i8_to_f16(
    struct csinn_tensor *kernel, struct csinn_conv2d_params *params, __fp16 *kernel_fp16)
{
    int8_t *kernel_int8 = (int8_t *)kernel->data;
    const int group = params->group;
    const int m = kernel->dim[1] * kernel->dim[2] * kernel->dim[3] / group;
    const int k = kernel->dim[0];
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

//判断a<b
inline static int is_a_ge_zero_and_a_lt_b(int a, int b) { return (unsigned)(a) < (unsigned)(b); }

static void col2im_cpu_ext(const __fp16 *data_col, const __fp16 *bias, const int batch,
                           const int channels, const int height, const int width,
                           const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
                           const int stride_h, const int stride_w, const int dilation_h,
                           const int dilation_w, __fp16 *data_im)
{
    const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int channel_size = height * width;
    const int batch_size = channels * height * width;

    int channel, kernel_row, kernel_col, output_rows, output_col;
    for (int b = 0; b < batch; b++) {
        for (channel = 0; channel < channels; channel++) {
            for (int i = 0; i < channel_size; i++) {
                data_im[i] = (!bias) ? 0.0 : bias[channel];
            }
            for (kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
                for (kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
                    int input_row = -pad_h + kernel_row * dilation_h;
                    for (output_rows = output_h; output_rows; output_rows--) {
                        if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                            data_col += output_w;
                        } else {
                            int input_col = -pad_w + kernel_col * dilation_w;
                            for (output_col = output_w; output_col; output_col--) {
                                if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                    data_im[input_row * width + input_col] += *data_col;
                                }
                                data_col++;
                                input_col += stride_w;
                            }
                        }
                        input_row += stride_h;
                    }
                }
            }
            data_im += channel_size;
        }
    }
}

// Data format : NCHW  Input:[N,IC,IH,IW] Kernel:[OC,KH,KW,IC] Output:[N,OC,OH,OW]
int shl_rvv_deconv2d_gemm_col2im_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                      struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                      struct csinn_conv2d_params *params)
{
    if (input->layout == CSINN_LAYOUT_NC1HWC0) {
        shl_debug_info("Data Format: NC1HWC0\n");
        shl_rvv_tensor_nc1xc0_to_ndarray_replace_fp16(input);
    } else if (input->layout != CSINN_LAYOUT_NCHW) {
        shl_debug_error("Unsupported data format\n");
        return CSINN_FALSE;
    }

    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    __fp16 *kernel_data = NULL;
    __fp16 *bias_data = (__fp16 *)bias->data;

    int32_t batch = input->dim[0];
    int32_t in_c = input->dim[1];
    int32_t in_h = input->dim[2];
    int32_t in_w = input->dim[3];

    int32_t out_c = output->dim[1];
    int32_t out_h = output->dim[2];
    int32_t out_w = output->dim[3];

    int32_t kernel_h = kernel->dim[2];
    int32_t kernel_w = kernel->dim[3];

    int32_t group = params->group;
    int32_t stride_h = params->stride_height;
    int32_t stride_w = params->stride_width;
    int32_t pad_t = params->pad_top;
    int32_t pad_l = params->pad_left;
    int32_t pad_d = params->pad_down;
    int32_t pad_r = params->pad_right;
    int32_t dilation_h = params->dilation_height;
    int32_t dilation_w = params->dilation_width;
    int32_t out_pad_h = params->out_pad_height;
    int32_t out_pad_w = params->out_pad_width;

    int32_t m = out_c / group * kernel_h * kernel_w;
    int32_t k = in_c / group;
    int32_t n = in_h * in_w;

    __fp16 *kernel_fp16 = NULL;
    if (kernel->is_const && kernel->dtype == CSINN_DTYPE_INT8) {
        int size = csinn_tensor_size(kernel);
        kernel_fp16 = (__fp16 *)shl_mem_alloc(size * sizeof(__fp16));
        if (kernel->quant_channel > 1) {
            shl_rvv_deconv2d_gemm_col2im_dequantize_per_channel_i8_to_f16(kernel, params,
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

    __fp16 *reorder_buf = (__fp16 *)shl_mem_alloc(k * n * sizeof(__fp16));
    __fp16 *output_buf = (__fp16 *)shl_mem_alloc(batch * group * m * n * sizeof(__fp16));
    const int vlen = csrr_vlenb() * 8;

    __fp16 *output_buf_ptr = output_buf;
    for (int i = 0; i < batch; i++) {
        for (int g = 0; g < group; g++) {
            if (vlen == 128) {
                // Pack
                shl_rvv_reorder_input_z16_fp16(input_data, reorder_buf, k, n, n);
                // Gemm
                shl_rvv_gemm_8x16_fp16(output_buf_ptr, (kernel_data + g * m * k), reorder_buf, NULL,
                                       m, k, n, n);
            } else {
                shl_debug_error("The vector length is temporarily not supported.");
            }
            input_data += k * n;
            output_buf_ptr += m * n;
        }
    }
    shl_mem_free(reorder_buf);

    col2im_cpu_ext(output_buf, bias_data, batch, out_c, out_h, out_w, kernel_h, kernel_w, pad_t,
                   pad_l, stride_h, stride_w, dilation_h, dilation_w, output_data);

    shl_mem_free(output_buf);

    if (kernel->is_const && kernel->dtype == CSINN_DTYPE_INT8) {
        shl_mem_free(kernel_fp16);
        return CSINN_TRUE;
    }

    // requantize
    shl_rvv_sidcso_op_requantize_fp16(input, output, kernel);
    return CSINN_TRUE;
}