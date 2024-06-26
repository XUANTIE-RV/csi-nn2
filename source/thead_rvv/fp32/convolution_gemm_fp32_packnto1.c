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

/*************************************************************
 * packn = vlenb / sizeof(float)
 * maxk = ksize_h * ksize_w
 * constrain: out_c % packn != 0 and in_ch % packn = 0
 * layout: [out_c/pack2n, in_c/packn, maxk, packn, pack2n]
 *         [out_c/packna, in_c/packnb, maxk, packnb, packna]
 *         [out_c/tail, in_c/packnb, maxk, packnb, tail]
 ************************************************************/
static void im2col_gemm_reorder_kernel_packnto1_per_group_fp32(float *src, float *dst, int out_c,
                                                               int in_c, int maxk)
{
    const int packn = csrr_vlenb() / sizeof(float);
    const int pack2n = packn * 2;

    int vl = vsetvl_e32m2(pack2n);
    int oc = 0;
    // [out_c/pack2n, in_c/packn, maxk, packn, pack2n]
    for (; oc + pack2n - 1 < out_c; oc += pack2n) {
        float *k0 = src + oc * in_c * maxk;
        float *g0 = dst + oc * in_c / packn * maxk * packn;

        for (int ic = 0; ic + packn - 1 < in_c; ic += packn) {
            for (int k = 0; k < maxk; k++) {
                for (int p = 0; p < packn; p++) {
                    vfloat32m2_t _tmp =
                        vlse32_v_f32m2(k0 + ((ic + p) * maxk + k), in_c * maxk * sizeof(float), vl);
                    vse32_v_f32m2(g0, _tmp, vl);
                    g0 += vl;
                }
            }
        }
    }
    vl = vsetvl_e32m1(packn);
    // [out_c/packn, in_c/packn, maxk, packn, packn]
    for (; oc + packn - 1 < out_c; oc += packn) {
        float *k0 = src + oc * in_c * maxk;
        float *g0 = dst + oc * in_c / packn * maxk * packn;

        for (int ic = 0; ic + packn - 1 < in_c; ic += packn) {
            for (int k = 0; k < maxk; k++) {
                for (int p = 0; p < packn; p++) {
                    vfloat32m1_t _tmp =
                        vlse32_v_f32m1(k0 + ((ic + p) * maxk + k), in_c * maxk * sizeof(float), vl);
                    vse32_v_f32m1(g0, _tmp, vl);
                    g0 += vl;
                }
            }
        }
    }
    // [out_c/tail, in_c/packnb, maxk, packnb, tail]
    if (oc < out_c) {
        vl = vsetvl_e32m1(out_c - oc);
        float *k0 = src + oc * in_c * maxk;
        float *g0 = dst + oc * in_c / packn * maxk * packn;

        for (int ic = 0; ic + packn - 1 < in_c; ic += packn) {
            for (int k = 0; k < maxk; k++) {
                for (int p = 0; p < packn; p++) {
                    vfloat32m1_t _tmp =
                        vlse32_v_f32m1(k0 + ((ic + p) * maxk + k), in_c * maxk * sizeof(float), vl);
                    vse32_v_f32m1(g0, _tmp, vl);
                    g0 += vl;
                }
            }
        }
    }
}

/*************************************************************************************
 * reorder kernel_data inplace, means the origin kernel_data be destoried.
 * The reason to do this is that the packaging process must not consume more memory.
 **************************************************************************************/
void shl_rvv_conv_im2col_gemm_reorder_kernel_packnto1_fp32(struct csinn_tensor *kernel,
                                                           struct csinn_conv2d_params *params)
{
    float *kernel_data = (float *)kernel->data;
    int group = params->group;

    int out_c = kernel->dim[0];
    int out_cp = out_c / group;  // per-group out channel
    int in_c = kernel->dim[1];
    int maxk = kernel->dim[2] * kernel->dim[3];

    float *pa_reorder = (float *)shl_mem_alloc(out_c * in_c * maxk * sizeof(float));
    for (int g = 0; g < group; g++) {
        float *ker_ptr = kernel_data + g * out_cp * in_c * maxk;
        float *ker_tm_ptr = pa_reorder + g * out_cp * in_c * maxk;
        im2col_gemm_reorder_kernel_packnto1_per_group_fp32(ker_ptr, ker_tm_ptr, out_cp, in_c, maxk);
    }
    memcpy(kernel_data, pa_reorder, out_c * in_c * maxk * sizeof(float));
    shl_mem_free(pa_reorder);
}

int shl_rvv_common_conv_gemm_packnto1_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                           struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                           struct csinn_conv2d_params *params,
                                           void (*reorder_input)(float *, float *, int, int, int),
                                           void (*gemm)(float *, const float *, const float *,
                                                        float *, int, int, int, bool))
{
    if (input->layout == CSINN_LAYOUT_NCHW) {
        shl_rvv_tensor_ndarray_to_nc1xc0_replace_fp32(input);
    }
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    float *kernel_data = (float *)kernel->data;
    float *bias_data = (float *)bias->data;

    int32_t group = params->group;
    int32_t batch = input->dim[0];
    int32_t in_c = input->dim[1] * input->dim[4];
    int32_t in_h = input->dim[2];
    int32_t in_w = input->dim[3];
    int32_t out_c = kernel->dim[0];
    int32_t out_h = output->dim[2];
    int32_t out_w = output->dim[3];
    int32_t ksize_h = kernel->dim[2];
    int32_t ksize_w = kernel->dim[3];
    int32_t stride_h = params->stride_height;
    int32_t stride_w = params->stride_width;
    int32_t dilation_h = params->dilation_height;
    int32_t dilation_w = params->dilation_width;

    int32_t m = out_c / group;
    int32_t in_cp = in_c / group;
    int32_t maxk = ksize_h * ksize_w;
    int32_t n = out_h * out_w;

    float *output_ncxhwx = (float *)shl_mem_alloc(m * n * sizeof(float));

    for (int i = 0; i < batch; i++) {
        for (int g = 0; g < group; g++) {
            // padding
            int padded_in_h = in_h + params->pad_top + params->pad_down;
            int padded_in_w = in_w + params->pad_left + params->pad_right;
            int padded_in_hw = padded_in_h * padded_in_w;
            float *input_pad_buf = (float *)shl_mem_alloc(in_cp * padded_in_hw * sizeof(float));
            shl_rvv_pad_input_packn_fp32(input_data, input_pad_buf, in_cp, in_h, in_w, padded_in_h,
                                         padded_in_w, params->pad_top, params->pad_left);

            // im2col
            const int packn = csrr_vlenb() / sizeof(float);
            const int vl = vsetvl_e32m1(packn);

            // [in_c/packn, maxk, out_h, out_w, packn]
            float *im2col_buf = (float *)shl_mem_alloc(in_cp / packn * maxk * out_h * out_w *
                                                       packn * sizeof(float));
            const int tailstep = (padded_in_w * stride_h - out_w * stride_w) * packn;

            for (int c = 0; c + packn - 1 < in_cp; c += packn) {
                const float *img0 = input_pad_buf + c * padded_in_hw;
                float *dst_ptr = im2col_buf + c * maxk * out_h * out_w;

                for (int a = 0; a < ksize_h; a++) {
                    for (int b = 0; b < ksize_w; b++) {
                        const float *img1 =
                            img0 + a * dilation_h * padded_in_w * packn + b * dilation_w * packn;

                        for (int p = 0; p < out_h; p++) {
                            for (int q = 0; q < out_w; q++) {
                                vfloat32m1_t _tmp = vle32_v_f32m1(img1, vl);
                                img1 += stride_w * packn;
                                vse32_v_f32m1(dst_ptr, _tmp, vl);
                                dst_ptr += packn;
                            }
                            img1 += tailstep;
                        }
                    }
                }
            }
            shl_mem_free(input_pad_buf);

            // reorder(pack)
            float *reorder_buf = (float *)shl_mem_alloc(in_cp * maxk * n * sizeof(float));
            reorder_input(im2col_buf, reorder_buf, in_cp * maxk, n, n);
            shl_mem_free(im2col_buf);

            // gemm
            float *ker_ptr = kernel_data + g * m * maxk * in_cp;
            float *bias_ptr = bias_data ? (bias_data + g * m) : NULL;
            gemm(output_ncxhwx, ker_ptr, reorder_buf, bias_ptr, m, in_cp * maxk, n, false);
            shl_rvv_reorder_input_packnto1_fp32(output_ncxhwx, output_data, m, out_h, out_w);

            shl_mem_free(reorder_buf);

            input_data += in_cp * in_h * in_w;
            output_data += m * n;
        }
    }
    shl_mem_free(output_ncxhwx);
    return CSINN_TRUE;
}

int shl_rvv_conv_im2col_gemm_packnto1_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                           struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                           struct csinn_conv2d_params *params)
{
    return shl_rvv_common_conv_gemm_packnto1_fp32(input, output, kernel, bias, params,
                                                  shl_rvv_reorder_input_z12_packn_fp32,
                                                  shl_rvv_ncxhwx_gemm_12xpack2n_fp32);
}
