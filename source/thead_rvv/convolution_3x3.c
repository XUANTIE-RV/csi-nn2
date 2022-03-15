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

#include "csi_thead_rvv.h"

/*************************************************************
    note: VLEN = 128/256 ...
*************************************************************/
/*
    padding input for winograd input transform , and change memory layout to [n c/4 h w 4]
    input layout: [n c h w]
    input_padded layout: [n c/packn h w packn]
    constrain: input channel % packn = 0
*/

static void winograd_pad_input_pack1ton_fp32(const float *input, float *input_padded, int inc,
                                             int inh, int inw, int padded_h, int padded_w,
                                             int pad_top, int pad_left)
{
    const int packn = csrr_vlenb() / sizeof(float);
    const int vl = vsetvl_e32m1(packn);

    int padded_hw = padded_h * padded_w;
    const int in_size = inh * inw;  // per-channel size

    float *pad_ptr = input_padded;
    float *inp_ptr = (float *)input;
    int pad_down = padded_h - pad_top - inh;    // remain to pad on h (pad_down)
    int pad_right = padded_w - pad_left - inw;  // remain to pad on w (pad_right)

    vfloat32m1_t _zero = vfmv_v_f_f32m1(0.0f, vl);

    int c = 0;
    for (; c + packn - 1 < inc; c += packn) {
        inp_ptr = (float *)input + c * in_size;
        // pad h_top
        for (int i = 0; i < pad_top * padded_w; i++) {
            vse32_v_f32m1(pad_ptr, _zero, vl);
            pad_ptr += packn;
        }
        // pad h_mid
        for (int i = 0; i < inh; i++) {
            // pad w_left
            for (int j = 0; j < pad_left; j++) {
                vse32_v_f32m1(pad_ptr, _zero, vl);
                pad_ptr += packn;
            }
            // pad w_mid
            for (int j = 0; j < inw; j++) {
                vfloat32m1_t _tmp = vlse32_v_f32m1(inp_ptr, in_size * sizeof(float), vl);
                inp_ptr++;
                vse32_v_f32m1(pad_ptr, _tmp, vl);
                pad_ptr += packn;
            }
            // pad w_end
            for (int j = 0; j < pad_right; j++) {
                vse32_v_f32m1(pad_ptr, _zero, vl);
                pad_ptr += packn;
            }
        }
        // pad h_bottom
        for (int i = 0; i < pad_down * padded_w; i++) {
            vse32_v_f32m1(pad_ptr, _zero, vl);
            pad_ptr += packn;
        }
    }
}

static void winograd_crop_output_packnto1_fp32(const float *output_trans, float *output, int out_c,
                                               int out_h, int out_w, int wino_h, int wino_w)
{
    const int packn = csrr_vlenb() / sizeof(float);
    const int vl = vsetvl_e32m1(packn);

    const int out_size = out_h * out_w;  // per-channel size
    const int crop_size = wino_h * wino_w;

    float *out_tm_ptr = (float *)output_trans;
    float *out_ptr = output;

    int c = 0;
    for (; c + packn - 1 < out_c; c += packn) {
        out_tm_ptr = (float *)output_trans + c * crop_size;
        out_ptr = output + c * out_size;

        for (int h = 0; h < out_h; h++) {
            float *crop_ptr = out_tm_ptr + h * wino_w * packn;
            for (int w = 0; w < out_w; w++) {
                vfloat32m1_t _tmp = vle32_v_f32m1(crop_ptr, vl);
                crop_ptr += packn;
                vsse32_v_f32m1(out_ptr, out_size * sizeof(float), _tmp, vl);
                out_ptr++;
            }
        }
    }
}

/*
    packn = VLEN / 32  (128/32=4  or  256/32=8)
    constrain: output channel % packn = 0
               input channel % packn = 0
    kernel before:  [O I 3*3]
    kernel after :  [O/packn 8*8 I packn]
*/
void csi_nn_rvv_conv3x3s1_winograd64_transform_kernel_packn_fp32(struct csi_tensor *o_kernel,
                                                                 struct csi_tensor *t_kernel)
{
    int32_t outch = o_kernel->dim[0];
    int32_t inch = o_kernel->dim[1];

    float *kernel_data = (float *)o_kernel->data;
    // for kernel transform buf, 3x3 --> 8x8
    float *kernel_tm = (float *)csi_mem_alloc(outch * inch * 8 * 8 * sizeof(float));
    // kernel transform matrix: G
    const float ktm[8][3] = {{1.0f, 0.0f, 0.0f},
                             {-2.0f / 9, -2.0f / 9, -2.0f / 9},
                             {-2.0f / 9, 2.0f / 9, -2.0f / 9},
                             {1.0f / 90, 1.0f / 45, 2.0f / 45},
                             {1.0f / 90, -1.0f / 45, 2.0f / 45},
                             {1.0f / 45, 1.0f / 90, 1.0f / 180},
                             {1.0f / 45, -1.0f / 90, 1.0f / 180},
                             {0.0f, 0.0f, 1.0f}};

    // const float ktm[8][3] = {
    //     {1.0f, 0.0f, 0.0f},
    //     {-2.0f / 9, -2.0f / 9, -2.0f / 9},
    //     {-2.0f / 9, 2.0f / 9, -2.0f / 9},
    //     {1.0f / 90, 1.0f / 45, 2.0f / 45},
    //     {1.0f / 90, -1.0f / 45, 2.0f / 45},
    //     {32.0f / 45, 16.0f / 45, 8.0f / 45},
    //     {32.0f / 45, -16.0f / 45, 8.0f / 45},
    //     {0.0f, 0.0f, 1.0f}
    // };

    csi_tensor_copy(t_kernel, o_kernel);

    for (int p = 0; p < outch; p++) {
        for (int q = 0; q < inch; q++) {
            const float *kernel0 = kernel_data + p * inch * 9 + q * 9;
            float *kernel_tmp = kernel_tm + p * inch * 64 + q * 64;

            // transform kernel
            const float *k0 = kernel0;
            const float *k1 = kernel0 + 3;
            const float *k2 = kernel0 + 6;

            // h : first compute the transport matrix tmp = (g * GT)T
            float tmp[8][3];
            for (int i = 0; i < 8; i++) {
                tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
            }

            // U
            for (int j = 0; j < 8; j++) {
                float *tmpp = &tmp[j][0];

                for (int i = 0; i < 8; i++) {
                    kernel_tmp[j * 8 + i] =
                        tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                }
            }
        }
    }
    // optimized layout for winograd64

    const int packn = csrr_vlenb() / sizeof(float);

    float *kernel_tm_packn = (float *)csi_mem_alloc(outch * inch * 8 * 8 * sizeof(float));
    t_kernel->data = kernel_tm_packn;

    for (int oc = 0; oc < outch / packn; oc++) {
        float *g0 = kernel_tm_packn + oc * 64 * inch * packn;

        for (int k = 0; k < 64; k++) {
            float *g00 = g0 + k * inch * packn;

            for (int ic = 0; ic < inch / packn; ic++) {
                for (int i = 0; i < packn; i++) {
                    for (int j = 0; j < packn; j++) {
                        const float *k00 =
                            kernel_tm + (oc * packn + j) * 64 * inch + (ic * packn + i) * 64;
                        *g00++ = k00[k];
                    }
                }
            }
        }
    }
    csi_mem_free(kernel_tm);
}

/*
    n = VLEN / 32
    constrain: output channel % n = 0
               input channel % n = 0
*/
int csi_nn_rvv_conv3x3s1_winograd64_packn_fp32(struct csi_tensor *input, struct csi_tensor *output,
                                               struct csi_tensor *kernel, struct csi_tensor *bias,
                                               struct conv2d_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    float *kernel_data = (float *)params->conv_extra.kernel_tm->data;
    float *bias_data = (float *)bias->data;

    // param
    int kernel_h = kernel->dim[2];
    int kernel_w = kernel->dim[3];
    int stride_h = params->stride_height;
    int stride_w = params->stride_width;
    int dilation_h = params->dilation_height;
    int dilation_w = params->dilation_width;
    int pad_left = params->pad_left;
    int pad_top = params->pad_top;

    int batch = input->dim[0];
    int in_c = input->dim[1];
    int in_h = input->dim[2];
    int in_w = input->dim[3];
    int input_size = in_c * in_h * in_w;
    int kernel_size = in_c * kernel_h * kernel_w;

    int out_c = kernel->dim[0];
    int out_h = output->dim[2];
    int out_w = output->dim[3];
    int output_size = out_c * out_h * out_w;

    // winograd param
    int block_h = (out_h + 5) / 6;
    int block_w = (out_w + 5) / 6;

    // block * 4 for alignment with 4，kernel = 3 * 3 ，stride = 1，thus input_size + 2
    int padded_in_h = block_h * 6 + 2;
    int padded_in_w = block_w * 6 + 2;
    int padded_in_hw = padded_in_h * padded_in_w;  // element size after padding per channel

    /****************************** bias *****************************/
    bool flag_bias = 1;  // default: conv2d layer include bias
    if (bias_data == NULL) {
        flag_bias = 0;
        bias_data = (float *)csi_mem_alloc(out_c * sizeof(float));
    }

    const int packn = csrr_vlenb() / sizeof(float);
    const int vl = vsetvl_e32m1(packn);

    for (int n = 0; n < batch; n++) {
        // pad buffer: [in_c/8 h w 8]
        float *input_padd_buf = (float *)csi_mem_alloc(in_c * padded_in_hw * sizeof(float));

        // pad input
        winograd_pad_input_pack1ton_fp32(input_data, input_padd_buf, in_c, in_h, in_w, padded_in_h,
                                         padded_in_w, pad_top, pad_left);
        input_data += input_size;

        // input transform buffer1: [in_ch/8, 64, blocks, 8]
        float *input_tm1_buf =
            (float *)csi_mem_alloc(in_c * block_h * block_w * 8 * 8 * sizeof(float));

        /****************************** transform input *****************************/
        /*
        BT = {
            { 1   0    -5.25    0    5.25     0    -1  0 };
            { 0   1      1    -4.25  -4.25    1    1   0 };
            { 0   -1     1    4.25   -4.25   -1    1   0 };
            { 0  0.5    0.25   -2.5   -1.25     2    1   0 };
            { 0  -0.5   0.25    2.5   -1.25    -2    1   0 };
            { 0   2      4    -2.5    -5     0.5   1   0 };
            { 0   -2     4     2.5    -5    -0.5   1   0 };
            { 0   -1     0    5.25     0    -5.25  0   1 }
        };
        */
        int tiles = block_h * block_w;

#pragma omp parallel for num_threads(1)
        for (int q = 0; q < in_c / packn; q++) {
            float *img0 = input_padd_buf + q * padded_in_h * padded_in_w *
                                               packn;  // feature map after padding - q channel
            float *img0_tm =
                input_tm1_buf + q * 64 * tiles * packn;  // transform and interleave - q channel

            float tmp[8][8][packn];

            for (int i = 0; i < block_h; i++) {
                for (int j = 0; j < block_w; j++) {
                    float *r0 = img0 + (i * padded_in_w * 6 + j * 6) *
                                           packn;  // feature map after padding 8*8 start addr
                    float *r0_tm =
                        img0_tm + (i * block_w + j) * packn;  // input_tm1 8*8 block start addr

                    for (int m = 0; m < 8; m++) {
                        vfloat32m1_t _r00 = vle32_v_f32m1(r0, vl);
                        vfloat32m1_t _r01 = vle32_v_f32m1(r0 + packn * 1, vl);
                        vfloat32m1_t _r02 = vle32_v_f32m1(r0 + packn * 2, vl);
                        vfloat32m1_t _r03 = vle32_v_f32m1(r0 + packn * 3, vl);
                        vfloat32m1_t _r04 = vle32_v_f32m1(r0 + packn * 4, vl);
                        vfloat32m1_t _r05 = vle32_v_f32m1(r0 + packn * 5, vl);
                        vfloat32m1_t _r06 = vle32_v_f32m1(r0 + packn * 6, vl);
                        vfloat32m1_t _r07 = vle32_v_f32m1(r0 + packn * 7, vl);

                        vfloat32m1_t _tmp0m = vfmacc_vf_f32m1(vfsub_vv_f32m1(_r00, _r06, vl), 5.25f,
                                                              vfsub_vv_f32m1(_r04, _r02, vl), vl);
                        vfloat32m1_t _tmp7m = vfmacc_vf_f32m1(vfsub_vv_f32m1(_r07, _r01, vl), 5.25f,
                                                              vfsub_vv_f32m1(_r03, _r05, vl), vl);

                        vfloat32m1_t _tmp12a =
                            vfmacc_vf_f32m1(vfadd_vv_f32m1(_r02, _r06, vl), -4.25f, _r04, vl);
                        vfloat32m1_t _tmp12b =
                            vfmacc_vf_f32m1(vfadd_vv_f32m1(_r01, _r05, vl), -4.25f, _r03, vl);
                        vfloat32m1_t _tmp1m = vfadd_vv_f32m1(_tmp12a, _tmp12b, vl);
                        vfloat32m1_t _tmp2m = vfsub_vv_f32m1(_tmp12a, _tmp12b, vl);

                        vfloat32m1_t _tmp34a = vfmacc_vf_f32m1(
                            vfmacc_vf_f32m1(_r06, 0.25f, _r02, vl), -1.25f, _r04, vl);
                        vfloat32m1_t _tmp34b = vfmacc_vf_f32m1(
                            vfmacc_vf_f32m1(vfmul_vf_f32m1(_r01, 0.5f, vl), -2.5f, _r03, vl), 2.f,
                            _r05, vl);
                        vfloat32m1_t _tmp3m = vfadd_vv_f32m1(_tmp34a, _tmp34b, vl);
                        vfloat32m1_t _tmp4m = vfsub_vv_f32m1(_tmp34a, _tmp34b, vl);

                        vfloat32m1_t _tmp56a =
                            vfmacc_vf_f32m1(_r06, 4.f, vfmacc_vf_f32m1(_r02, -1.25f, _r04, vl), vl);
                        vfloat32m1_t _tmp56b = vfmacc_vf_f32m1(
                            vfmacc_vf_f32m1(vfmul_vf_f32m1(_r01, 2.f, vl), -2.5f, _r03, vl), 0.5f,
                            _r05, vl);
                        vfloat32m1_t _tmp5m = vfadd_vv_f32m1(_tmp56a, _tmp56b, vl);
                        vfloat32m1_t _tmp6m = vfsub_vv_f32m1(_tmp56a, _tmp56b, vl);

                        vse32_v_f32m1(tmp[0][m], _tmp0m, vl);
                        vse32_v_f32m1(tmp[7][m], _tmp7m, vl);
                        vse32_v_f32m1(tmp[1][m], _tmp1m, vl);
                        vse32_v_f32m1(tmp[2][m], _tmp2m, vl);
                        vse32_v_f32m1(tmp[3][m], _tmp3m, vl);
                        vse32_v_f32m1(tmp[4][m], _tmp4m, vl);
                        vse32_v_f32m1(tmp[5][m], _tmp5m, vl);
                        vse32_v_f32m1(tmp[6][m], _tmp6m, vl);

                        r0 += padded_in_w * packn;
                    }

                    for (int m = 0; m < 8; m++) {
                        float *r0_tm0 = r0_tm;
                        float *r0_tm1 = r0_tm0 + tiles * packn;
                        float *r0_tm2 = r0_tm1 + tiles * packn;
                        float *r0_tm3 = r0_tm2 + tiles * packn;
                        float *r0_tm4 = r0_tm3 + tiles * packn;
                        float *r0_tm5 = r0_tm4 + tiles * packn;
                        float *r0_tm6 = r0_tm5 + tiles * packn;
                        float *r0_tm7 = r0_tm6 + tiles * packn;

                        vfloat32m1_t _tmp00 = vle32_v_f32m1(tmp[m][0], vl);
                        vfloat32m1_t _tmp01 = vle32_v_f32m1(tmp[m][1], vl);
                        vfloat32m1_t _tmp02 = vle32_v_f32m1(tmp[m][2], vl);
                        vfloat32m1_t _tmp03 = vle32_v_f32m1(tmp[m][3], vl);
                        vfloat32m1_t _tmp04 = vle32_v_f32m1(tmp[m][4], vl);
                        vfloat32m1_t _tmp05 = vle32_v_f32m1(tmp[m][5], vl);
                        vfloat32m1_t _tmp06 = vle32_v_f32m1(tmp[m][6], vl);
                        vfloat32m1_t _tmp07 = vle32_v_f32m1(tmp[m][7], vl);

                        vfloat32m1_t _r0tm0 =
                            vfmacc_vf_f32m1(vfsub_vv_f32m1(_tmp00, _tmp06, vl), 5.25f,
                                            vfsub_vv_f32m1(_tmp04, _tmp02, vl), vl);
                        vfloat32m1_t _r0tm7 =
                            vfmacc_vf_f32m1(vfsub_vv_f32m1(_tmp07, _tmp01, vl), 5.25f,
                                            vfsub_vv_f32m1(_tmp03, _tmp05, vl), vl);

                        vfloat32m1_t _tmp12a =
                            vfmacc_vf_f32m1(vfadd_vv_f32m1(_tmp02, _tmp06, vl), -4.25f, _tmp04, vl);
                        vfloat32m1_t _tmp12b =
                            vfmacc_vf_f32m1(vfadd_vv_f32m1(_tmp01, _tmp05, vl), -4.25f, _tmp03, vl);
                        vfloat32m1_t _r0tm1 = vfadd_vv_f32m1(_tmp12a, _tmp12b, vl);
                        vfloat32m1_t _r0tm2 = vfsub_vv_f32m1(_tmp12a, _tmp12b, vl);

                        vfloat32m1_t _tmp34a = vfmacc_vf_f32m1(
                            vfmacc_vf_f32m1(_tmp06, 0.25f, _tmp02, vl), -1.25f, _tmp04, vl);
                        vfloat32m1_t _tmp34b = vfmacc_vf_f32m1(
                            vfmacc_vf_f32m1(vfmul_vf_f32m1(_tmp01, 0.5f, vl), -2.5f, _tmp03, vl),
                            2.f, _tmp05, vl);
                        vfloat32m1_t _r0tm3 = vfadd_vv_f32m1(_tmp34a, _tmp34b, vl);
                        vfloat32m1_t _r0tm4 = vfsub_vv_f32m1(_tmp34a, _tmp34b, vl);

                        vfloat32m1_t _tmp56a = vfmacc_vf_f32m1(
                            _tmp06, 4.f, vfmacc_vf_f32m1(_tmp02, -1.25f, _tmp04, vl), vl);
                        vfloat32m1_t _tmp56b = vfmacc_vf_f32m1(
                            vfmacc_vf_f32m1(vfmul_vf_f32m1(_tmp01, 2.f, vl), -2.5f, _tmp03, vl),
                            0.5f, _tmp05, vl);
                        vfloat32m1_t _r0tm5 = vfadd_vv_f32m1(_tmp56a, _tmp56b, vl);
                        vfloat32m1_t _r0tm6 = vfsub_vv_f32m1(_tmp56a, _tmp56b, vl);

                        vse32_v_f32m1(r0_tm0, _r0tm0, vl);
                        vse32_v_f32m1(r0_tm7, _r0tm7, vl);
                        vse32_v_f32m1(r0_tm1, _r0tm1, vl);
                        vse32_v_f32m1(r0_tm2, _r0tm2, vl);
                        vse32_v_f32m1(r0_tm3, _r0tm3, vl);
                        vse32_v_f32m1(r0_tm4, _r0tm4, vl);
                        vse32_v_f32m1(r0_tm5, _r0tm5, vl);
                        vse32_v_f32m1(r0_tm6, _r0tm6, vl);

                        r0_tm += tiles * packn * 8;
                    }
                }
            }
        }
        csi_mem_free(input_padd_buf);

        /*********************************** dot ***************************************/
        // reorder input_tm1_buf
        int size_input_tm2 = 0;
        if (tiles >= 8) {
            size_input_tm2 =
                64 * (tiles / 8 + (tiles % 8) / 4 + (tiles % 4) / 2 + tiles % 2) * in_c * 8;
        } else if (tiles >= 4) {
            size_input_tm2 = 64 * (tiles / 4 + (tiles % 4) / 2 + tiles % 2) * in_c * 4;
        } else if (tiles >= 2) {
            size_input_tm2 = 64 * (tiles / 2 + tiles % 2) * in_c * 2;
        } else {
            size_input_tm2 = 64 * tiles * in_c;
        }
        float *input_tm2_buf = (float *)csi_mem_alloc(size_input_tm2 * sizeof(float));

#pragma omp parallel for num_threads(1)
        for (int r = 0; r < 64; r++) {
            float *img_tm2 = input_tm2_buf + r * size_input_tm2 / 64;  // input_tm2 r channel data

            int t = 0;
            for (; t + 7 < tiles; t += 8) {
                float *tm2 = img_tm2 + t * in_c;  // img_tm2 row data
                float *tm1 = input_tm1_buf;

                tm1 += (r * tiles + t) * packn;
                for (int q = 0; q < in_c / packn; q++) {
                    vfloat32m1_t _tmp0 = vle32_v_f32m1(tm1, vl);
                    vfloat32m1_t _tmp1 = vle32_v_f32m1(tm1 + packn * 1, vl);
                    vfloat32m1_t _tmp2 = vle32_v_f32m1(tm1 + packn * 2, vl);
                    vfloat32m1_t _tmp3 = vle32_v_f32m1(tm1 + packn * 3, vl);
                    vfloat32m1_t _tmp4 = vle32_v_f32m1(tm1 + packn * 4, vl);
                    vfloat32m1_t _tmp5 = vle32_v_f32m1(tm1 + packn * 5, vl);
                    vfloat32m1_t _tmp6 = vle32_v_f32m1(tm1 + packn * 6, vl);
                    vfloat32m1_t _tmp7 = vle32_v_f32m1(tm1 + packn * 7, vl);

                    vsseg8e32_v_f32m1(tm2, _tmp0, _tmp1, _tmp2, _tmp3, _tmp4, _tmp5, _tmp6, _tmp7,
                                      vl);
                    tm1 += 64 * tiles * packn;
                    tm2 += 8 * packn;
                }
            }
            for (; t + 3 < tiles; t += 4) {
                float *tm2 = img_tm2 + (t / 8 + (t % 8) / 4) * in_c * 8;  // img_tm2 row data
                float *tm1 = input_tm1_buf;

                tm1 += (r * tiles + t) * packn;
                for (int q = 0; q < in_c / packn; q++) {
                    vfloat32m1_t _tmp0 = vle32_v_f32m1(tm1, vl);
                    vfloat32m1_t _tmp1 = vle32_v_f32m1(tm1 + packn * 1, vl);
                    vfloat32m1_t _tmp2 = vle32_v_f32m1(tm1 + packn * 2, vl);
                    vfloat32m1_t _tmp3 = vle32_v_f32m1(tm1 + packn * 3, vl);

                    vsseg4e32_v_f32m1(tm2, _tmp0, _tmp1, _tmp2, _tmp3, vl);
                    tm1 += 64 * tiles * packn;
                    tm2 += 4 * packn;
                }
            }
            for (; t + 1 < tiles; t += 2) {
                float *tm2 =
                    img_tm2 + (t / 8 + (t % 8) / 4 + (t % 4) / 2) * in_c * 8;  // img_tm2 row data
                float *tm1 = input_tm1_buf;

                tm1 += (r * tiles + t) * packn;
                for (int q = 0; q < in_c / packn; q++) {
                    vfloat32m1_t _tmp0 = vle32_v_f32m1(tm1, vl);
                    vfloat32m1_t _tmp1 = vle32_v_f32m1(tm1 + packn * 1, vl);

                    vsseg2e32_v_f32m1(tm2, _tmp0, _tmp1, vl);
                    tm1 += 64 * tiles * packn;
                    tm2 += 2 * packn;
                }
            }
            for (; t < tiles; t++) {
                float *tm2 = img_tm2 + (t / 8 + (t % 8) / 4 + (t % 4) / 2 + t % 2) * in_c *
                                           8;  // img_tm2 row data
                float *tm1 = input_tm1_buf;

                tm1 += (r * tiles + t) * packn;
                for (int q = 0; q < in_c / packn; q++) {
                    vfloat32m1_t _tmp0 = vle32_v_f32m1(tm1, vl);

                    vse32_v_f32m1(tm2, _tmp0, vl);
                    tm1 += 64 * tiles * packn;
                    tm2 += 1 * packn;
                }
            }
        }
        csi_mem_free(input_tm1_buf);

        // output_dot_buf： [out_c/packn, 64, blocks, packn]
        float *output_dot_buf =
            (float *)csi_mem_alloc(out_c * block_h * block_w * 8 * 8 * sizeof(float));
#pragma omp parallel for num_threads(1)
        for (int p = 0; p < out_c / packn; p++) {
            float *output0_tm = output_dot_buf + p * 64 * tiles * packn;  // 4 channel dot output
            float *kernel0_tm = kernel_data + p * 64 * in_c * packn;      // 4 channel kernel

            for (int r = 0; r < 64; r++) {
                float *img_tm2 = input_tm2_buf + r * size_input_tm2 / 64;  // img_tm2 第r个channel

                int t = 0;
                for (; t + 7 < tiles; t += 8) {
                    float *r0 = img_tm2 + t * in_c;
                    float *k0 = kernel0_tm + r * in_c * packn;

                    vfloat32m1_t _acc0 = vfmv_v_f_f32m1(0.0f, vl);
                    vfloat32m1_t _acc1 = vfmv_v_f_f32m1(0.0f, vl);
                    vfloat32m1_t _acc2 = vfmv_v_f_f32m1(0.0f, vl);
                    vfloat32m1_t _acc3 = vfmv_v_f_f32m1(0.0f, vl);
                    vfloat32m1_t _acc4 = vfmv_v_f_f32m1(0.0f, vl);
                    vfloat32m1_t _acc5 = vfmv_v_f_f32m1(0.0f, vl);
                    vfloat32m1_t _acc6 = vfmv_v_f_f32m1(0.0f, vl);
                    vfloat32m1_t _acc7 = vfmv_v_f_f32m1(0.0f, vl);

                    for (int c = 0; c < in_c; c++) {
                        vfloat32m1_t _kernel = vle32_v_f32m1(k0, vl);
                        k0 += packn;
                        _acc0 = vfmacc_vf_f32m1(_acc0, r0[0], _kernel, vl);
                        _acc1 = vfmacc_vf_f32m1(_acc1, r0[1], _kernel, vl);
                        _acc2 = vfmacc_vf_f32m1(_acc2, r0[2], _kernel, vl);
                        _acc3 = vfmacc_vf_f32m1(_acc3, r0[3], _kernel, vl);
                        _acc4 = vfmacc_vf_f32m1(_acc4, r0[4], _kernel, vl);
                        _acc5 = vfmacc_vf_f32m1(_acc5, r0[5], _kernel, vl);
                        _acc6 = vfmacc_vf_f32m1(_acc6, r0[6], _kernel, vl);
                        _acc7 = vfmacc_vf_f32m1(_acc7, r0[7], _kernel, vl);
                        r0 += 8;
                    }

                    vse32_v_f32m1(output0_tm, _acc0, vl);
                    vse32_v_f32m1(output0_tm + packn * 1, _acc1, vl);
                    vse32_v_f32m1(output0_tm + packn * 2, _acc2, vl);
                    vse32_v_f32m1(output0_tm + packn * 3, _acc3, vl);
                    vse32_v_f32m1(output0_tm + packn * 4, _acc4, vl);
                    vse32_v_f32m1(output0_tm + packn * 5, _acc5, vl);
                    vse32_v_f32m1(output0_tm + packn * 6, _acc6, vl);
                    vse32_v_f32m1(output0_tm + packn * 7, _acc7, vl);
                    output0_tm += packn * 8;
                }

                for (; t + 3 < tiles; t += 4) {
                    float *r0 = img_tm2 + (t / 8 + (t % 8) / 4) * in_c * 8;
                    float *k0 = kernel0_tm + r * in_c * packn;

                    vfloat32m1_t _acc0 = vfmv_v_f_f32m1(0.0f, vl);
                    vfloat32m1_t _acc1 = vfmv_v_f_f32m1(0.0f, vl);
                    vfloat32m1_t _acc2 = vfmv_v_f_f32m1(0.0f, vl);
                    vfloat32m1_t _acc3 = vfmv_v_f_f32m1(0.0f, vl);

                    for (int c = 0; c < in_c; c++) {
                        vfloat32m1_t _kernel = vle32_v_f32m1(k0, vl);
                        k0 += packn;
                        _acc0 = vfmacc_vf_f32m1(_acc0, r0[0], _kernel, vl);
                        _acc1 = vfmacc_vf_f32m1(_acc1, r0[1], _kernel, vl);
                        _acc2 = vfmacc_vf_f32m1(_acc2, r0[2], _kernel, vl);
                        _acc3 = vfmacc_vf_f32m1(_acc3, r0[3], _kernel, vl);
                        r0 += 4;
                    }

                    vse32_v_f32m1(output0_tm, _acc0, vl);
                    vse32_v_f32m1(output0_tm + packn * 1, _acc1, vl);
                    vse32_v_f32m1(output0_tm + packn * 2, _acc2, vl);
                    vse32_v_f32m1(output0_tm + packn * 3, _acc3, vl);
                    output0_tm += packn * 4;
                }
                for (; t + 1 < tiles; t += 2) {
                    float *r0 = img_tm2 + (t / 8 + (t % 8) / 4 + (t % 4) / 2) * in_c * 8;
                    float *k0 = kernel0_tm + r * in_c * packn;

                    vfloat32m1_t _acc0 = vfmv_v_f_f32m1(0.0f, vl);
                    vfloat32m1_t _acc1 = vfmv_v_f_f32m1(0.0f, vl);

                    for (int c = 0; c < in_c; c++) {
                        vfloat32m1_t _kernel = vle32_v_f32m1(k0, vl);
                        k0 += packn;
                        _acc0 = vfmacc_vf_f32m1(_acc0, r0[0], _kernel, vl);
                        _acc1 = vfmacc_vf_f32m1(_acc1, r0[1], _kernel, vl);
                        r0 += 2;
                    }

                    vse32_v_f32m1(output0_tm, _acc0, vl);
                    vse32_v_f32m1(output0_tm + packn * 1, _acc1, vl);
                    output0_tm += packn * 2;
                }
                for (; t < tiles; t++) {
                    float *r0 = img_tm2 + (t / 8 + (t % 8) / 4 + (t % 4) / 2 + t % 2) * in_c * 8;
                    float *k0 = kernel0_tm + r * in_c * packn;

                    vfloat32m1_t _acc0 = vfmv_v_f_f32m1(0.0f, vl);

                    for (int c = 0; c < in_c; c++) {
                        vfloat32m1_t _kernel = vle32_v_f32m1(k0, vl);
                        k0 += packn;
                        _acc0 = vfmacc_vf_f32m1(_acc0, r0[0], _kernel, vl);
                        r0 += 1;
                    }

                    vse32_v_f32m1(output0_tm, _acc0, vl);
                    output0_tm += packn * 1;
                }
            }
        }

        csi_mem_free(input_tm2_buf);

        /*************************** transform output ****************************/
        // output_tm1_buf: [out_c/packn, out_h6, out_w6, packn]
        float *output_tm1_buf =
            (float *)csi_mem_alloc(out_c * block_h * block_w * 6 * 6 * sizeof(float));

/*
AT = {
    { 1  1  1   1    1    1      1    0 };
    { 0  1  -1  2   -2   1/2   -1/2   0 };
    { 0  1  1   4    4   1/4    1/4   0 };
    { 0  1  -1  8   -8   1/8   -1/8   0 };
    { 0  1  1   16  16   1/16  1/16   0 };
    { 0  1  -1  32  -32  1/32  -1/32  1 }
};
AT = {
    { 1  1  1   1    1   32    32   0 };
    { 0  1  -1  2   -2   16   -16   0 };
    { 0  1  1   4    4   8     8    0 };
    { 0  1  -1  8   -8   4    -4    0 };
    { 0  1  1   16  16   2     2    0 };
    { 0  1  -1  32  -32  1    -1    1 }
};
*/
#pragma omp parallel for num_threads(1)
        for (int p = 0; p < out_c / packn; p++) {
            float *bias_tmp = bias_data + p * packn;

            float *out0_tm = output_dot_buf +
                             p * 64 * block_h * block_w * packn;  // 输出转换前/dot后 第p个channel
            float *out0 =
                output_tm1_buf + p * 6 * block_h * 6 * block_w * packn;  // 转换后输出 第p个channel

            float tmp[6][8][packn];

            for (int i = 0; i < block_h; i++) {
                for (int j = 0; j < block_w; j++) {
                    float *output0_tm_0 = out0_tm + (i * block_w + j) * packn;  // 8*8 起始地址
                    float *output0_tm_1 = output0_tm_0 + tiles * packn * 1;
                    float *output0_tm_2 = output0_tm_0 + tiles * packn * 2;
                    float *output0_tm_3 = output0_tm_0 + tiles * packn * 3;
                    float *output0_tm_4 = output0_tm_0 + tiles * packn * 4;
                    float *output0_tm_5 = output0_tm_0 + tiles * packn * 5;
                    float *output0_tm_6 = output0_tm_0 + tiles * packn * 6;
                    float *output0_tm_7 = output0_tm_0 + tiles * packn * 7;

                    float *output0 =
                        out0 + (i * block_w * 6 * 6 + j * 6) * packn;  // 输出 6*6 的起始地址

                    for (int m = 0; m < 8; m++) {
                        vfloat32m1_t _r00 = vle32_v_f32m1(output0_tm_0, vl);
                        vfloat32m1_t _r01 = vle32_v_f32m1(output0_tm_1, vl);
                        vfloat32m1_t _r02 = vle32_v_f32m1(output0_tm_2, vl);
                        vfloat32m1_t _r03 = vle32_v_f32m1(output0_tm_3, vl);
                        vfloat32m1_t _r04 = vle32_v_f32m1(output0_tm_4, vl);
                        vfloat32m1_t _r05 = vle32_v_f32m1(output0_tm_5, vl);
                        vfloat32m1_t _r06 = vle32_v_f32m1(output0_tm_6, vl);
                        vfloat32m1_t _r07 = vle32_v_f32m1(output0_tm_7, vl);

                        vfloat32m1_t _tmp024a = vfadd_vv_f32m1(_r01, _r02, vl);
                        vfloat32m1_t _tmp135a = vfsub_vv_f32m1(_r01, _r02, vl);

                        vfloat32m1_t _tmp024b = vfadd_vv_f32m1(_r03, _r04, vl);
                        vfloat32m1_t _tmp135b = vfsub_vv_f32m1(_r03, _r04, vl);

                        vfloat32m1_t _tmp024c = vfadd_vv_f32m1(_r05, _r06, vl);
                        vfloat32m1_t _tmp135c = vfsub_vv_f32m1(_r05, _r06, vl);

                        vfloat32m1_t _tmp0m =
                            vfadd_vv_f32m1(vfadd_vv_f32m1(_r00, _tmp024a, vl),
                                           vfmacc_vf_f32m1(_tmp024b, 32.f, _tmp024c, vl), vl);
                        vfloat32m1_t _tmp2m = vfmacc_vf_f32m1(
                            vfmacc_vf_f32m1(_tmp024a, 4.f, _tmp024b, vl), 8.f, _tmp024c, vl);
                        vfloat32m1_t _tmp4m = vfmacc_vf_f32m1(
                            vfmacc_vf_f32m1(_tmp024a, 16.f, _tmp024b, vl), 2.f, _tmp024c, vl);

                        vfloat32m1_t _tmp1m = vfmacc_vf_f32m1(
                            vfmacc_vf_f32m1(_tmp135a, 2.f, _tmp135b, vl), 16.f, _tmp135c, vl);
                        vfloat32m1_t _tmp3m = vfmacc_vf_f32m1(
                            vfmacc_vf_f32m1(_tmp135a, 8.f, _tmp135b, vl), 4.f, _tmp135c, vl);
                        vfloat32m1_t _tmp5m =
                            vfadd_vv_f32m1(vfadd_vv_f32m1(_r07, _tmp135a, vl),
                                           vfmacc_vf_f32m1(_tmp135c, 32.f, _tmp135b, vl), vl);

                        vse32_v_f32m1(tmp[0][m], _tmp0m, vl);
                        vse32_v_f32m1(tmp[2][m], _tmp2m, vl);
                        vse32_v_f32m1(tmp[4][m], _tmp4m, vl);
                        vse32_v_f32m1(tmp[1][m], _tmp1m, vl);
                        vse32_v_f32m1(tmp[3][m], _tmp3m, vl);
                        vse32_v_f32m1(tmp[5][m], _tmp5m, vl);

                        output0_tm_0 += tiles * packn * 8;
                        output0_tm_1 += tiles * packn * 8;
                        output0_tm_2 += tiles * packn * 8;
                        output0_tm_3 += tiles * packn * 8;
                        output0_tm_4 += tiles * packn * 8;
                        output0_tm_5 += tiles * packn * 8;
                        output0_tm_6 += tiles * packn * 8;
                        output0_tm_7 += tiles * packn * 8;
                    }

                    vfloat32m1_t _bias = vle32_v_f32m1(bias_tmp, vl);
                    for (int m = 0; m < 6; m++) {
                        vfloat32m1_t _tmp00 = vle32_v_f32m1(tmp[m][0], vl);
                        vfloat32m1_t _tmp01 = vle32_v_f32m1(tmp[m][1], vl);
                        vfloat32m1_t _tmp02 = vle32_v_f32m1(tmp[m][2], vl);
                        vfloat32m1_t _tmp03 = vle32_v_f32m1(tmp[m][3], vl);
                        vfloat32m1_t _tmp04 = vle32_v_f32m1(tmp[m][4], vl);
                        vfloat32m1_t _tmp05 = vle32_v_f32m1(tmp[m][5], vl);
                        vfloat32m1_t _tmp06 = vle32_v_f32m1(tmp[m][6], vl);
                        vfloat32m1_t _tmp07 = vle32_v_f32m1(tmp[m][7], vl);

                        vfloat32m1_t _tmp024a = vfadd_vv_f32m1(_tmp01, _tmp02, vl);
                        vfloat32m1_t _tmp135a = vfsub_vv_f32m1(_tmp01, _tmp02, vl);

                        vfloat32m1_t _tmp024b = vfadd_vv_f32m1(_tmp03, _tmp04, vl);
                        vfloat32m1_t _tmp135b = vfsub_vv_f32m1(_tmp03, _tmp04, vl);

                        vfloat32m1_t _tmp024c = vfadd_vv_f32m1(_tmp05, _tmp06, vl);
                        vfloat32m1_t _tmp135c = vfsub_vv_f32m1(_tmp05, _tmp06, vl);

                        vfloat32m1_t _output00 = vfadd_vv_f32m1(
                            _bias,
                            vfadd_vv_f32m1(vfadd_vv_f32m1(_tmp00, _tmp024a, vl),
                                           vfmacc_vf_f32m1(_tmp024b, 32.f, _tmp024c, vl), vl),
                            vl);
                        vfloat32m1_t _output02 = vfadd_vv_f32m1(
                            _bias,
                            vfmacc_vf_f32m1(vfmacc_vf_f32m1(_tmp024a, 4.f, _tmp024b, vl), 8.f,
                                            _tmp024c, vl),
                            vl);
                        vfloat32m1_t _output04 = vfadd_vv_f32m1(
                            _bias,
                            vfmacc_vf_f32m1(vfmacc_vf_f32m1(_tmp024a, 16.f, _tmp024b, vl), 2.f,
                                            _tmp024c, vl),
                            vl);

                        vfloat32m1_t _output01 = vfadd_vv_f32m1(
                            _bias,
                            vfmacc_vf_f32m1(vfmacc_vf_f32m1(_tmp135a, 2.f, _tmp135b, vl), 16.f,
                                            _tmp135c, vl),
                            vl);
                        vfloat32m1_t _output03 = vfadd_vv_f32m1(
                            _bias,
                            vfmacc_vf_f32m1(vfmacc_vf_f32m1(_tmp135a, 8.f, _tmp135b, vl), 4.f,
                                            _tmp135c, vl),
                            vl);
                        vfloat32m1_t _output05 = vfadd_vv_f32m1(
                            _bias,
                            vfadd_vv_f32m1(vfadd_vv_f32m1(_tmp07, _tmp135a, vl),
                                           vfmacc_vf_f32m1(_tmp135c, 32.f, _tmp135b, vl), vl),
                            vl);

                        vse32_v_f32m1(output0, _output00, vl);
                        vse32_v_f32m1(output0 + packn * 2, _output02, vl);
                        vse32_v_f32m1(output0 + packn * 4, _output04, vl);
                        vse32_v_f32m1(output0 + packn * 1, _output01, vl);
                        vse32_v_f32m1(output0 + packn * 3, _output03, vl);
                        vse32_v_f32m1(output0 + packn * 5, _output05, vl);

                        output0 += block_w * 6 * packn;
                    }
                }
            }
        }

        csi_mem_free(output_dot_buf);

        // crop the output after transform: cut extra part (right , bottom)
        winograd_crop_output_packnto1_fp32(output_tm1_buf, output_data, out_c, out_h, out_w,
                                           block_h * 6, block_w * 6);
        output_data += output_size;
        csi_mem_free(output_tm1_buf);
    }

    if (!flag_bias) {
        csi_mem_free(bias_data);
        bias_data = NULL;
    }
    return CSINN_TRUE;
}
