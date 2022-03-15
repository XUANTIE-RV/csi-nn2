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
static void winograd_pad_input_pack1ton_fp16(const __fp16 *input, __fp16 *input_padded, int inc,
                                             int inh, int inw, int padded_h, int padded_w,
                                             int pad_top, int pad_left)
{
    const int packn = csrr_vlenb() / sizeof(__fp16);
    const int vl = vsetvl_e16m1(packn);

    int padded_hw = padded_h * padded_w;
    const int in_size = inh * inw;  // per-channel size

    __fp16 *pad_ptr = input_padded;
    __fp16 *inp_ptr = (__fp16 *)input;
    int pad_down = padded_h - pad_top - inh;    // remain to pad on h (pad_down)
    int pad_right = padded_w - pad_left - inw;  // remain to pad on w (pad_right)

    vfloat16m1_t _zero = vfmv_v_f_f16m1(0.0f, vl);

    int c = 0;
    for (; c + packn - 1 < inc; c += packn) {
        inp_ptr = (__fp16 *)input + c * in_size;
        // pad h_top
        for (int i = 0; i < pad_top * padded_w; i++) {
            vse16_v_f16m1(pad_ptr, _zero, vl);
            pad_ptr += packn;
        }
        // pad h_mid
        for (int i = 0; i < inh; i++) {
            // pad w_left
            for (int j = 0; j < pad_left; j++) {
                vse16_v_f16m1(pad_ptr, _zero, vl);
                pad_ptr += packn;
            }
            // pad w_mid
            for (int j = 0; j < inw; j++) {
                vfloat16m1_t _tmp = vlse16_v_f16m1(inp_ptr, in_size * sizeof(__fp16), vl);
                inp_ptr++;
                vse16_v_f16m1(pad_ptr, _tmp, vl);
                pad_ptr += packn;
            }
            // pad w_end
            for (int j = 0; j < pad_right; j++) {
                vse16_v_f16m1(pad_ptr, _zero, vl);
                pad_ptr += packn;
            }
        }
        // pad h_bottom
        for (int i = 0; i < pad_down * padded_w; i++) {
            vse16_v_f16m1(pad_ptr, _zero, vl);
            pad_ptr += packn;
        }
    }
}

static void winograd_crop_output_packnto1_fp16(const __fp16 *output_trans, __fp16 *output,
                                               int out_c, int out_h, int out_w, int wino_h,
                                               int wino_w)
{
    const int packn = csrr_vlenb() / sizeof(__fp16);
    const int vl = vsetvl_e16m1(packn);

    const int out_size = out_h * out_w;  // per-channel size
    const int crop_size = wino_h * wino_w;

    __fp16 *out_tm_ptr = (__fp16 *)output_trans;
    __fp16 *out_ptr = output;

    int c = 0;
    for (; c + packn - 1 < out_c; c += packn) {
        out_tm_ptr = (__fp16 *)output_trans + c * crop_size;
        out_ptr = output + c * out_size;

        for (int h = 0; h < out_h; h++) {
            __fp16 *crop_ptr = out_tm_ptr + h * wino_w * packn;
            for (int w = 0; w < out_w; w++) {
                vfloat16m1_t _tmp = vle16_v_f16m1(crop_ptr, vl);
                crop_ptr += packn;
                vsse16_v_f16m1(out_ptr, out_size * sizeof(__fp16), _tmp, vl);
                out_ptr++;
            }
        }
    }
}

/*
    pack n = VLEN / 16  (128/16=8  or  256/16=16)
    constrain: output channel % n = 0
               input channel % n = 0
    kernel before:  [O I 3*3]
    kernel after :  [O/n 8*8 I n]
*/
void csi_nn_rvv_conv3x3s1_winograd64_transform_kernel_packn_fp16(struct csi_tensor *o_kernel,
                                                                 struct csi_tensor *t_kernel)
{
    int32_t outch = o_kernel->dim[0];
    int32_t inch = o_kernel->dim[1];

    __fp16 *kernel_data = (__fp16 *)o_kernel->data;
    // for kernel transform buf, 3x3 --> 8x8
    __fp16 *kernel_tm = (__fp16 *)csi_mem_alloc(outch * inch * 8 * 8 * sizeof(__fp16));
    // kernel transform matrix: G
    const __fp16 ktm[8][3] = {{1.0f, 0.0f, 0.0f},
                              {-2.0f / 9, -2.0f / 9, -2.0f / 9},
                              {-2.0f / 9, 2.0f / 9, -2.0f / 9},
                              {1.0f / 90, 1.0f / 45, 2.0f / 45},
                              {1.0f / 90, -1.0f / 45, 2.0f / 45},
                              {1.0f / 45, 1.0f / 90, 1.0f / 180},
                              {1.0f / 45, -1.0f / 90, 1.0f / 180},
                              {0.0f, 0.0f, 1.0f}};

    // const __fp16 ktm[8][3] = {
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
            const __fp16 *kernel0 = kernel_data + p * inch * 9 + q * 9;
            __fp16 *kernel_tmp = kernel_tm + p * inch * 64 + q * 64;

            // transform kernel
            const __fp16 *k0 = kernel0;
            const __fp16 *k1 = kernel0 + 3;
            const __fp16 *k2 = kernel0 + 6;

            // h : first compute the transport matrix tmp = (g * GT)T
            __fp16 tmp[8][3];
            for (int i = 0; i < 8; i++) {
                tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
            }

            // U
            for (int j = 0; j < 8; j++) {
                __fp16 *tmpp = &tmp[j][0];

                for (int i = 0; i < 8; i++) {
                    kernel_tmp[j * 8 + i] =
                        tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                }
            }
        }
    }
    // optimized layout for winograd64
    const int packn = csrr_vlenb() / sizeof(__fp16);

    __fp16 *kernel_tm_packn = (__fp16 *)csi_mem_alloc(outch * inch * 8 * 8 * sizeof(__fp16));
    t_kernel->data = kernel_tm_packn;

    for (int oc = 0; oc < outch / packn; oc++) {
        __fp16 *g0 = kernel_tm_packn + oc * 64 * inch * packn;

        for (int k = 0; k < 64; k++) {
            __fp16 *g00 = g0 + k * inch * packn;

            for (int ic = 0; ic < inch / packn; ic++) {
                for (int i = 0; i < packn; i++) {
                    for (int j = 0; j < packn; j++) {
                        const __fp16 *k00 =
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
    n = VLEN / 16
    constrain: output channel % n = 0
               input channel % n = 0
*/
int csi_nn_rvv_conv3x3s1_winograd64_packn_fp16(struct csi_tensor *input, struct csi_tensor *output,
                                               struct csi_tensor *kernel, struct csi_tensor *bias,
                                               struct conv2d_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    __fp16 *kernel_data = (__fp16 *)params->conv_extra.kernel_tm->data;
    __fp16 *bias_data = (__fp16 *)bias->data;

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

    int padded_in_h =
        block_h * 6 +
        2;  // block * 4 for alignment with 4，kernel = 3 * 3 ，stride = 1，thus input_size + 2
    int padded_in_w = block_w * 6 + 2;
    int padded_in_hw = padded_in_h * padded_in_w;  // element size after padding per channel

    /****************************** bias *****************************/
    bool flag_bias = 1;  // default: conv2d layer include bias
    if (bias_data == NULL) {
        flag_bias = 0;
        bias_data = (__fp16 *)csi_mem_alloc(out_c * sizeof(__fp16));
    }

    const int packn = csrr_vlenb() / sizeof(__fp16);
    const int vl = vsetvl_e16m1(packn);

    for (int n = 0; n < batch; n++) {
        // pad buffer: [in_c/8 h w 8]
        __fp16 *input_padd_buf = (__fp16 *)csi_mem_alloc(in_c * padded_in_hw * sizeof(__fp16));

        // pad input
        winograd_pad_input_pack1ton_fp16(input_data, input_padd_buf, in_c, in_h, in_w, padded_in_h,
                                         padded_in_w, pad_top, pad_left);
        input_data += input_size;

        // input transform buffer1: [in_ch/8, 64, blocks, 8]
        __fp16 *input_tm1_buf =
            (__fp16 *)csi_mem_alloc(in_c * block_h * block_w * 8 * 8 * sizeof(__fp16));

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
            __fp16 *img0 = input_padd_buf + q * padded_in_h * padded_in_w *
                                                packn;  // feature map after padding - q channel
            __fp16 *img0_tm =
                input_tm1_buf + q * 64 * tiles * packn;  // transform and interleave - q channel

            __fp16 tmp[8][8][packn];

            for (int i = 0; i < block_h; i++) {
                for (int j = 0; j < block_w; j++) {
                    __fp16 *r0 = img0 + (i * padded_in_w * 6 + j * 6) *
                                            packn;  // feature map after padding 8*8 start addr
                    __fp16 *r0_tm =
                        img0_tm + (i * block_w + j) * packn;  // input_tm1 8*8 block start addr

                    for (int m = 0; m < 8; m++) {
                        vfloat16m1_t _r00 = vle16_v_f16m1(r0, vl);
                        vfloat16m1_t _r01 = vle16_v_f16m1(r0 + packn * 1, vl);
                        vfloat16m1_t _r02 = vle16_v_f16m1(r0 + packn * 2, vl);
                        vfloat16m1_t _r03 = vle16_v_f16m1(r0 + packn * 3, vl);
                        vfloat16m1_t _r04 = vle16_v_f16m1(r0 + packn * 4, vl);
                        vfloat16m1_t _r05 = vle16_v_f16m1(r0 + packn * 5, vl);
                        vfloat16m1_t _r06 = vle16_v_f16m1(r0 + packn * 6, vl);
                        vfloat16m1_t _r07 = vle16_v_f16m1(r0 + packn * 7, vl);

                        vfloat16m1_t _tmp0m = vfmacc_vf_f16m1(vfsub_vv_f16m1(_r00, _r06, vl), 5.25f,
                                                              vfsub_vv_f16m1(_r04, _r02, vl), vl);
                        vfloat16m1_t _tmp7m = vfmacc_vf_f16m1(vfsub_vv_f16m1(_r07, _r01, vl), 5.25f,
                                                              vfsub_vv_f16m1(_r03, _r05, vl), vl);

                        vfloat16m1_t _tmp12a =
                            vfmacc_vf_f16m1(vfadd_vv_f16m1(_r02, _r06, vl), -4.25f, _r04, vl);
                        vfloat16m1_t _tmp12b =
                            vfmacc_vf_f16m1(vfadd_vv_f16m1(_r01, _r05, vl), -4.25f, _r03, vl);
                        vfloat16m1_t _tmp1m = vfadd_vv_f16m1(_tmp12a, _tmp12b, vl);
                        vfloat16m1_t _tmp2m = vfsub_vv_f16m1(_tmp12a, _tmp12b, vl);

                        vfloat16m1_t _tmp34a = vfmacc_vf_f16m1(
                            vfmacc_vf_f16m1(_r06, 0.25f, _r02, vl), -1.25f, _r04, vl);
                        vfloat16m1_t _tmp34b = vfmacc_vf_f16m1(
                            vfmacc_vf_f16m1(vfmul_vf_f16m1(_r01, 0.5f, vl), -2.5f, _r03, vl), 2.f,
                            _r05, vl);
                        vfloat16m1_t _tmp3m = vfadd_vv_f16m1(_tmp34a, _tmp34b, vl);
                        vfloat16m1_t _tmp4m = vfsub_vv_f16m1(_tmp34a, _tmp34b, vl);

                        vfloat16m1_t _tmp56a =
                            vfmacc_vf_f16m1(_r06, 4.f, vfmacc_vf_f16m1(_r02, -1.25f, _r04, vl), vl);
                        vfloat16m1_t _tmp56b = vfmacc_vf_f16m1(
                            vfmacc_vf_f16m1(vfmul_vf_f16m1(_r01, 2.f, vl), -2.5f, _r03, vl), 0.5f,
                            _r05, vl);
                        vfloat16m1_t _tmp5m = vfadd_vv_f16m1(_tmp56a, _tmp56b, vl);
                        vfloat16m1_t _tmp6m = vfsub_vv_f16m1(_tmp56a, _tmp56b, vl);

                        vse16_v_f16m1(tmp[0][m], _tmp0m, vl);
                        vse16_v_f16m1(tmp[7][m], _tmp7m, vl);
                        vse16_v_f16m1(tmp[1][m], _tmp1m, vl);
                        vse16_v_f16m1(tmp[2][m], _tmp2m, vl);
                        vse16_v_f16m1(tmp[3][m], _tmp3m, vl);
                        vse16_v_f16m1(tmp[4][m], _tmp4m, vl);
                        vse16_v_f16m1(tmp[5][m], _tmp5m, vl);
                        vse16_v_f16m1(tmp[6][m], _tmp6m, vl);

                        r0 += padded_in_w * packn;
                    }

                    for (int m = 0; m < 8; m++) {
                        __fp16 *r0_tm0 = r0_tm;
                        __fp16 *r0_tm1 = r0_tm0 + tiles * packn;
                        __fp16 *r0_tm2 = r0_tm1 + tiles * packn;
                        __fp16 *r0_tm3 = r0_tm2 + tiles * packn;
                        __fp16 *r0_tm4 = r0_tm3 + tiles * packn;
                        __fp16 *r0_tm5 = r0_tm4 + tiles * packn;
                        __fp16 *r0_tm6 = r0_tm5 + tiles * packn;
                        __fp16 *r0_tm7 = r0_tm6 + tiles * packn;

                        vfloat16m1_t _tmp00 = vle16_v_f16m1(tmp[m][0], vl);
                        vfloat16m1_t _tmp01 = vle16_v_f16m1(tmp[m][1], vl);
                        vfloat16m1_t _tmp02 = vle16_v_f16m1(tmp[m][2], vl);
                        vfloat16m1_t _tmp03 = vle16_v_f16m1(tmp[m][3], vl);
                        vfloat16m1_t _tmp04 = vle16_v_f16m1(tmp[m][4], vl);
                        vfloat16m1_t _tmp05 = vle16_v_f16m1(tmp[m][5], vl);
                        vfloat16m1_t _tmp06 = vle16_v_f16m1(tmp[m][6], vl);
                        vfloat16m1_t _tmp07 = vle16_v_f16m1(tmp[m][7], vl);

                        vfloat16m1_t _r0tm0 =
                            vfmacc_vf_f16m1(vfsub_vv_f16m1(_tmp00, _tmp06, vl), 5.25f,
                                            vfsub_vv_f16m1(_tmp04, _tmp02, vl), vl);
                        vfloat16m1_t _r0tm7 =
                            vfmacc_vf_f16m1(vfsub_vv_f16m1(_tmp07, _tmp01, vl), 5.25f,
                                            vfsub_vv_f16m1(_tmp03, _tmp05, vl), vl);

                        vfloat16m1_t _tmp12a =
                            vfmacc_vf_f16m1(vfadd_vv_f16m1(_tmp02, _tmp06, vl), -4.25f, _tmp04, vl);
                        vfloat16m1_t _tmp12b =
                            vfmacc_vf_f16m1(vfadd_vv_f16m1(_tmp01, _tmp05, vl), -4.25f, _tmp03, vl);
                        vfloat16m1_t _r0tm1 = vfadd_vv_f16m1(_tmp12a, _tmp12b, vl);
                        vfloat16m1_t _r0tm2 = vfsub_vv_f16m1(_tmp12a, _tmp12b, vl);

                        vfloat16m1_t _tmp34a = vfmacc_vf_f16m1(
                            vfmacc_vf_f16m1(_tmp06, 0.25f, _tmp02, vl), -1.25f, _tmp04, vl);
                        vfloat16m1_t _tmp34b = vfmacc_vf_f16m1(
                            vfmacc_vf_f16m1(vfmul_vf_f16m1(_tmp01, 0.5f, vl), -2.5f, _tmp03, vl),
                            2.f, _tmp05, vl);
                        vfloat16m1_t _r0tm3 = vfadd_vv_f16m1(_tmp34a, _tmp34b, vl);
                        vfloat16m1_t _r0tm4 = vfsub_vv_f16m1(_tmp34a, _tmp34b, vl);

                        vfloat16m1_t _tmp56a = vfmacc_vf_f16m1(
                            _tmp06, 4.f, vfmacc_vf_f16m1(_tmp02, -1.25f, _tmp04, vl), vl);
                        vfloat16m1_t _tmp56b = vfmacc_vf_f16m1(
                            vfmacc_vf_f16m1(vfmul_vf_f16m1(_tmp01, 2.f, vl), -2.5f, _tmp03, vl),
                            0.5f, _tmp05, vl);
                        vfloat16m1_t _r0tm5 = vfadd_vv_f16m1(_tmp56a, _tmp56b, vl);
                        vfloat16m1_t _r0tm6 = vfsub_vv_f16m1(_tmp56a, _tmp56b, vl);

                        vse16_v_f16m1(r0_tm0, _r0tm0, vl);
                        vse16_v_f16m1(r0_tm7, _r0tm7, vl);
                        vse16_v_f16m1(r0_tm1, _r0tm1, vl);
                        vse16_v_f16m1(r0_tm2, _r0tm2, vl);
                        vse16_v_f16m1(r0_tm3, _r0tm3, vl);
                        vse16_v_f16m1(r0_tm4, _r0tm4, vl);
                        vse16_v_f16m1(r0_tm5, _r0tm5, vl);
                        vse16_v_f16m1(r0_tm6, _r0tm6, vl);

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
        __fp16 *input_tm2_buf = (__fp16 *)csi_mem_alloc(size_input_tm2 * sizeof(__fp16));

#pragma omp parallel for num_threads(1)
        for (int r = 0; r < 64; r++) {
            __fp16 *img_tm2 = input_tm2_buf + r * size_input_tm2 / 64;  // input_tm2 r channel data

            int t = 0;
            for (; t + 7 < tiles; t += 8) {
                __fp16 *tm2 = img_tm2 + t * in_c;  // img_tm2 row data
                __fp16 *tm1 = input_tm1_buf;

                tm1 += (r * tiles + t) * packn;
                for (int q = 0; q < in_c / packn; q++) {
                    vfloat16m1_t _tmp0 = vle16_v_f16m1(tm1, vl);
                    vfloat16m1_t _tmp1 = vle16_v_f16m1(tm1 + packn * 1, vl);
                    vfloat16m1_t _tmp2 = vle16_v_f16m1(tm1 + packn * 2, vl);
                    vfloat16m1_t _tmp3 = vle16_v_f16m1(tm1 + packn * 3, vl);
                    vfloat16m1_t _tmp4 = vle16_v_f16m1(tm1 + packn * 4, vl);
                    vfloat16m1_t _tmp5 = vle16_v_f16m1(tm1 + packn * 5, vl);
                    vfloat16m1_t _tmp6 = vle16_v_f16m1(tm1 + packn * 6, vl);
                    vfloat16m1_t _tmp7 = vle16_v_f16m1(tm1 + packn * 7, vl);

                    vsseg8e16_v_f16m1(tm2, _tmp0, _tmp1, _tmp2, _tmp3, _tmp4, _tmp5, _tmp6, _tmp7,
                                      vl);
                    tm1 += 64 * tiles * packn;
                    tm2 += 8 * packn;
                }
            }
            for (; t + 3 < tiles; t += 4) {
                __fp16 *tm2 = img_tm2 + (t / 8 + (t % 8) / 4) * in_c * 8;  // img_tm2 row data
                __fp16 *tm1 = input_tm1_buf;

                tm1 += (r * tiles + t) * packn;
                for (int q = 0; q < in_c / packn; q++) {
                    vfloat16m1_t _tmp0 = vle16_v_f16m1(tm1, vl);
                    vfloat16m1_t _tmp1 = vle16_v_f16m1(tm1 + packn * 1, vl);
                    vfloat16m1_t _tmp2 = vle16_v_f16m1(tm1 + packn * 2, vl);
                    vfloat16m1_t _tmp3 = vle16_v_f16m1(tm1 + packn * 3, vl);

                    vsseg4e16_v_f16m1(tm2, _tmp0, _tmp1, _tmp2, _tmp3, vl);
                    tm1 += 64 * tiles * packn;
                    tm2 += 4 * packn;
                }
            }
            for (; t + 1 < tiles; t += 2) {
                __fp16 *tm2 =
                    img_tm2 + (t / 8 + (t % 8) / 4 + (t % 4) / 2) * in_c * 8;  // img_tm2 row data
                __fp16 *tm1 = input_tm1_buf;

                tm1 += (r * tiles + t) * packn;
                for (int q = 0; q < in_c / packn; q++) {
                    vfloat16m1_t _tmp0 = vle16_v_f16m1(tm1, vl);
                    vfloat16m1_t _tmp1 = vle16_v_f16m1(tm1 + packn * 1, vl);

                    vsseg2e16_v_f16m1(tm2, _tmp0, _tmp1, vl);
                    tm1 += 64 * tiles * packn;
                    tm2 += 2 * packn;
                }
            }
            for (; t < tiles; t++) {
                __fp16 *tm2 = img_tm2 + (t / 8 + (t % 8) / 4 + (t % 4) / 2 + t % 2) * in_c *
                                            8;  // img_tm2 row data
                __fp16 *tm1 = input_tm1_buf;

                tm1 += (r * tiles + t) * packn;
                for (int q = 0; q < in_c / packn; q++) {
                    vfloat16m1_t _tmp0 = vle16_v_f16m1(tm1, vl);

                    vse16_v_f16m1(tm2, _tmp0, vl);
                    tm1 += 64 * tiles * packn;
                    tm2 += 1 * packn;
                }
            }
        }

        csi_mem_free(input_tm1_buf);

        // output_dot_buf： [out_c/8, 64, blocks, 8]
        __fp16 *output_dot_buf =
            (__fp16 *)csi_mem_alloc(out_c * block_h * block_w * 8 * 8 * sizeof(__fp16));

#pragma omp parallel for num_threads(1)
        for (int p = 0; p < out_c / packn; p++) {
            __fp16 *output0_tm = output_dot_buf + p * 64 * tiles * packn;
            __fp16 *kernel0_tm = kernel_data + p * 64 * in_c * packn;

            for (int r = 0; r < 64; r++) {
                __fp16 *img_tm2 = input_tm2_buf + r * size_input_tm2 / 64;  // img_tm2 第r个channel

                int t = 0;
                for (; t + 7 < tiles; t += 8) {
                    __fp16 *r0 = img_tm2 + t * in_c;
                    __fp16 *k0 = kernel0_tm + r * in_c * packn;

                    vfloat16m1_t _acc0 = vfmv_v_f_f16m1(0.0f, vl);
                    vfloat16m1_t _acc1 = vfmv_v_f_f16m1(0.0f, vl);
                    vfloat16m1_t _acc2 = vfmv_v_f_f16m1(0.0f, vl);
                    vfloat16m1_t _acc3 = vfmv_v_f_f16m1(0.0f, vl);
                    vfloat16m1_t _acc4 = vfmv_v_f_f16m1(0.0f, vl);
                    vfloat16m1_t _acc5 = vfmv_v_f_f16m1(0.0f, vl);
                    vfloat16m1_t _acc6 = vfmv_v_f_f16m1(0.0f, vl);
                    vfloat16m1_t _acc7 = vfmv_v_f_f16m1(0.0f, vl);

                    for (int c = 0; c < in_c; c++) {
                        vfloat16m1_t _kernel = vle16_v_f16m1(k0, vl);
                        k0 += packn;
                        _acc0 = vfmacc_vf_f16m1(_acc0, r0[0], _kernel, vl);
                        _acc1 = vfmacc_vf_f16m1(_acc1, r0[1], _kernel, vl);
                        _acc2 = vfmacc_vf_f16m1(_acc2, r0[2], _kernel, vl);
                        _acc3 = vfmacc_vf_f16m1(_acc3, r0[3], _kernel, vl);
                        _acc4 = vfmacc_vf_f16m1(_acc4, r0[4], _kernel, vl);
                        _acc5 = vfmacc_vf_f16m1(_acc5, r0[5], _kernel, vl);
                        _acc6 = vfmacc_vf_f16m1(_acc6, r0[6], _kernel, vl);
                        _acc7 = vfmacc_vf_f16m1(_acc7, r0[7], _kernel, vl);
                        r0 += 8;
                    }

                    vse16_v_f16m1(output0_tm, _acc0, vl);
                    vse16_v_f16m1(output0_tm + packn * 1, _acc1, vl);
                    vse16_v_f16m1(output0_tm + packn * 2, _acc2, vl);
                    vse16_v_f16m1(output0_tm + packn * 3, _acc3, vl);
                    vse16_v_f16m1(output0_tm + packn * 4, _acc4, vl);
                    vse16_v_f16m1(output0_tm + packn * 5, _acc5, vl);
                    vse16_v_f16m1(output0_tm + packn * 6, _acc6, vl);
                    vse16_v_f16m1(output0_tm + packn * 7, _acc7, vl);
                    output0_tm += packn * 8;
                }
                for (; t + 3 < tiles; t += 4) {
                    __fp16 *r0 = img_tm2 + (t / 8 + (t % 8) / 4) * in_c * 8;
                    __fp16 *k0 = kernel0_tm + r * in_c * packn;

                    vfloat16m1_t _acc0 = vfmv_v_f_f16m1(0.0f, vl);
                    vfloat16m1_t _acc1 = vfmv_v_f_f16m1(0.0f, vl);
                    vfloat16m1_t _acc2 = vfmv_v_f_f16m1(0.0f, vl);
                    vfloat16m1_t _acc3 = vfmv_v_f_f16m1(0.0f, vl);

                    for (int c = 0; c < in_c; c++) {
                        vfloat16m1_t _kernel = vle16_v_f16m1(k0, vl);
                        k0 += packn;
                        _acc0 = vfmacc_vf_f16m1(_acc0, r0[0], _kernel, vl);
                        _acc1 = vfmacc_vf_f16m1(_acc1, r0[1], _kernel, vl);
                        _acc2 = vfmacc_vf_f16m1(_acc2, r0[2], _kernel, vl);
                        _acc3 = vfmacc_vf_f16m1(_acc3, r0[3], _kernel, vl);
                        r0 += 4;
                    }

                    vse16_v_f16m1(output0_tm, _acc0, vl);
                    vse16_v_f16m1(output0_tm + packn * 1, _acc1, vl);
                    vse16_v_f16m1(output0_tm + packn * 2, _acc2, vl);
                    vse16_v_f16m1(output0_tm + packn * 3, _acc3, vl);
                    output0_tm += packn * 4;
                }
                for (; t + 1 < tiles; t += 2) {
                    __fp16 *r0 = img_tm2 + (t / 8 + (t % 8) / 4 + (t % 4) / 2) * in_c * 8;
                    __fp16 *k0 = kernel0_tm + r * in_c * packn;

                    vfloat16m1_t _acc0 = vfmv_v_f_f16m1(0.0f, vl);
                    vfloat16m1_t _acc1 = vfmv_v_f_f16m1(0.0f, vl);

                    for (int c = 0; c < in_c; c++) {
                        vfloat16m1_t _kernel = vle16_v_f16m1(k0, vl);
                        k0 += packn;
                        _acc0 = vfmacc_vf_f16m1(_acc0, r0[0], _kernel, vl);
                        _acc1 = vfmacc_vf_f16m1(_acc1, r0[1], _kernel, vl);
                        r0 += 2;
                    }

                    vse16_v_f16m1(output0_tm, _acc0, vl);
                    vse16_v_f16m1(output0_tm + packn * 1, _acc1, vl);
                    output0_tm += packn * 2;
                }
                for (; t < tiles; t++) {
                    __fp16 *r0 = img_tm2 + (t / 8 + (t % 8) / 4 + (t % 4) / 2 + t % 2) * in_c * 8;
                    __fp16 *k0 = kernel0_tm + r * in_c * packn;

                    vfloat16m1_t _acc0 = vfmv_v_f_f16m1(0.0f, vl);

                    for (int c = 0; c < in_c; c++) {
                        vfloat16m1_t _kernel = vle16_v_f16m1(k0, vl);
                        k0 += packn;
                        _acc0 = vfmacc_vf_f16m1(_acc0, r0[0], _kernel, vl);
                        r0 += 1;
                    }

                    vse16_v_f16m1(output0_tm, _acc0, vl);
                    output0_tm += packn * 1;
                }
            }
        }

        csi_mem_free(input_tm2_buf);
        /*************************** transform output ****************************/
        // output_tm1_buf: [out_c/8, out_h6, out_w6, 8]
        __fp16 *output_tm1_buf =
            (__fp16 *)csi_mem_alloc(out_c * block_h * block_w * 6 * 6 * sizeof(__fp16));

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
            __fp16 *bias_tmp = bias_data + p * packn;

            __fp16 *out0_tm = output_dot_buf +
                              p * 64 * block_h * block_w * packn;  // 输出转换前/dot后 第p个channel
            __fp16 *out0 =
                output_tm1_buf + p * 6 * block_h * 6 * block_w * packn;  // 转换后输出 第p个channel

            __fp16 tmp[6][8][packn];

            for (int i = 0; i < block_h; i++) {
                for (int j = 0; j < block_w; j++) {
                    __fp16 *output0_tm_0 = out0_tm + (i * block_w + j) * packn;  // 8*8 起始地址
                    __fp16 *output0_tm_1 = output0_tm_0 + tiles * packn * 1;
                    __fp16 *output0_tm_2 = output0_tm_0 + tiles * packn * 2;
                    __fp16 *output0_tm_3 = output0_tm_0 + tiles * packn * 3;
                    __fp16 *output0_tm_4 = output0_tm_0 + tiles * packn * 4;
                    __fp16 *output0_tm_5 = output0_tm_0 + tiles * packn * 5;
                    __fp16 *output0_tm_6 = output0_tm_0 + tiles * packn * 6;
                    __fp16 *output0_tm_7 = output0_tm_0 + tiles * packn * 7;

                    __fp16 *output0 =
                        out0 + (i * block_w * 6 * 6 + j * 6) * packn;  // 输出 6*6 的起始地址

                    for (int m = 0; m < 8; m++) {
                        vfloat16m1_t _r00 = vle16_v_f16m1(output0_tm_0, vl);
                        vfloat16m1_t _r01 = vle16_v_f16m1(output0_tm_1, vl);
                        vfloat16m1_t _r02 = vle16_v_f16m1(output0_tm_2, vl);
                        vfloat16m1_t _r03 = vle16_v_f16m1(output0_tm_3, vl);
                        vfloat16m1_t _r04 = vle16_v_f16m1(output0_tm_4, vl);
                        vfloat16m1_t _r05 = vle16_v_f16m1(output0_tm_5, vl);
                        vfloat16m1_t _r06 = vle16_v_f16m1(output0_tm_6, vl);
                        vfloat16m1_t _r07 = vle16_v_f16m1(output0_tm_7, vl);

                        vfloat16m1_t _tmp024a = vfadd_vv_f16m1(_r01, _r02, vl);
                        vfloat16m1_t _tmp135a = vfsub_vv_f16m1(_r01, _r02, vl);

                        vfloat16m1_t _tmp024b = vfadd_vv_f16m1(_r03, _r04, vl);
                        vfloat16m1_t _tmp135b = vfsub_vv_f16m1(_r03, _r04, vl);

                        vfloat16m1_t _tmp024c = vfadd_vv_f16m1(_r05, _r06, vl);
                        vfloat16m1_t _tmp135c = vfsub_vv_f16m1(_r05, _r06, vl);

                        vfloat16m1_t _tmp0m =
                            vfadd_vv_f16m1(vfadd_vv_f16m1(_r00, _tmp024a, vl),
                                           vfmacc_vf_f16m1(_tmp024b, 32.f, _tmp024c, vl), vl);
                        vfloat16m1_t _tmp2m = vfmacc_vf_f16m1(
                            vfmacc_vf_f16m1(_tmp024a, 4.f, _tmp024b, vl), 8.f, _tmp024c, vl);
                        vfloat16m1_t _tmp4m = vfmacc_vf_f16m1(
                            vfmacc_vf_f16m1(_tmp024a, 16.f, _tmp024b, vl), 2.f, _tmp024c, vl);

                        vfloat16m1_t _tmp1m = vfmacc_vf_f16m1(
                            vfmacc_vf_f16m1(_tmp135a, 2.f, _tmp135b, vl), 16.f, _tmp135c, vl);
                        vfloat16m1_t _tmp3m = vfmacc_vf_f16m1(
                            vfmacc_vf_f16m1(_tmp135a, 8.f, _tmp135b, vl), 4.f, _tmp135c, vl);
                        vfloat16m1_t _tmp5m =
                            vfadd_vv_f16m1(vfadd_vv_f16m1(_r07, _tmp135a, vl),
                                           vfmacc_vf_f16m1(_tmp135c, 32.f, _tmp135b, vl), vl);

                        vse16_v_f16m1(tmp[0][m], _tmp0m, vl);
                        vse16_v_f16m1(tmp[2][m], _tmp2m, vl);
                        vse16_v_f16m1(tmp[4][m], _tmp4m, vl);
                        vse16_v_f16m1(tmp[1][m], _tmp1m, vl);
                        vse16_v_f16m1(tmp[3][m], _tmp3m, vl);
                        vse16_v_f16m1(tmp[5][m], _tmp5m, vl);

                        output0_tm_0 += tiles * packn * 8;
                        output0_tm_1 += tiles * packn * 8;
                        output0_tm_2 += tiles * packn * 8;
                        output0_tm_3 += tiles * packn * 8;
                        output0_tm_4 += tiles * packn * 8;
                        output0_tm_5 += tiles * packn * 8;
                        output0_tm_6 += tiles * packn * 8;
                        output0_tm_7 += tiles * packn * 8;
                    }

                    vfloat16m1_t _bias = vle16_v_f16m1(bias_tmp, vl);
                    for (int m = 0; m < 6; m++) {
                        vfloat16m1_t _tmp00 = vle16_v_f16m1(tmp[m][0], vl);
                        vfloat16m1_t _tmp01 = vle16_v_f16m1(tmp[m][1], vl);
                        vfloat16m1_t _tmp02 = vle16_v_f16m1(tmp[m][2], vl);
                        vfloat16m1_t _tmp03 = vle16_v_f16m1(tmp[m][3], vl);
                        vfloat16m1_t _tmp04 = vle16_v_f16m1(tmp[m][4], vl);
                        vfloat16m1_t _tmp05 = vle16_v_f16m1(tmp[m][5], vl);
                        vfloat16m1_t _tmp06 = vle16_v_f16m1(tmp[m][6], vl);
                        vfloat16m1_t _tmp07 = vle16_v_f16m1(tmp[m][7], vl);

                        vfloat16m1_t _tmp024a = vfadd_vv_f16m1(_tmp01, _tmp02, vl);
                        vfloat16m1_t _tmp135a = vfsub_vv_f16m1(_tmp01, _tmp02, vl);

                        vfloat16m1_t _tmp024b = vfadd_vv_f16m1(_tmp03, _tmp04, vl);
                        vfloat16m1_t _tmp135b = vfsub_vv_f16m1(_tmp03, _tmp04, vl);

                        vfloat16m1_t _tmp024c = vfadd_vv_f16m1(_tmp05, _tmp06, vl);
                        vfloat16m1_t _tmp135c = vfsub_vv_f16m1(_tmp05, _tmp06, vl);

                        vfloat16m1_t _output00 = vfadd_vv_f16m1(
                            _bias,
                            vfadd_vv_f16m1(vfadd_vv_f16m1(_tmp00, _tmp024a, vl),
                                           vfmacc_vf_f16m1(_tmp024b, 32.f, _tmp024c, vl), vl),
                            vl);
                        vfloat16m1_t _output02 = vfadd_vv_f16m1(
                            _bias,
                            vfmacc_vf_f16m1(vfmacc_vf_f16m1(_tmp024a, 4.f, _tmp024b, vl), 8.f,
                                            _tmp024c, vl),
                            vl);
                        vfloat16m1_t _output04 = vfadd_vv_f16m1(
                            _bias,
                            vfmacc_vf_f16m1(vfmacc_vf_f16m1(_tmp024a, 16.f, _tmp024b, vl), 2.f,
                                            _tmp024c, vl),
                            vl);

                        vfloat16m1_t _output01 = vfadd_vv_f16m1(
                            _bias,
                            vfmacc_vf_f16m1(vfmacc_vf_f16m1(_tmp135a, 2.f, _tmp135b, vl), 16.f,
                                            _tmp135c, vl),
                            vl);
                        vfloat16m1_t _output03 = vfadd_vv_f16m1(
                            _bias,
                            vfmacc_vf_f16m1(vfmacc_vf_f16m1(_tmp135a, 8.f, _tmp135b, vl), 4.f,
                                            _tmp135c, vl),
                            vl);
                        vfloat16m1_t _output05 = vfadd_vv_f16m1(
                            _bias,
                            vfadd_vv_f16m1(vfadd_vv_f16m1(_tmp07, _tmp135a, vl),
                                           vfmacc_vf_f16m1(_tmp135c, 32.f, _tmp135b, vl), vl),
                            vl);

                        vse16_v_f16m1(output0, _output00, vl);
                        vse16_v_f16m1(output0 + packn * 2, _output02, vl);
                        vse16_v_f16m1(output0 + packn * 4, _output04, vl);
                        vse16_v_f16m1(output0 + packn * 1, _output01, vl);
                        vse16_v_f16m1(output0 + packn * 3, _output03, vl);
                        vse16_v_f16m1(output0 + packn * 5, _output05, vl);

                        output0 += block_w * 6 * packn;
                    }
                }
            }
        }

        csi_mem_free(output_dot_buf);
        // crop the output after transform: cut extra part (right , bottom)
        winograd_crop_output_packnto1_fp16(output_tm1_buf, output_data, out_c, out_h, out_w,
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
