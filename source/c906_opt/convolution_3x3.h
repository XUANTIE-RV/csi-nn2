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


/*
    the conditions for using winograd convolution
    in_channel >= 16
    out_channel >= 16
    input_height <= 120
    input_width <= 120
*/

#include "sgemm.h"


/*  params:
    input:          origin input data
    input_padded:   input data after pad
    inc:            origin input channel
    inh:            origin input height
    inw:            origin input width
    padded_h:       input height after pad
    padded_w:       input width after pad
    pad_top:        origin pad top
    pad_left:       origin pad left
*/
static void pad_input(const float *input, float *input_padded, int inc, int inh, int inw,
                      int padded_h, int padded_w, int pad_top, int pad_left)
{
    int padded_hw = padded_h * padded_w;

    float *pad_ptr = NULL;
    float *inp_ptr = (float *)input;
    int resi_h = padded_h - pad_top - inh;  // remain to pad on h (pad_down)
    int resi_w = padded_w - pad_left - inw; // remain to pad on w (pad_right)
    for (int c = 0; c < inc; c++) {
        pad_ptr = input_padded + c * padded_hw;
        // pad h_top
        memset(pad_ptr, 0, padded_w * pad_top * sizeof(float));
        pad_ptr += pad_top * padded_w;
        // pad h_mid
        for (int h = 0; h < inh; h++) {
            // pad w_left
            memset(pad_ptr, 0, pad_left * sizeof(float));
            // pad w_mid
            memcpy(pad_ptr + pad_left, inp_ptr, inw * sizeof(float));
            // pad w_end
            memset(pad_ptr + pad_left + inw, 0, resi_w * sizeof(float));

            inp_ptr += inw;
            pad_ptr += padded_w;
        }
        // pad h_bottom
        memset(pad_ptr, 0, padded_w * resi_h * sizeof(float));
    }
}

/*  params:
    output_trans:   transflorm output after dot
    output：        final output data
    out_c:          final output channel
    out_h:          final output height
    out_w:          final output width
    wino_h:         winograd conv out_h, alignment with 2/4/6
    wino_w：        winograd conv out_w, alignment with 2/4/6
*/
static void crop_output(float *output_trans, float *output, int out_c, int out_h, int out_w,
                        int wino_h, int wino_w)
{
    int resi_h = wino_h - out_h;
    int resi_w = wino_w - out_w;
    float *out_ptr = output;
    for(int c = 0; c < out_c; c++) {

        float *crop_ptr = output_trans + c * wino_h * wino_w;

        for(int h = 0; h < out_h; h++) {
            memcpy(out_ptr, crop_ptr, out_w * sizeof(float));
            out_ptr += out_w;
            crop_ptr += wino_w;
        }
    }
}


static void conv3x3s1_winograd23_transform_kernel(struct csi_tensor *o_kernel,
                                                  struct csi_tensor *t_kernel)
{
    int32_t outch = o_kernel->dim[0];
    int32_t inch  = o_kernel->dim[1];

    float *kernel_data = (float *)o_kernel->data;
    // for kernel transform buf, 3x3 --> 4x4
    float *kernel_tm = (float *)malloc(outch * inch * 4 * 4 * sizeof(float));
    // kernel transform matrix: G
    const float ktm[4][3] = {
        {1, 0, 0},
        {0.5, 0.5, 0.5},
        {0.5, -0.5, 0.5},
        {0, 0, 1}
    };

    csi_tensor_copy(t_kernel, o_kernel);
    t_kernel->data = kernel_tm;

    for (int p = 0; p < outch; p++) {
        for (int q = 0; q < inch; q++) {

            const float* kernel0 = kernel_data + p * inch * 9 + q * 9;
            float* kernel_tm0 = kernel_tm + p * inch * 16 + q * 16;

            // transform kernel
            const float *k0 = kernel0;
            const float *k1 = kernel0 + 3;
            const float *k2 = kernel0 + 6;

            // h : first compute the transport matrix tmp = (g * GT)T  // tmp = G * gT
            float tmp[4][3];
            for (int i = 0; i < 4; i++) {

                tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
            }

            // U
            for (int j = 0; j < 4; j++) {
                float* tmpp = &tmp[j][0];

                for (int i = 0; i < 4; i++) {
                    kernel_tm0[i * 4 + j] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                }
            }
        }
    }
}

static int conv3x3s1_winograd23(struct csi_tensor *input,
                                struct csi_tensor *output,
                                struct csi_tensor *kernel,
                                struct csi_tensor *bias,
                                struct conv2d_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    float *kernel_data = (float *)kernel->data;
    float *bias_data = (float *)bias->data;

    // param
    int kernel_h = kernel->dim[2];
    int kernel_w = kernel->dim[3];
    int stride_h = params->stride_height;
    int stride_w = params->stride_width;
    int dilation_h = params->dilation_height;
    int dilation_w = params->dilation_width;
    int pad_left =  params->pad_left;
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
    int block_h = (out_h + 1) / 2;
    int block_w = (out_w + 1) / 2;

    int padded_in_h = block_h * 2 + 2;  // block * 2 for alignment with 2，kernel = 3 * 3 ，stride = 1，thus input_size + 2
    int padded_in_w = block_w * 2 + 2;
    int padded_in_hw = padded_in_h * padded_in_w;   // element size after padding per channel

    // buffer addr
    float *input_padd_buf = (float *)malloc(in_c * padded_in_hw * sizeof(float));
    float *input_trans_buf = (float *)malloc(in_c * block_h * block_w * 4 * 4 * sizeof(float));
    float *output_trans_buf = (float *)malloc(out_c * block_h * block_w * 2 * 2 * sizeof(float));

    for(int n = 0; n < batch; n++) {

        // pad input
        pad_input(input_data, input_padd_buf, in_c, in_h, in_w, padded_in_h, padded_in_w, pad_top, pad_left);
        input_data += input_size;

        // transform input
        /*
        BT = {
            { 1  0   -1  0 };
            { 0  1   1   0 };
            { 0  -1  1   0 };
            { 0  -1  0   1 }
        };
        */
        int in_h_tm = block_h * 4;  // input height after transform
        int in_w_tm = block_w * 4;

        const int tiles = block_h * block_w;

        for(int q = 0; q < in_c; q++) {

            const float *img0 = input_padd_buf + q * padded_in_h * padded_in_w;
            float *img0_tm = input_trans_buf + q * block_h * block_w * 4 * 4;

            float tmp[4][4];

            for(int i = 0; i < block_h; i++) {

                for(int j = 0; j < block_w; j++) {

                    const float *r0 = img0 + i * padded_in_w * 2 + j * 2;

                    for(int m = 0; m < 4; m++) {
                        tmp[0][m] = r0[0] - r0[2];
                        tmp[1][m] = r0[1] + r0[2];
                        tmp[2][m] = r0[2] - r0[1];
                        tmp[3][m] = r0[3] - r0[1];
                        r0 += padded_in_w;
                    }

                    float *r0_tm_0 = img0_tm + i * in_w_tm * 4 + j * 4;
                    float *r0_tm_1 = r0_tm_0 + in_w_tm;
                    float *r0_tm_2 = r0_tm_1 + in_w_tm;
                    float *r0_tm_3 = r0_tm_2 + in_w_tm;

                    for(int m = 0; m < 4; m++) {

                        const float *tmp0 = tmp[m];
                        r0_tm_0[m] = tmp0[0] - tmp0[2];
                        r0_tm_1[m] = tmp0[1] + tmp0[2];
                        r0_tm_2[m] = tmp0[2] - tmp0[1];
                        r0_tm_3[m] = tmp0[3] - tmp0[1];
                    }
                }
            }
        }

        // dot
        float *output_dot_buf = (float *)calloc(out_c * block_h * block_w * 4 * 4, sizeof(float));

        for(int i = 0; i < out_c; i++) {
            for(int j = 0; j < block_h; j++) {
                for(int k = 0; k < block_w; k++) {
                    float *input_0 = input_trans_buf + j * 4 * 4 * block_w + k * 4;
                    float *input_1 = input_0 + block_w * 4;
                    float *input_2 = input_1 + block_w * 4;
                    float *input_3 = input_2 + block_w * 4;

                    float *kernel_0 = kernel_data + i * in_c * 16;
                    float *kernel_1 = kernel_0 + 4;
                    float *kernel_2 = kernel_1 + 4;
                    float *kernel_3 = kernel_2 + 4;

                    float *output_0 = output_dot_buf + i * block_h * block_w * 16 + j * 16 * block_w + k * 4;
                    float *output_1 = output_0 + block_w * 4;
                    float *output_2 = output_1 + block_w * 4;
                    float *output_3 = output_2 + block_w * 4;

                    for(int a = 0; a < in_c; a++) {
                        output_0[0] += input_0[0] * kernel_0[0];
                        output_0[1] += input_0[1] * kernel_0[1];
                        output_0[2] += input_0[2] * kernel_0[2];
                        output_0[3] += input_0[3] * kernel_0[3];

                        output_1[0] += input_1[0] * kernel_1[0];
                        output_1[1] += input_1[1] * kernel_1[1];
                        output_1[2] += input_1[2] * kernel_1[2];
                        output_1[3] += input_1[3] * kernel_1[3];

                        output_2[0] += input_2[0] * kernel_2[0];
                        output_2[1] += input_2[1] * kernel_2[1];
                        output_2[2] += input_2[2] * kernel_2[2];
                        output_2[3] += input_2[3] * kernel_2[3];

                        output_3[0] += input_3[0] * kernel_3[0];
                        output_3[1] += input_3[1] * kernel_3[1];
                        output_3[2] += input_3[2] * kernel_3[2];
                        output_3[3] += input_3[3] * kernel_3[3];

                        input_0 += block_h * block_w * 16;
                        input_1 += block_h * block_w * 16;
                        input_2 += block_h * block_w * 16;
                        input_3 += block_h * block_w * 16;

                        kernel_0 += 16;
                        kernel_1 += 16;
                        kernel_2 += 16;
                        kernel_3 += 16;
                    }
                }
            }
        }

        // transform output
        /*
        AT = {
            { 1  1  1   0 };
            { 0  1  -1  1 }
        };
        */
        for(int i = 0; i < out_c; i++) {

            const float bias = bias_data ? bias_data[i] : 0.f;
            const float *img1 = output_dot_buf + i * block_h * block_w * 4 * 4;
            float *img1_tm = output_trans_buf + i * block_h * block_w * 2 * 2;

            float tmp[2][4];
            for(int j = 0; j < block_h; j++) {
                for(int k = 0; k < block_w; k++) {
                    const float *r1 = img1 + j * block_w * 4 * 4 + k * 4;

                    for(int m = 0; m < 4; m++) {
                        tmp[0][m] = r1[0] + r1[1] + r1[2];
                        tmp[1][m] = r1[1] - r1[2] + r1[3];
                        r1 += block_w * 4;
                    }
                    float *r1_tm_0 = img1_tm + j * block_w * 2 * 2 + k * 2;
                    float *r1_tm_1 = r1_tm_0 + block_w * 2;

                    for(int m = 0; m < 2; m++) {
                        const float *tmp1 = tmp[m];
                        r1_tm_0[m] = tmp1[0] + tmp1[1] + tmp1[2] + bias;
                        r1_tm_1[m] = tmp1[1] - tmp1[2] + tmp1[3] + bias;
                    }
                }
            }
        }
        free(output_dot_buf);
        // crop the output after transform: cut extra part (right , bottom)
        crop_output(output_trans_buf, output_data, out_c, out_h, out_w, block_h * 2, block_w * 2);
        output_data += output_size;
    }
    free(input_padd_buf);
    free(input_trans_buf);
    free(output_trans_buf);
    return CSINN_TRUE;
}



static void conv3x3s1_winograd43_transform_kernel(struct csi_tensor *o_kernel,
                                                  struct csi_tensor *t_kernel)
{
    int32_t outch = o_kernel->dim[0];
    int32_t inch  = o_kernel->dim[1];

    float *kernel_data = (float *)o_kernel->data;
    // for kernel transform buf, 3x3 --> 6x6
    float *kernel_tm = (float *)malloc(outch * inch * 6 * 6 * sizeof(float));

    // kernel transform matrix: G
    const float ktm[6][3] = {
        {  1.0f/4,     0.0f,    0.0f},
        { -1.0f/6,  -1.0f/6, -1.0f/6},
        { -1.0f/6,   1.0f/6, -1.0f/6},
        { 1.0f/24,  1.0f/12,  1.0f/6},
        { 1.0f/24, -1.0f/12,  1.0f/6},
        {    0.0f,     0.0f,    1.0f}
    };

    csi_tensor_copy(t_kernel, o_kernel);
    t_kernel->data = kernel_tm;

    for (int p = 0; p < outch; p++) {
        for (int q = 0; q < inch; q++) {

            const float* kernel0 = kernel_data + p * inch * 9 + q * 9;
            float* kernel_tm0 = kernel_tm + p * inch * 36 + q * 36;

            // transform kernel
            const float *k0 = kernel0;
            const float *k1 = kernel0 + 3;
            const float *k2 = kernel0 + 6;

            // h : first compute the transport matrix tmp = (g * GT)T
            float tmp[6][3];
            for (int i = 0; i < 6; i++) {

                tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
            }

            // U
            for (int j = 0; j < 6; j++) {
                float* tmpp = &tmp[j][0];

                for (int i = 0; i < 6; i++) {
                    kernel_tm0[i * 6 + j] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                }
            }
        }
    }

}

static int conv3x3s1_winograd43(struct csi_tensor *input,
                                struct csi_tensor *output,
                                struct csi_tensor *kernel,
                                struct csi_tensor *bias,
                                struct conv2d_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    float *kernel_data = (float *)kernel->data;
    float *bias_data = (float *)bias->data;

    // param
    int kernel_h = kernel->dim[2];
    int kernel_w = kernel->dim[3];
    int stride_h = params->stride_height;
    int stride_w = params->stride_width;
    int dilation_h = params->dilation_height;
    int dilation_w = params->dilation_width;
    int pad_left =  params->pad_left;
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
    int block_h = (out_h + 3) / 4;
    int block_w = (out_w + 3) / 4;

    int padded_in_h = block_h * 4 + 2;  // block * 4 for alignment with 4，kernel = 3 * 3 ，stride = 1，thus input_size + 2
    int padded_in_w = block_w * 4 + 2;
    int padded_in_hw = padded_in_h * padded_in_w;   // element size after padding per channel

    // buffer addr
    float *input_padd_buf = (float *)malloc(in_c * padded_in_hw * sizeof(float));
    float *input_trans_buf = (float *)malloc(in_c * block_h * block_w * 6 * 6 * sizeof(float));
    float *output_trans_buf = (float *)malloc(out_c * block_h * block_w * 4 * 4 * sizeof(float));

    for(int n = 0; n < batch; n++) {

        // pad input
        pad_input(input_data, input_padd_buf, in_c, in_h, in_w, padded_in_h, padded_in_w, pad_top, pad_left);
        input_data += input_size;

        // transform input
        /*
        BT = {
            { 4  0   -5  0   1  0 };
            { 0  -4  -4  1   1  0 };
            { 0  4   -4  -1  1  0 };
            { 0  -2  -1  2   1  0 };
            { 0  2   -1  -2  1  0 };
            { 0  4   0   -5  0  1 }
        };
        */
        int in_h_tm = block_h * 6;  // input height after transform
        int in_w_tm = block_w * 6;

        const int tiles = block_h * block_w;

        for(int q = 0; q < in_c; q++) {

            const float *img0 = input_padd_buf + q * padded_in_h * padded_in_w;
            float *img0_tm = input_trans_buf + q * block_h * block_w * 6 * 6;

            float tmp[6][6];

            for(int i = 0; i < block_h; i++) {

                for(int j = 0; j < block_w; j++) {

                    const float *r0 = img0 + i * padded_in_w * 4 + j * 4;

                    for(int m = 0; m < 6; m++) {
                        tmp[0][m] = 4 * r0[0] - 5 * r0[2] + r0[4];
                        tmp[1][m] = r0[3] + r0[4] - 4 * r0[1] - 4 * r0[2];
                        tmp[2][m] = 4 * r0[1] + r0[4] - 4 * r0[2] - r0[3];
                        tmp[3][m] = 2 * r0[3] + r0[4] - 2 * r0[1] - r0[2];
                        tmp[4][m] = 2 * r0[1] + r0[4] - 2 * r0[3] - r0[2];
                        tmp[5][m] = 4 * r0[1] - 5 * r0[3] + r0[5];
                        r0 += padded_in_w;
                    }

                    float *r0_tm_0 = img0_tm + i * in_w_tm * 6 + j * 6;
                    float *r0_tm_1 = r0_tm_0 + in_w_tm;
                    float *r0_tm_2 = r0_tm_1 + in_w_tm;
                    float *r0_tm_3 = r0_tm_2 + in_w_tm;
                    float *r0_tm_4 = r0_tm_3 + in_w_tm;
                    float *r0_tm_5 = r0_tm_4 + in_w_tm;

                    for(int m = 0; m < 6; m++) {

                        const float *tmp0 = tmp[m];
                        r0_tm_0[m] = 4 * tmp0[0] - 5 * tmp0[2] + tmp0[4];
                        r0_tm_1[m] = tmp0[3] + tmp0[4] - 4 * tmp0[1] - 4 * tmp0[2];
                        r0_tm_2[m] = 4 * tmp0[1] + tmp0[4] - 4 * tmp0[2] - tmp0[3];
                        r0_tm_3[m] = 2 * tmp0[3] + tmp0[4] - 2 * tmp0[1] - tmp0[2];
                        r0_tm_4[m] = 2 * tmp0[1] + tmp0[4] - 2 * tmp0[3] - tmp0[2];
                        r0_tm_5[m] = 4 * tmp0[1] - 5 * tmp0[3] + tmp0[5];
                    }
                }
            }
        }

        // dot
        float *output_dot_buf = (float *)calloc(out_c * block_h * block_w * 6 * 6, sizeof(float));

        for(int i = 0; i < out_c; i++) {
            for(int j = 0; j < block_h; j++) {
                for(int k = 0; k < block_w; k++) {
                    float *input_0 = input_trans_buf + j * 6 * 6 * block_w + k * 6;
                    float *input_1 = input_0 + block_w * 6;
                    float *input_2 = input_1 + block_w * 6;
                    float *input_3 = input_2 + block_w * 6;
                    float *input_4 = input_3 + block_w * 6;
                    float *input_5 = input_4 + block_w * 6;

                    float *kernel_0 = kernel_data + i * in_c * 36;
                    float *kernel_1 = kernel_0 + 6;
                    float *kernel_2 = kernel_1 + 6;
                    float *kernel_3 = kernel_2 + 6;
                    float *kernel_4 = kernel_3 + 6;
                    float *kernel_5 = kernel_4 + 6;

                    float *output_0 = output_dot_buf + i * block_h * block_w * 36 + j * 36 * block_w + k * 6;
                    float *output_1 = output_0 + block_w * 6;
                    float *output_2 = output_1 + block_w * 6;
                    float *output_3 = output_2 + block_w * 6;
                    float *output_4 = output_3 + block_w * 6;
                    float *output_5 = output_4 + block_w * 6;

                    for(int a = 0; a < in_c; a++) {
                        output_0[0] += input_0[0] * kernel_0[0];
                        output_0[1] += input_0[1] * kernel_0[1];
                        output_0[2] += input_0[2] * kernel_0[2];
                        output_0[3] += input_0[3] * kernel_0[3];
                        output_0[4] += input_0[4] * kernel_0[4];
                        output_0[5] += input_0[5] * kernel_0[5];

                        output_1[0] += input_1[0] * kernel_1[0];
                        output_1[1] += input_1[1] * kernel_1[1];
                        output_1[2] += input_1[2] * kernel_1[2];
                        output_1[3] += input_1[3] * kernel_1[3];
                        output_1[4] += input_1[4] * kernel_1[4];
                        output_1[5] += input_1[5] * kernel_1[5];

                        output_2[0] += input_2[0] * kernel_2[0];
                        output_2[1] += input_2[1] * kernel_2[1];
                        output_2[2] += input_2[2] * kernel_2[2];
                        output_2[3] += input_2[3] * kernel_2[3];
                        output_2[4] += input_2[4] * kernel_2[4];
                        output_2[5] += input_2[5] * kernel_2[5];

                        output_3[0] += input_3[0] * kernel_3[0];
                        output_3[1] += input_3[1] * kernel_3[1];
                        output_3[2] += input_3[2] * kernel_3[2];
                        output_3[3] += input_3[3] * kernel_3[3];
                        output_3[4] += input_3[4] * kernel_3[4];
                        output_3[5] += input_3[5] * kernel_3[5];

                        output_4[0] += input_4[0] * kernel_4[0];
                        output_4[1] += input_4[1] * kernel_4[1];
                        output_4[2] += input_4[2] * kernel_4[2];
                        output_4[3] += input_4[3] * kernel_4[3];
                        output_4[4] += input_4[4] * kernel_4[4];
                        output_4[5] += input_4[5] * kernel_4[5];

                        output_5[0] += input_5[0] * kernel_5[0];
                        output_5[1] += input_5[1] * kernel_5[1];
                        output_5[2] += input_5[2] * kernel_5[2];
                        output_5[3] += input_5[3] * kernel_5[3];
                        output_5[4] += input_5[4] * kernel_5[4];
                        output_5[5] += input_5[5] * kernel_5[5];

                        input_0 += block_h * block_w * 36;
                        input_1 += block_h * block_w * 36;
                        input_2 += block_h * block_w * 36;
                        input_3 += block_h * block_w * 36;
                        input_4 += block_h * block_w * 36;
                        input_5 += block_h * block_w * 36;

                        kernel_0 += 36;
                        kernel_1 += 36;
                        kernel_2 += 36;
                        kernel_3 += 36;
                        kernel_4 += 36;
                        kernel_5 += 36;
                    }
                }
            }
        }

        // transform output
        /*
        AT = {
            { 1  1  1   1  1   0 },
            { 0  1  -1  2  -2  0 },
            { 0  1  1   4  4   0 },
            { 0  1  -1  8  -8  1 }
        };
        */
        for(int i = 0; i < out_c; i++) {

            const float bias = bias_data ? bias_data[i] : 0.f;
            const float *img1 = output_dot_buf + i * block_h * block_w * 6 * 6;
            float *img1_tm = output_trans_buf + i * block_h * block_w * 4 * 4;

            float tmp[4][6];
            for(int j = 0; j < block_h; j++) {
                for(int k = 0; k < block_w; k++) {
                    const float *r1 = img1 + j * block_w * 6 * 6 + k * 6;

                    for(int m = 0; m < 6; m++) {
                        tmp[0][m] = r1[0] + r1[1] + r1[2] + r1[3] + r1[4];
                        tmp[1][m] = r1[1] - r1[2] + 2 * r1[3] - 2 * r1[4];
                        tmp[2][m] = r1[1] + r1[2] + 4 * r1[3] + 4 * r1[4];
                        tmp[3][m] = r1[1] - r1[2] + 8 * r1[3] - 8 * r1[4] + r1[5];
                        r1 += block_w * 6;
                    }
                    float *r1_tm_0 = img1_tm + j * block_w * 4 * 4 + k * 4;
                    float *r1_tm_1 = r1_tm_0 + block_w * 4;
                    float *r1_tm_2 = r1_tm_1 + block_w * 4;
                    float *r1_tm_3 = r1_tm_2 + block_w * 4;

                    for(int m = 0; m < 4; m++) {
                        const float *tmp1 = tmp[m];
                        r1_tm_0[m] = tmp1[0] + tmp1[1] + tmp1[2] + tmp1[3] + tmp1[4] + bias;
                        r1_tm_1[m] = tmp1[1] - tmp1[2] + 2 * tmp1[3] - 2 * tmp1[4] + bias;
                        r1_tm_2[m] = tmp1[1] + tmp1[2] + 4 * tmp1[3] + 4 * tmp1[4] + bias;
                        r1_tm_3[m] = tmp1[1] - tmp1[2] + 8 * tmp1[3] - 8 * tmp1[4] + tmp1[5] + bias;
                    }
                }
            }
        }
        free(output_dot_buf);
        // crop the output after transform: cut extra part (right , bottom)
        crop_output(output_trans_buf, output_data, out_c, out_h, out_w, block_h * 4, block_w * 4);
        output_data += output_size;
    }
    free(input_padd_buf);
    free(input_trans_buf);
    free(output_trans_buf);
    return CSINN_TRUE;
}


static void conv3x3s1_winograd64_transform_kernel(struct csi_tensor *o_kernel,
                                                  struct csi_tensor *t_kernel)
{
    int32_t outch = o_kernel->dim[0];
    int32_t inch  = o_kernel->dim[1];

    float *kernel_data = (float *)o_kernel->data;
    // for kernel transform buf, 3x3 --> 8x8
    float *kernel_tm = (float *)malloc(outch * inch * 8 * 8 * sizeof(float));
    // kernel transform matrix: G
    const float ktm[8][3] = {
        {1.0f, 0.0f, 0.0f},
        {-2.0f / 9, -2.0f / 9, -2.0f / 9},
        {-2.0f / 9, 2.0f / 9, -2.0f / 9},
        {1.0f / 90, 1.0f / 45, 2.0f / 45},
        {1.0f / 90, -1.0f / 45, 2.0f / 45},
        {1.0f / 45, 1.0f / 90, 1.0f / 180},
        {1.0f / 45, -1.0f / 90, 1.0f / 180},
        {0.0f, 0.0f, 1.0f}
    };

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
    t_kernel->data = kernel_tm;

    for (int p = 0; p < outch; p++) {
        for (int q = 0; q < inch; q++) {

            const float* kernel0 = kernel_data + p * inch * 9 + q * 9;
            float* kernel_tm0 = kernel_tm + p * inch * 64 + q * 64;

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
                float* tmpp = &tmp[j][0];

                for (int i = 0; i < 8; i++) {
                    kernel_tm0[i * 8 + j] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                }
            }
        }
    }

}

static int conv3x3s1_winograd64(struct csi_tensor *input,
                                struct csi_tensor *output,
                                struct csi_tensor *kernel,
                                struct csi_tensor *bias,
                                struct conv2d_params *params)
{

    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    float *kernel_data = (float *)kernel->data;
    float *bias_data = (float *)bias->data;

    // param
    int kernel_h = kernel->dim[2];
    int kernel_w = kernel->dim[3];
    int stride_h = params->stride_height;
    int stride_w = params->stride_width;
    int dilation_h = params->dilation_height;
    int dilation_w = params->dilation_width;
    int pad_left =  params->pad_left;
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

    int padded_in_h = block_h * 6 + 2;  // block * 4 for alignment with 4，kernel = 3 * 3 ，stride = 1，thus input_size + 2
    int padded_in_w = block_w * 6 + 2;
    int padded_in_hw = padded_in_h * padded_in_w;   // element size after padding per channel

    // buffer addr
    float *input_padd_buf = (float *)malloc(in_c * padded_in_hw * sizeof(float));
    float *input_trans_buf = (float *)malloc(in_c * block_h * block_w * 8 * 8 * sizeof(float));
    float *output_trans_buf = (float *)malloc(out_c * block_h * block_w * 6 * 6 * sizeof(float));

    for(int n = 0; n < batch; n++) {

        // pad input
        pad_input(input_data, input_padd_buf, in_c, in_h, in_w, padded_in_h, padded_in_w, pad_top, pad_left);
        input_data += input_size;

        // transform input
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
        int in_h_tm = block_h * 8;  // input height after transform
        int in_w_tm = block_w * 8;

        const int tiles = block_h * block_w;

        for(int q = 0; q < in_c; q++) {

            const float *img0 = input_padd_buf + q * padded_in_h * padded_in_w;
            float *img0_tm = input_trans_buf + q * block_h * block_w * 8 * 8;

            float tmp[8][8];

            for(int i = 0; i < block_h; i++) {

                for(int j = 0; j < block_w; j++) {

                    const float *r0 = img0 + i * padded_in_w * 6 + j * 6;

                    for(int m = 0; m < 8; m++) {
                        tmp[0][m] = r0[0] - r0[6] + 5.25 * (r0[4] - r0[2]);
                        tmp[7][m] = r0[7] - r0[1] + 5.25 * (r0[3] - r0[5]);

                        float tmp12a = (r0[2] + r0[6] - r0[4] * 4.25f);
                        float tmp12b = (r0[1] + r0[5] - r0[3] * 4.25f);
                        tmp[1][m] = tmp12a + tmp12b;
                        tmp[2][m] = tmp12a - tmp12b;

                        float tmp34a = (r0[6] + r0[2] * 0.25f - r0[4] * 1.25f);
                        float tmp34b = (r0[1] * 0.5f - r0[3] * 2.5f + r0[5] * 2.f);
                        tmp[3][m] = tmp34a + tmp34b;
                        tmp[4][m] = tmp34a - tmp34b;

                        float tmp56a = (r0[6] + (r0[2] - r0[4] * 1.25f) * 4.f);
                        float tmp56b = (r0[1] * 2.f - r0[3] * 2.5f + r0[5] * 0.5f);
                        tmp[5][m] = tmp56a + tmp56b;
                        tmp[6][m] = tmp56a - tmp56b;

                        // tmp[0][m] = r0[0] - r0[6] + 5.25 * (r0[4] - r0[2]);
                        // tmp[1][m] = r0[1] + r0[2] + r0[5] + r0[6] - 4.25 * (r0[3] + r0[4]);
                        // tmp[2][m] = r0[2] - r0[1] + r0[6] - r0[5] + 4.25 * (r0[3] - r0[4]);
                        // tmp[3][m] = 0.5 * r0[1] + 0.25 * r0[2] - 2.5 * r0[3] - 1.25 * r0[4] + 2 * r0[5] + r0[6];
                        // tmp[4][m] = 0.25 * r0[2] - 0.5 * r0[1] + 2.5 * r0[3] - 1.25 * r0[4] - 2 * r0[5] + r0[6];
                        // tmp[5][m] = 2 * r0[1] + 4 * r0[2] - 2.5 * r0[3] - 5 * r0[4] + 0.5 * r0[5] + r0[6];
                        // tmp[6][m] = 4 * r0[2] - 2 * r0[1] + 2.5 * r0[3] - 5 * r0[4] - 0.5 * r0[5] + r0[6];
                        // tmp[7][m] = r0[7] - r0[1] + 5.25 * (r0[3] - r0[5]);

                        r0 += padded_in_w;
                    }

                    float *r0_tm_0 = img0_tm + i * in_w_tm * 8 + j * 8;
                    float *r0_tm_1 = r0_tm_0 + in_w_tm;
                    float *r0_tm_2 = r0_tm_1 + in_w_tm;
                    float *r0_tm_3 = r0_tm_2 + in_w_tm;
                    float *r0_tm_4 = r0_tm_3 + in_w_tm;
                    float *r0_tm_5 = r0_tm_4 + in_w_tm;
                    float *r0_tm_6 = r0_tm_5 + in_w_tm;
                    float *r0_tm_7 = r0_tm_6 + in_w_tm;

                    for(int m = 0; m < 8; m++) {

                        const float *tmp0 = tmp[m];

                        r0_tm_0[m] = tmp0[0] - tmp0[6] + 5.25 * (tmp0[4] - tmp0[2]);
                        r0_tm_7[m] = tmp0[7] - tmp0[1] + 5.25 * (tmp0[3] - tmp0[5]);

                        float tmp12a = (tmp0[2] + tmp0[6] - tmp0[4] * 4.25f);
                        float tmp12b = (tmp0[1] + tmp0[5] - tmp0[3] * 4.25f);
                        r0_tm_1[m] = tmp12a + tmp12b;
                        r0_tm_2[m] = tmp12a - tmp12b;

                        float tmp34a = (tmp0[6] + tmp0[2] * 0.25f - tmp0[4] * 1.25f);
                        float tmp34b = (tmp0[1] * 0.5f - tmp0[3] * 2.5f + tmp0[5] * 2.f);
                        r0_tm_3[m] = tmp34a + tmp34b;
                        r0_tm_4[m] = tmp34a - tmp34b;

                        float tmp56a = (tmp0[6] + (tmp0[2] - tmp0[4] * 1.25f) * 4.f);
                        float tmp56b = (tmp0[1] * 2.f - tmp0[3] * 2.5f + tmp0[5] * 0.5f);
                        r0_tm_5[m] = tmp56a + tmp56b;
                        r0_tm_6[m] = tmp56a - tmp56b;


                        // r0_tm_0[m] = tmp0[0] - tmp0[6] + 5.25 * (tmp0[4] - tmp0[2]);
                        // r0_tm_1[m] = tmp0[1] + tmp0[2] + tmp0[5] + tmp0[6] - 4.25 * (tmp0[3] + tmp0[4]);
                        // r0_tm_2[m] = tmp0[2] - tmp0[1] + tmp0[6] - tmp0[5] + 4.25 * (tmp0[3] - tmp0[4]);
                        // r0_tm_3[m] = 0.5 * tmp0[1] + 0.25 * tmp0[2] - 2.5 * tmp0[3] - 1.25 * tmp0[4] + 2 * tmp0[5] + tmp0[6];
                        // r0_tm_4[m] = 0.25 * tmp0[2] - 0.5 * tmp0[1] + 2.5 * tmp0[3] - 1.25 * tmp0[4] - 2 * tmp0[5] + tmp0[6];
                        // r0_tm_5[m] = 2 * tmp0[1] + 4 * tmp0[2] - 2.5 * tmp0[3] - 5 * tmp0[4] + 0.5 * tmp0[5] + tmp0[6];
                        // r0_tm_6[m] = 4 * tmp0[2] - 2 * tmp0[1] + 2.5 * tmp0[3] - 5 * tmp0[4] - 0.5 * tmp0[5] + tmp0[6];
                        // r0_tm_7[m] = tmp0[7] - tmp0[1] + 5.25 * (tmp0[3] - tmp0[5]);

                    }
                }
            }
        }

        // dot
        float *output_dot_buf = (float *)calloc(out_c * block_h * block_w * 8 * 8, sizeof(float));

        for(int i = 0; i < out_c; i++) {
            for(int j = 0; j < block_h; j++) {
                for(int k = 0; k < block_w; k++) {
                    float *input_0 = input_trans_buf + j * 8 * 8 * block_w + k * 8;
                    float *input_1 = input_0 + block_w * 8;
                    float *input_2 = input_1 + block_w * 8;
                    float *input_3 = input_2 + block_w * 8;
                    float *input_4 = input_3 + block_w * 8;
                    float *input_5 = input_4 + block_w * 8;
                    float *input_6 = input_5 + block_w * 8;
                    float *input_7 = input_6 + block_w * 8;

                    float *kernel_0 = kernel_data + i * in_c * 64;
                    float *kernel_1 = kernel_0 + 8;
                    float *kernel_2 = kernel_1 + 8;
                    float *kernel_3 = kernel_2 + 8;
                    float *kernel_4 = kernel_3 + 8;
                    float *kernel_5 = kernel_4 + 8;
                    float *kernel_6 = kernel_5 + 8;
                    float *kernel_7 = kernel_6 + 8;

                    float *output_0 = output_dot_buf + i * block_h * block_w * 64 + j * 64 * block_w + k * 8;
                    float *output_1 = output_0 + block_w * 8;
                    float *output_2 = output_1 + block_w * 8;
                    float *output_3 = output_2 + block_w * 8;
                    float *output_4 = output_3 + block_w * 8;
                    float *output_5 = output_4 + block_w * 8;
                    float *output_6 = output_5 + block_w * 8;
                    float *output_7 = output_6 + block_w * 8;

                    for(int a = 0; a < in_c; a++) {
                        output_0[0] += input_0[0] * kernel_0[0];
                        output_0[1] += input_0[1] * kernel_0[1];
                        output_0[2] += input_0[2] * kernel_0[2];
                        output_0[3] += input_0[3] * kernel_0[3];
                        output_0[4] += input_0[4] * kernel_0[4];
                        output_0[5] += input_0[5] * kernel_0[5];
                        output_0[6] += input_0[6] * kernel_0[6];
                        output_0[7] += input_0[7] * kernel_0[7];

                        output_1[0] += input_1[0] * kernel_1[0];
                        output_1[1] += input_1[1] * kernel_1[1];
                        output_1[2] += input_1[2] * kernel_1[2];
                        output_1[3] += input_1[3] * kernel_1[3];
                        output_1[4] += input_1[4] * kernel_1[4];
                        output_1[5] += input_1[5] * kernel_1[5];
                        output_1[6] += input_1[6] * kernel_1[6];
                        output_1[7] += input_1[7] * kernel_1[7];

                        output_2[0] += input_2[0] * kernel_2[0];
                        output_2[1] += input_2[1] * kernel_2[1];
                        output_2[2] += input_2[2] * kernel_2[2];
                        output_2[3] += input_2[3] * kernel_2[3];
                        output_2[4] += input_2[4] * kernel_2[4];
                        output_2[5] += input_2[5] * kernel_2[5];
                        output_2[6] += input_2[6] * kernel_2[6];
                        output_2[7] += input_2[7] * kernel_2[7];

                        output_3[0] += input_3[0] * kernel_3[0];
                        output_3[1] += input_3[1] * kernel_3[1];
                        output_3[2] += input_3[2] * kernel_3[2];
                        output_3[3] += input_3[3] * kernel_3[3];
                        output_3[4] += input_3[4] * kernel_3[4];
                        output_3[5] += input_3[5] * kernel_3[5];
                        output_3[6] += input_3[6] * kernel_3[6];
                        output_3[7] += input_3[7] * kernel_3[7];

                        output_4[0] += input_4[0] * kernel_4[0];
                        output_4[1] += input_4[1] * kernel_4[1];
                        output_4[2] += input_4[2] * kernel_4[2];
                        output_4[3] += input_4[3] * kernel_4[3];
                        output_4[4] += input_4[4] * kernel_4[4];
                        output_4[5] += input_4[5] * kernel_4[5];
                        output_4[6] += input_4[6] * kernel_4[6];
                        output_4[7] += input_4[7] * kernel_4[7];

                        output_5[0] += input_5[0] * kernel_5[0];
                        output_5[1] += input_5[1] * kernel_5[1];
                        output_5[2] += input_5[2] * kernel_5[2];
                        output_5[3] += input_5[3] * kernel_5[3];
                        output_5[4] += input_5[4] * kernel_5[4];
                        output_5[5] += input_5[5] * kernel_5[5];
                        output_5[6] += input_5[6] * kernel_5[6];
                        output_5[7] += input_5[7] * kernel_5[7];

                        output_6[0] += input_6[0] * kernel_6[0];
                        output_6[1] += input_6[1] * kernel_6[1];
                        output_6[2] += input_6[2] * kernel_6[2];
                        output_6[3] += input_6[3] * kernel_6[3];
                        output_6[4] += input_6[4] * kernel_6[4];
                        output_6[5] += input_6[5] * kernel_6[5];
                        output_6[6] += input_6[6] * kernel_6[6];
                        output_6[7] += input_6[7] * kernel_6[7];

                        output_7[0] += input_7[0] * kernel_7[0];
                        output_7[1] += input_7[1] * kernel_7[1];
                        output_7[2] += input_7[2] * kernel_7[2];
                        output_7[3] += input_7[3] * kernel_7[3];
                        output_7[4] += input_7[4] * kernel_7[4];
                        output_7[5] += input_7[5] * kernel_7[5];
                        output_7[6] += input_7[6] * kernel_7[6];
                        output_7[7] += input_7[7] * kernel_7[7];

                        input_0 += block_h * block_w * 64;
                        input_1 += block_h * block_w * 64;
                        input_2 += block_h * block_w * 64;
                        input_3 += block_h * block_w * 64;
                        input_4 += block_h * block_w * 64;
                        input_5 += block_h * block_w * 64;
                        input_6 += block_h * block_w * 64;
                        input_7 += block_h * block_w * 64;

                        kernel_0 += 64;
                        kernel_1 += 64;
                        kernel_2 += 64;
                        kernel_3 += 64;
                        kernel_4 += 64;
                        kernel_5 += 64;
                        kernel_6 += 64;
                        kernel_7 += 64;
                    }
                }
            }
        }

        // transform output
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
        for(int i = 0; i < out_c; i++) {

            const float bias = bias_data ? bias_data[i] : 0.f;
            const float *img1 = output_dot_buf + i * block_h * block_w * 8 * 8;
            float *img1_tm = output_trans_buf + i * block_h * block_w * 6 * 6;

            float tmp[6][8];
            for(int j = 0; j < block_h; j++) {
                for(int k = 0; k < block_w; k++) {
                    const float *r1 = img1 + j * block_w * 8 * 8 + k * 8;

                    for(int m = 0; m < 8; m++) {
                        float tmp024a = r1[1] + r1[2];
                        float tmp135a = r1[1] - r1[2];

                        float tmp024b = r1[3] + r1[4];
                        float tmp135b = r1[3] - r1[4];

                        float tmp024c = r1[5] + r1[6];
                        float tmp135c = r1[5] - r1[6];

                        tmp[0][m] = r1[0] + tmp024a + tmp024b + tmp024c * 32;
                        tmp[2][m] = tmp024a + tmp024b * 4 + tmp024c * 8;
                        tmp[4][m] = tmp024a + tmp024b * 16 + tmp024c + tmp024c;

                        tmp[1][m] = tmp135a + tmp135b + tmp135b + tmp135c * 16;
                        tmp[3][m] = tmp135a + tmp135b * 8 + tmp135c * 4;
                        tmp[5][m] = r1[7] + tmp135a + tmp135b * 32 + tmp135c;


                        // tmp[0][m] = r1[0] + r1[1] + r1[2] + r1[3] + r1[4] + r1[5] + r1[6];
                        // tmp[1][m] = r1[1] - r1[2] + 2 * r1[3] - 2 * r1[4] + 0.5 * r1[5] - 0.5 * r1[6];
                        // tmp[2][m] = r1[1] + r1[2] + 4 * r1[3] + 4 * r1[4] + 0.25 * r1[5] + 0.25 * r1[6];
                        // tmp[3][m] = r1[1] - r1[2] + 8 * r1[3] - 8 * r1[4] + 0.125 * r1[5] - 0.125 * r1[6];
                        // tmp[4][m] = r1[1] + r1[2] + 16 * r1[3] + 16 * r1[4] + 0.0625 * r1[5] + 0.0625 * r1[6];
                        // tmp[5][m] = r1[1] - r1[2] + 32 * r1[3] - 32 * r1[4] + 0.03125 * r1[5] - 0.03125 * r1[6] + r1[7];

                        r1 += block_w * 8;
                    }
                    float *r1_tm_0 = img1_tm + j * block_w * 6 * 6 + k * 6;
                    float *r1_tm_1 = r1_tm_0 + block_w * 6;
                    float *r1_tm_2 = r1_tm_1 + block_w * 6;
                    float *r1_tm_3 = r1_tm_2 + block_w * 6;
                    float *r1_tm_4 = r1_tm_3 + block_w * 6;
                    float *r1_tm_5 = r1_tm_4 + block_w * 6;

                    for(int m = 0; m < 6; m++) {
                        const float *tmp1 = tmp[m];

                        float tmp024a = tmp1[1] + tmp1[2];
                        float tmp135a = tmp1[1] - tmp1[2];

                        float tmp024b = tmp1[3] + tmp1[4];
                        float tmp135b = tmp1[3] - tmp1[4];

                        float tmp024c = tmp1[5] + tmp1[6];
                        float tmp135c = tmp1[5] - tmp1[6];

                        r1_tm_0[m] = tmp1[0] + tmp024a + tmp024b + tmp024c * 32 + bias;
                        r1_tm_2[m] = tmp024a + tmp024b * 4 + tmp024c * 8 + bias;
                        r1_tm_4[m] = tmp024a + tmp024b * 16 + tmp024c + tmp024c + bias;

                        r1_tm_1[m] = tmp135a + tmp135b + tmp135b + tmp135c * 16 + bias;
                        r1_tm_3[m] = tmp135a + tmp135b * 8 + tmp135c * 4 + bias;
                        r1_tm_5[m] = tmp1[7] + tmp135a + tmp135b * 32 + tmp135c + bias;

                        // r1_tm_0[m] = tmp1[0] + tmp1[1] + tmp1[2] + tmp1[3] + tmp1[4] + tmp1[5] + tmp1[6] + bias_data[i];
                        // r1_tm_1[m] = tmp1[1] - tmp1[2] + 2 * tmp1[3] - 2 * tmp1[4] + 0.5 * tmp1[5] - 0.5 * tmp1[6] + bias_data[i];
                        // r1_tm_2[m] = tmp1[1] + tmp1[2] + 4 * tmp1[3] + 4 * tmp1[4] + 0.25 * tmp1[5] + 0.25 * tmp1[6] + bias_data[i];
                        // r1_tm_3[m] = tmp1[1] - tmp1[2] + 8 * tmp1[3] - 8 * tmp1[4] + 0.125 * tmp1[5] - 0.125 * tmp1[6] + bias_data[i];
                        // r1_tm_4[m] = tmp1[1] + tmp1[2] + 16 * tmp1[3] + 16 * tmp1[4] + 0.0625 * tmp1[5] + 0.0625 * tmp1[6] + bias_data[i];
                        // r1_tm_5[m] = tmp1[1] - tmp1[2] + 32 * tmp1[3] - 32 * tmp1[4] + 0.03125 * tmp1[5] - 0.03125 * tmp1[6] + tmp1[7] + bias_data[i];

                    }
                }
            }
        }
        free(output_dot_buf);
        // crop the output after transform: cut extra part (right , bottom)
        crop_output(output_trans_buf, output_data, out_c, out_h, out_w, block_h * 6, block_w * 6);
        output_data += output_size;
    }
    free(input_padd_buf);
    free(input_trans_buf);
    free(output_trans_buf);
    return CSINN_TRUE;
}


// reference by ncnn
static void conv3x3s1_winograd64_transform_kernel_1(struct csi_tensor *o_kernel,
                                                    struct csi_tensor *t_kernel)
{
    int32_t outch = o_kernel->dim[0];
    int32_t inch  = o_kernel->dim[1];

    float *kernel_data = (float *)o_kernel->data;
    // for kernel transform buf, 3x3 --> 8x8
    float *kernel_tm = (float *)malloc(outch * inch * 8 * 8 * sizeof(float));
    // kernel transform matrix: G
    const float ktm[8][3] = {
        {1.0f, 0.0f, 0.0f},
        {-2.0f / 9, -2.0f / 9, -2.0f / 9},
        {-2.0f / 9, 2.0f / 9, -2.0f / 9},
        {1.0f / 90, 1.0f / 45, 2.0f / 45},
        {1.0f / 90, -1.0f / 45, 2.0f / 45},
        {1.0f / 45, 1.0f / 90, 1.0f / 180},
        {1.0f / 45, -1.0f / 90, 1.0f / 180},
        {0.0f, 0.0f, 1.0f}
    };

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

            const float* kernel0 = kernel_data + p * inch * 9 + q * 9;
            float* kernel_tm0 = kernel_tm + p * inch * 64 + q * 64;

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
                float* tmpp = &tmp[j][0];

                for (int i = 0; i < 8; i++) {
                    kernel_tm0[j * 8 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                }
            }
        }
    }
    // optimized layout for winograd64
    // interleave kernel
    int outch4 = outch >> 2;
    int remain_outch_start = outch4 << 2;
    // float *kernel_tm2 = (float *)malloc(8 * 8 * inch * 4 * (outch4 + (outch % 4 + 3) / 4) * sizeof(float));
    float *kernel_tm2 = (float *)malloc(8 * 8 * inch * outch * sizeof(float));

    for(int pp = 0; pp < outch4; pp++) {

        int p = pp * 4;
        float *ktm2 = kernel_tm2 + pp * 8 * 8 * inch * 4;

        const float *kernel0_tm = kernel_tm + p * 64 * inch;
        const float *kernel1_tm = kernel0_tm + 64 * inch;
        const float *kernel2_tm = kernel1_tm + 64 * inch;
        const float *kernel3_tm = kernel2_tm + 64 * inch;

        int q = 0;
        for(; q + 1 < inch; q += 2) {

            const float *k00 = kernel0_tm + q * 64;
            const float *k01 = k00 + 64;
            const float *k10 = kernel1_tm + q * 64;
            const float *k11 = k10 + 64;
            const float *k20 = kernel2_tm + q * 64;
            const float *k21 = k20 + 64;
            const float *k30 = kernel3_tm + q * 64;
            const float *k31 = k30 + 64;

            for(int r = 0; r < 16; r++) {

                for (int m = 0; m < 4; m++) {

                    ktm2[0 + m] = k00[m];
                    ktm2[4 + m] = k01[m];
                    ktm2[8 + m] = k10[m];
                    ktm2[12 + m] = k11[m];
                    ktm2[16 + m] = k20[m];
                    ktm2[20 + m] = k21[m];
                    ktm2[24 + m] = k30[m];
                    ktm2[28 + m] = k31[m];
                }

                k00 += 4;
                k01 += 4;
                k10 += 4;
                k11 += 4;
                k20 += 4;
                k21 += 4;
                k30 += 4;
                k31 += 4;
                ktm2 += 32;
            }
        }
        for(; q < inch; q++) {

            const float* k00 = kernel0_tm + q * 64;
            const float* k10 = kernel1_tm + q * 64;
            const float* k20 = kernel2_tm + q * 64;
            const float* k30 = kernel3_tm + q * 64;

            for(int r = 0; r < 16; r++) {

                for (int m = 0; m < 4; m++) {

                    ktm2[0 + m] = k00[m];
                    ktm2[4 + m] = k10[m];
                    ktm2[8 + m] = k20[m];
                    ktm2[12 + m] = k30[m];
                }

                k00 += 4;
                k10 += 4;
                k20 += 4;
                k30 += 4;
                ktm2 += 16;
            }
        }
    }

    // remain outch
    for(int p = remain_outch_start; p < outch; p++) {

        float *ktm2 = kernel_tm2 + p * 64 * inch;
        const float *kernel0_tm = kernel_tm + p * 64 * inch;
        int q = 0;
        for(; q < inch; q++) {

            const float *k00 = kernel0_tm + q * 64;
            for(int r = 0; r < 16; r++) {

                for(int m = 0; m < 4; m++) {
                    ktm2[m] = k00[m];
                }
                k00 += 4;
                ktm2 += 4;
            }
        }
    }
    free(kernel_tm);
    t_kernel->data = kernel_tm2;
}


// reference by ncnn
static int conv3x3s1_winograd64_1(struct csi_tensor *input,
                                  struct csi_tensor *output,
                                  struct csi_tensor *kernel,
                                  struct csi_tensor *bias,
                                  struct conv2d_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    float *kernel_data = (float *)kernel->data;
    float *bias_data = (float *)bias->data;

    // param
    int kernel_h = kernel->dim[2];
    int kernel_w = kernel->dim[3];
    int stride_h = params->stride_height;
    int stride_w = params->stride_width;
    int dilation_h = params->dilation_height;
    int dilation_w = params->dilation_width;
    int pad_left =  params->pad_left;
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

    int padded_in_h = block_h * 6 + 2;  // block * 4 for alignment with 4，kernel = 3 * 3 ，stride = 1，thus input_size + 2
    int padded_in_w = block_w * 6 + 2;
    int padded_in_hw = padded_in_h * padded_in_w;   // element size after padding per channel

    // buffer addr
    float *input_padd_buf = (float *)malloc(in_c * padded_in_hw * sizeof(float));
    // interleave by （4, 16 * block_h * block_w, in_c）
    float *input_trans_buf = (float *)malloc(in_c * block_h * block_w * 8 * 8 * sizeof(float));

    float *output_trans_buf = (float *)malloc(out_c * block_h * block_w * 6 * 6 * sizeof(float));

    for(int n = 0; n < batch; n++) {

        // pad input
        pad_input(input_data, input_padd_buf, in_c, in_h, in_w, padded_in_h, padded_in_w, pad_top, pad_left);
        input_data += input_size;

        // transform input
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
        int in_h_tm = block_h * 8;  // input height after transform
        int in_w_tm = block_w * 8;

        const int tiles = block_h * block_w;

        for(int q = 0; q < in_c; q++) {

            const float *img0 = input_padd_buf + q * padded_in_h * padded_in_w; // pad后padinput的第q个channle
            float *img0_tm = input_trans_buf + q * block_h * block_w * 8 * 8;   // transform and interleave 后的第q个channel

            float tmp[8][8];

            for(int i = 0; i < block_h; i++) {

                for(int j = 0; j < block_w; j++) {

                    const float *r0 = img0 + i * padded_in_w * 6 + j * 6;

                    for(int m = 0; m < 8; m++) {
                        tmp[0][m] = r0[0] - r0[6] + 5.25 * (r0[4] - r0[2]);
                        tmp[7][m] = r0[7] - r0[1] + 5.25 * (r0[3] - r0[5]);

                        float tmp12a = (r0[2] + r0[6] - r0[4] * 4.25f);
                        float tmp12b = (r0[1] + r0[5] - r0[3] * 4.25f);
                        tmp[1][m] = tmp12a + tmp12b;
                        tmp[2][m] = tmp12a - tmp12b;

                        float tmp34a = (r0[6] + r0[2] * 0.25f - r0[4] * 1.25f);
                        float tmp34b = (r0[1] * 0.5f - r0[3] * 2.5f + r0[5] * 2.f);
                        tmp[3][m] = tmp34a + tmp34b;
                        tmp[4][m] = tmp34a - tmp34b;

                        float tmp56a = (r0[6] + (r0[2] - r0[4] * 1.25f) * 4.f);
                        float tmp56b = (r0[1] * 2.f - r0[3] * 2.5f + r0[5] * 0.5f);
                        tmp[5][m] = tmp56a + tmp56b;
                        tmp[6][m] = tmp56a - tmp56b;

                        r0 += padded_in_w;
                    }

                    float *r0_tm_0 = img0_tm + 4 * (i * block_w + j);
                    float *r0_tm_4 = img0_tm + 4 * (i * block_w + j + block_h * block_w);

                    for(int m = 0; m < 8; m++) {

                        const float *tmp0 = tmp[m];

                        r0_tm_0[0] = tmp0[0] - tmp0[6] + 5.25 * (tmp0[4] - tmp0[2]);
                        r0_tm_4[3] = tmp0[7] - tmp0[1] + 5.25 * (tmp0[3] - tmp0[5]);

                        float tmp12a = (tmp0[2] + tmp0[6] - tmp0[4] * 4.25f);
                        float tmp12b = (tmp0[1] + tmp0[5] - tmp0[3] * 4.25f);
                        r0_tm_0[1] = tmp12a + tmp12b;
                        r0_tm_0[2] = tmp12a - tmp12b;

                        float tmp34a = (tmp0[6] + tmp0[2] * 0.25f - tmp0[4] * 1.25f);
                        float tmp34b = (tmp0[1] * 0.5f - tmp0[3] * 2.5f + tmp0[5] * 2.f);
                        r0_tm_0[3] = tmp34a + tmp34b;
                        r0_tm_4[0] = tmp34a - tmp34b;

                        float tmp56a = (tmp0[6] + (tmp0[2] - tmp0[4] * 1.25f) * 4.f);
                        float tmp56b = (tmp0[1] * 2.f - tmp0[3] * 2.5f + tmp0[5] * 0.5f);
                        r0_tm_4[1] = tmp56a + tmp56b;
                        r0_tm_4[2] = tmp56a - tmp56b;

                        r0_tm_0 += 4 * block_h * block_w * 2;
                        r0_tm_4 += 4 * block_h * block_w * 2;

                    }
                }
            }
        }

        // dot
        // interleave by (4, 16 * block_h * block_w, out_c)
        float *output_dot_buf = (float *)calloc(out_c * block_h * block_w * 8 * 8, sizeof(float));
        int outch4 = out_c >> 2;
        int remain_outch_start = outch4 << 2;

        for(int pp = 0; pp < outch4; pp++) {

            int p = pp * 4;
            float *out0_tm = output_dot_buf + p * 4 * 16 * block_h * block_w;        // 每一个输出面
            float *out1_tm = out0_tm + 4 * 16 * block_h * block_w;
            float *out2_tm = out1_tm + 4 * 16 * block_h * block_w;
            float *out3_tm = out2_tm + 4 * 16 * block_h * block_w;

            const float *ktm = kernel_data + pp * 8 * 8 * in_c * 4;

            int q = 0;

            for(; q + 1 < in_c; q += 2) {

                const float *r0 = input_trans_buf + q * 4 * 16 * block_h * block_w;
                const float *r1 = r0 + 4 * 16 * block_h * block_w;

                float *output0_tm = out0_tm;
                float *output1_tm = out1_tm;
                float *output2_tm = out2_tm;
                float *output3_tm = out3_tm;

                for(int  r = 0; r < 16; r++) {

                    for(int t = 0; t < block_h * block_w; t++) {

                        for(int m = 0; m < 4; m++) {

                            output0_tm[m] += r0[m] * ktm[0 + m];
                            output0_tm[m] += r1[m] * ktm[4 + m];
                            output1_tm[m] += r0[m] * ktm[8 + m];
                            output1_tm[m] += r1[m] * ktm[12 + m];
                            output2_tm[m] += r0[m] * ktm[16 + m];
                            output2_tm[m] += r1[m] * ktm[20 + m];
                            output3_tm[m] += r0[m] * ktm[24 + m];
                            output3_tm[m] += r1[m] * ktm[28 + m];
                        }
                        r0 += 4;
                        r1 += 4;
                        output0_tm += 4;
                        output1_tm += 4;
                        output2_tm += 4;
                        output3_tm += 4;
                    }
                    ktm += 32;
                }


            }

            for(; q < in_c; q++) {

                const float *r0 = input_trans_buf + q * 4 * 16 * block_h * block_w;
                float* output0_tm = out0_tm;
                float* output1_tm = out1_tm;
                float* output2_tm = out2_tm;
                float* output3_tm = out3_tm;

                for(int r = 0; r < 16; r++) {

                    for(int t = 0; t < block_h * block_w; t++) {

                        for(int m = 0; m < 4; m++) {

                            output0_tm[m] += r0[m] * ktm[0 + m];
                            output1_tm[m] += r0[m] * ktm[4 + m];
                            output2_tm[m] += r0[m] * ktm[8 + m];
                            output3_tm[m] += r0[m] * ktm[12 + m];
                        }

                        r0 += 4;
                        output0_tm += 4;
                        output1_tm += 4;
                        output2_tm += 4;
                        output3_tm += 4;
                    }
                    ktm += 16;
                }

            }
        }
        // dot remain outch
        for(int p = remain_outch_start; p < out_c; p++) {

            float *out0_tm = output_dot_buf + p * 4 * 16 * block_h * block_w;
            const float *ktm = kernel_data + p * 64 * in_c;
            int q = 0;
            for(; q < in_c; q++) {

                const float *r0 = input_trans_buf + q * 4 * 16 * block_h * block_w;
                float *output0_tm = out0_tm;

                for(int r = 0; r < 16; r++) {

                    for(int t = 0; t < block_h * block_w; t++) {

                        for(int m = 0; m < 4; m++) {

                            output0_tm[m] += r0[m] * ktm[m];
                        }
                        r0 += 4;
                        output0_tm += 4;
                    }
                    ktm += 4;
                }
            }
        }

        // transform output
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

        for(int p = 0; p < out_c; p++) {

            const float bias = bias_data ? bias_data[p] : 0.f;

            const float *out0_tm = output_dot_buf + p * 64 * block_h * block_w;
            float *out0 = output_trans_buf + p * 36 * block_h * block_w;

            float tmp[6][8];
            for(int i = 0; i < block_h; i++) {

                for(int j = 0; j < block_w; j++) {

                    const float *output0_tm_0 = out0_tm + 4 * (i * block_w + j);
                    const float *output0_tm_4 = out0_tm + 4 * (i * block_w + j + block_h * block_w);

                    for(int m = 0; m < 8; m++) {

                        float tmp024a = output0_tm_0[1] + output0_tm_0[2];
                        float tmp135a = output0_tm_0[1] - output0_tm_0[2];

                        float tmp024b = output0_tm_0[3] + output0_tm_4[0];
                        float tmp135b = output0_tm_0[3] - output0_tm_4[0];

                        float tmp024c = output0_tm_4[1] + output0_tm_4[2];
                        float tmp135c = output0_tm_4[1] - output0_tm_4[2];

                        tmp[0][m] = output0_tm_0[0] + tmp024a + tmp024b + tmp024c * 32;
                        tmp[2][m] = tmp024a + tmp024b * 4 + tmp024c * 8;
                        tmp[4][m] = tmp024a + tmp024b * 16 + tmp024c + tmp024c;

                        tmp[1][m] = tmp135a + tmp135b + tmp135b + tmp135c * 16;
                        tmp[3][m] = tmp135a + tmp135b * 8 + tmp135c * 4;
                        tmp[5][m] = output0_tm_4[3] + tmp135a + tmp135b * 32 + tmp135c;

                        output0_tm_0 += 4 * block_h * block_w * 2;
                        output0_tm_4 += 4 * block_h * block_w * 2;

                    }

                    float *output0 = out0 + i * 6 * block_w * 6 + j * 6;

                    for (int m = 0; m < 6; m++) {

                        const float *tmp0 = tmp[m];

                        float tmp024a = tmp0[1] + tmp0[2];
                        float tmp135a = tmp0[1] - tmp0[2];

                        float tmp024b = tmp0[3] + tmp0[4];
                        float tmp135b = tmp0[3] - tmp0[4];

                        float tmp024c = tmp0[5] + tmp0[6];
                        float tmp135c = tmp0[5] - tmp0[6];

                        output0[0] = bias + tmp0[0] + tmp024a + tmp024b + tmp024c * 32;
                        output0[2] = bias + tmp024a + tmp024b * 4 + tmp024c * 8;
                        output0[4] = bias + tmp024a + tmp024b * 16 + tmp024c + tmp024c;

                        output0[1] = bias + tmp135a + tmp135b + tmp135b + tmp135c * 16;
                        output0[3] = bias + tmp135a + tmp135b * 8 + tmp135c * 4;
                        output0[5] = bias + tmp0[7] + tmp135a + tmp135b * 32 + tmp135c;

                        output0 += block_w * 6;
                    }

                }
            }
        }
        free(output_dot_buf);
        // crop the output after transform: cut extra part (right , bottom)
        crop_output(output_trans_buf, output_data, out_c, out_h, out_w, block_h * 6, block_w * 6);
        output_data += output_size;
    }
    free(input_padd_buf);
    free(input_trans_buf);
    free(output_trans_buf);
    return CSINN_TRUE;
}


static void conv3x3s1(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct csi_tensor *kernel,
                      struct csi_tensor *bias,
                      struct conv2d_params *params)
{
    /* to do */
}

static void conv3x3s2(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct csi_tensor *kernel,
                      struct csi_tensor *bias,
                      struct conv2d_params *params)
{
    /* to do */
}