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

#include "csi_ref.h"
#include "csi_utils.h"

int csi_ref_matmul_f32(struct csi_tensor *mat0, struct csi_tensor *mat1, struct csi_tensor *output,
                       struct matmul_params *params)
{
    float *mat0_data = mat0->data;
    float *mat1_data = mat1->data;
    float *output_data = output->data;
    const int dims_count = mat0->dim_count;
    int batches = 1;

    /* compute the outer size */
    for (int i = 0; i < dims_count - 2; i++) {
        batches *= mat0->dim[i];
    }

    const int dim_i = mat0->dim[dims_count - (params->trans_a ? 1 : 2)];
    const int dim_k = mat0->dim[dims_count - (params->trans_a ? 2 : 1)];
    const int dim_j = mat1->dim[dims_count - (params->trans_b ? 2 : 1)];
    const int mat0_offset = dim_i * dim_k;
    const int mat1_offset = dim_k * dim_j;
    const int out_offset = dim_i * dim_j;

    if (!params->trans_a && !params->trans_b) {
        for (int b = 0; b < batches; ++b) {
            for (int i = 0; i < dim_i; ++i) {
                for (int j = 0; j < dim_j; ++j) {
                    float total = 0.f;
                    for (int k = 0; k < dim_k; ++k) {
                        int offset0 = mat0_offset * b + i * dim_k + k;
                        int offset1 = mat1_offset * b + k * dim_j + j;
                        total += mat0_data[offset0] * mat1_data[offset1];
                    }
                    output_data[b * out_offset + i * dim_j + j] = total;
                }
            }
        }
    } else if (!params->trans_a && params->trans_b) {
        for (int b = 0; b < batches; ++b) {
            for (int i = 0; i < dim_i; ++i) {
                for (int j = 0; j < dim_j; ++j) {
                    float total = 0.f;
                    for (int k = 0; k < dim_k; ++k) {
                        int offset0 = mat0_offset * b + i * dim_k + k;
                        int offset1 = mat1_offset * b + j * dim_k + k;
                        total += mat0_data[offset0] * mat1_data[offset1];
                    }
                    output_data[b * out_offset + i * dim_j + j] = total;
                }
            }
        }
    } else if (params->trans_a && !params->trans_b) {
        for (int b = 0; b < batches; ++b) {
            for (int i = 0; i < dim_i; ++i) {
                for (int j = 0; j < dim_j; ++j) {
                    float total = 0.f;
                    for (int k = 0; k < dim_k; ++k) {
                        int offset0 = mat0_offset * b + k * dim_i + i;
                        int offset1 = mat1_offset * b + k * dim_j + j;
                        total += mat0_data[offset0] * mat1_data[offset1];
                    }
                    output_data[b * out_offset + i * dim_j + j] = total;
                }
            }
        }
    } else {
        for (int b = 0; b < batches; ++b) {
            for (int i = 0; i < dim_i; ++i) {
                for (int j = 0; j < dim_j; ++j) {
                    float total = 0.f;
                    for (int k = 0; k < dim_k; ++k) {
                        int offset0 = mat0_offset * b + k * dim_i + i;
                        int offset1 = mat1_offset * b + j * dim_k + k;
                        total += mat0_data[offset0] * mat1_data[offset1];
                    }
                    output_data[b * out_offset + i * dim_j + j] = total;
                }
            }
        }
    }

    return CSINN_TRUE;
}

int csi_ref_matmul_quant(struct csi_tensor *mat0, struct csi_tensor *mat1,
                         struct csi_tensor *output, struct matmul_params *params)
{
    return csi_ref_diso_callback_base(mat0, mat1, output, params, csi_ref_matmul_f32);
}
