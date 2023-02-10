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

#include "csi_nn.h"
#include "shl_gref.h"
#include "shl_tvmgen.h"

static int reg_func_in = 0;

int32_t conv2d_conv1_0_fuse_multiply_1_fuse_add_conv1_bn_PART_0_2___tvm_main__(
    void *args, int32_t *arg_type_ids, int32_t num_args, void *out_ret_value,
    int32_t *out_ret_tcode, void *resource_handle)
{
    reg_func_in++;
    printf("in func1: %s\n", __func__);
    return 0;
}

int32_t relu_output__relu1_3___tvm_main__(void *args, int32_t *arg_type_ids, int32_t num_args,
                                          void *out_ret_value, int32_t *out_ret_tcode,
                                          void *resource_handle)
{
    reg_func_in++;
    printf("in func2: %s\n", __func__);
    return 0;
}

void run_model()
{
    char *params_base = shl_mem_alloc(3704);
    struct csinn_session *sess = csinn_alloc_session();
    sess->base_run_mode = CSINN_RM_CPU_GRAPH;
    sess->model.save_mode = CSINN_RUN_ONLY;
    sess->base_api = CSINN_C906;
    sess->base_dtype = CSINN_DTYPE_FLOAT32;
    sess->dynamic_shape = CSINN_FALSE;
    csinn_session_init(sess);
    csinn_set_input_number(1, sess);
    csinn_set_output_number(1, sess);

    struct csinn_tensor *data = csinn_alloc_tensor(sess);
    data->name = "data@@conv2d_conv1_0_fuse_multiply_1_fuse_add_conv1/bn_PART_0_2_0";
    data->dtype = CSINN_DTYPE_FLOAT32;
    data->layout = CSINN_LAYOUT_NCHW;
    data->dim[0] = 1;
    data->dim[1] = 3;
    data->dim[2] = 224;
    data->dim[3] = 224;
    data->dim_count = 4;
    data->qinfo = (struct csinn_quant_info *)(params_base + 0);
    data->quant_channel = 1;
    struct csinn_tensor *output_0 = csinn_alloc_tensor(sess);
    output_0->name = "output_0";
    output_0->dtype = CSINN_DTYPE_FLOAT32;
    output_0->layout = CSINN_LAYOUT_NCHW;
    output_0->dim[0] = 1;
    output_0->dim[1] = 32;
    output_0->dim[2] = 112;
    output_0->dim[3] = 112;
    output_0->dim_count = 4;
    output_0->qinfo = (struct csinn_quant_info *)(params_base + 24);
    output_0->quant_channel = 1;
    struct csinn_tensor *kernel_0 = csinn_alloc_tensor(sess);
    kernel_0->name = "kernel_0";
    kernel_0->data = params_base + 72;
    kernel_0->is_const = 1;
    kernel_0->dtype = CSINN_DTYPE_FLOAT32;
    kernel_0->layout = CSINN_LAYOUT_OIHW;
    kernel_0->dim[0] = 32;
    kernel_0->dim[1] = 3;
    kernel_0->dim[2] = 3;
    kernel_0->dim[3] = 3;
    kernel_0->dim_count = 4;
    kernel_0->qinfo = (struct csinn_quant_info *)(params_base + 48);
    kernel_0->quant_channel = 1;
    struct csinn_tensor *bias_0 = csinn_alloc_tensor(sess);
    bias_0->name = "bias_0";
    bias_0->data = params_base + 3552;
    bias_0->is_const = 1;
    bias_0->dtype = CSINN_DTYPE_FLOAT32;
    bias_0->layout = CSINN_LAYOUT_O;
    bias_0->dim[0] = 32;
    bias_0->dim_count = 1;
    bias_0->qinfo = (struct csinn_quant_info *)(params_base + 3528);
    bias_0->quant_channel = 1;
    struct csinn_conv2d_params *params_0 =
        csinn_alloc_params(sizeof(struct csinn_conv2d_params), sess);
    params_0->group = 1;
    params_0->stride_height = 2;
    params_0->stride_width = 2;
    params_0->dilation_height = 1;
    params_0->dilation_width = 1;
    params_0->conv_extra.kernel_tm = NULL;
    params_0->conv_extra.conv_mode = CSINN_DIRECT;
    params_0->pad_top = 1;
    params_0->pad_left = 1;
    params_0->pad_down = 1;
    params_0->pad_right = 1;
    params_0->base.name = "conv2d_conv1_0_fuse_multiply_1_fuse_add_conv1/bn_PART_0_2";
    csinn_conv2d_init(data, output_0, kernel_0, bias_0, params_0);
    struct csinn_tensor *output_1 = csinn_alloc_tensor(sess);
    output_1->name = "relu_output@@relu1_3_1";
    output_1->dtype = CSINN_DTYPE_FLOAT32;
    output_1->layout = CSINN_LAYOUT_NCHW;
    output_1->dim[0] = 1;
    output_1->dim[1] = 32;
    output_1->dim[2] = 112;
    output_1->dim[3] = 112;
    output_1->dim_count = 4;
    output_1->qinfo = (struct csinn_quant_info *)(params_base + 3680);
    output_1->quant_channel = 1;
    struct csinn_relu_params *params_1 = csinn_alloc_params(sizeof(struct csinn_relu_params), sess);
    params_1->base.name = "relu_output@@relu1_3";
    csinn_relu_init(output_0, output_1, params_1);
    csinn_set_tensor_entry(data, sess);
    csinn_set_input(0, data, sess);

    csinn_conv2d(data, output_0, kernel_0, bias_0, params_0);
    csinn_relu(output_0, output_1, params_1);
    csinn_set_output(0, output_1, sess);

    csinn_session_setup(sess);

    struct csinn_tensor *input_tensors = csinn_alloc_tensor(NULL);
    input_tensors->dim_count = 4;
    input_tensors->dim[0] = 1;
    input_tensors->dim[1] = 3;
    input_tensors->dim[2] = 224;
    input_tensors->dim[3] = 224;
    input_tensors->data = shl_mem_alloc(4 * 3 * 224 * 224);
    csinn_update_input(0, input_tensors, sess);
    csinn_session_run(sess);
}

struct shl_tvmgen_name_func hhb_gen_func_map[2];
void register_functions()
{
    hhb_gen_func_map[0].name = "conv2d_conv1_0_fuse_multiply_1_fuse_add_conv1/bn_PART_0_2";
    hhb_gen_func_map[0].ptr =
        conv2d_conv1_0_fuse_multiply_1_fuse_add_conv1_bn_PART_0_2___tvm_main__;
    hhb_gen_func_map[0].opt_method = CSINN_OPT_TVMGEN;
    hhb_gen_func_map[1].name = "relu_output@@relu1_3";
    hhb_gen_func_map[1].ptr = relu_output__relu1_3___tvm_main__;
    hhb_gen_func_map[1].opt_method = CSINN_OPT_TVMGEN;
    shl_tvmgen_map_reg(hhb_gen_func_map, 2);
}

int test1()
{
    register_functions();
    run_model();

    if (reg_func_in == 2) {
        printf("Test1 sucessfully.\n");
        return 0;
    } else {
        printf("Failed Test1.\n");
        return 1;
    }
}

int test_register() { return test1(); }

int main(int argc, char **argv)
{
    printf("Testing function of tvmgen register.\n");

    return test_register();
}