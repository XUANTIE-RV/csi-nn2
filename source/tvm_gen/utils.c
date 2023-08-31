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

#include "dlpack/dlpack.h"
#include "tvmgen/shl_tvmgen.h"

static DLTensor *tensor_to_dltensor(struct csinn_tensor *tensor)
{
    DLTensor *ret = shl_mem_alloc(sizeof(DLTensor));
    ret->data = tensor->data;
    ret->ndim = tensor->dim_count;
    ret->shape = shl_mem_alloc(sizeof(int64_t *) * ret->ndim);
    for (int i = 0; i < tensor->dim_count; i++) {
        ret->shape[i] = tensor->dim[i];
    }
    ret->dtype.code = kDLFloat;
    ret->dtype.bits = csinn_tensor_byte_size(tensor) / csinn_tensor_size(tensor) * 8;
    ret->dtype.lanes = 1;
    ret->byte_offset = 0;
    return ret;
}

static void free_dltensor(DLTensor *tensor)
{
    shl_mem_free(tensor->shape);
    shl_mem_free(tensor);
}

int shl_tvmgen_layer_func(struct shl_node *node)
{
    /* base has same address with params */
    struct csinn_params_base *params = node->data;
    int (*func)();
    struct csinn_callback *cb = params->cb;
    func = cb->exec;

    int num_args = node->in_num + node->out_num;

    void *args[num_args];
    struct shl_tvmgen_resource_handle *handle =
        shl_mem_alloc(sizeof(struct shl_tvmgen_resource_handle));
    handle->input_output = shl_mem_alloc(sizeof(int *) * num_args);
    handle->op_params = params;

    for (int i = 0; i < node->in_num; i++) {
        args[i] = tensor_to_dltensor(node->in[i]->data);
        // shl input type = 0
        handle->input_output[i] = 0;
    }

    for (int i = node->in_num; i < num_args; i++) {
        args[i] = tensor_to_dltensor(node->out[i - node->in_num]->data);
        // shl output type = 1
        handle->input_output[i] = 1;
    }

    // kTVMDLTensorHandle = 7U
    int32_t arg_type_ids[num_args];
    for (int i = 0; i < num_args; i++) {
        arg_type_ids[i] = 7;
    }

    // for conv/dense
    // if bias dim_count==0, remve this dltensor.
    for (int i = 0; i < num_args; i++) {
        DLTensor *dlt = args[i];
        if (dlt->ndim == 0) {
            for (int j = i + 1; j < num_args; j++) {
                args[j - 1] = args[j];
            }
            args[num_args - 1] = dlt;
        }
    }

    shl_debug_info("Call tvmgen func: %s\n", params->name);
    func(args, arg_type_ids, num_args, NULL, NULL, handle);

    for (int i = 0; i < num_args; i++) {
        free_dltensor(args[i]);
    }

    shl_mem_free(handle->input_output);
    shl_mem_free(handle);

    return CSINN_TRUE;
}
