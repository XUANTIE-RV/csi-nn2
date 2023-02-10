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

#ifndef INCLUDE_SHL_TVMGEN_H_
#define INCLUDE_SHL_TVMGEN_H_

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "csi_nn.h"
#include "shl_node.h"

struct shl_tvmgen_name_func {
    char *name;
    int (*ptr)();
    enum csinn_optimize_method_enum opt_method;
};

struct shl_tvmgen_resource_handle {
    int *input_output;
    void *op_params;
};

int shl_tvmgen_layer_func(struct shl_node *node);

int shl_tvmgen_map_reg(struct shl_tvmgen_name_func *map, int size);
void *shl_tvmgen_find_reg(char *name, enum csinn_optimize_method_enum *opt_method);

#endif  // INCLUDE_SHL_TVMGEN_H_
