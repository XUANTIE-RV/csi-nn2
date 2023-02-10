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
#include "shl_tvmgen.h"
#include "shl_utils.h"

struct shl_tvmgen_name_func_map {
    int size;
    struct shl_tvmgen_name_func *reg;
};

static struct shl_tvmgen_name_func_map name_func_map;

int shl_tvmgen_map_reg(struct shl_tvmgen_name_func *map, int size)
{
    name_func_map.size = size;
    name_func_map.reg = map;
    return CSINN_TRUE;
}

void *shl_tvmgen_find_reg(char *name, enum csinn_optimize_method_enum *opt_method)
{
    if (name == NULL) {
        return NULL;
    }
    for (int i = 0; i < name_func_map.size; i++) {
        if (name_func_map.reg[i].name == NULL) {
            continue;
        }
        if (strcmp(name, name_func_map.reg[i].name) == 0) {
            if (name_func_map.reg[i].opt_method == 0) {
                shl_debug_warning("Get opt_method = 0\n, Please register valid opt_method\n");
            }
            *opt_method = name_func_map.reg[i].opt_method;
            return name_func_map.reg[i].ptr;
        }
    }
    return NULL;
}
