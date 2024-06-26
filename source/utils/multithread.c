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

#include "shl_debug.h"
#include "shl_multithread.h"

int shl_thread_num = 1;

void shl_multithread_set_threads(int threads)
{
#ifdef _OPENMP
    shl_thread_num = threads;
    omp_set_num_threads(threads);
#else
    shl_debug_warning("OPENMP is not defined!\n");
#endif
}

int shl_multithread_is_enable()
{
#ifdef _OPENMP
    omp_set_num_threads(shl_thread_num);
    if (omp_get_max_threads() > 1) {
        return CSINN_TRUE;
    }
#endif
    return CSINN_FALSE;
}
