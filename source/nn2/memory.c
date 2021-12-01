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

/* CSI-NN2 version 1.10.x */

#include "csi_nn.h"

// #define CSI_MEM_DEBUG

struct csi_mem_alloc_debug_element_ {
    void *ptr;
    int64_t size;
    int is_free;
};

struct csi_mem_alloc_debug_map_ {
    struct csi_mem_alloc_debug_element_ *element;
    int element_number;
    int index;
    int64_t total_size;
};

static struct csi_mem_alloc_debug_map_ csi_mem_alloc_debug_map;

void csi_mem_print_map()
{
    printf("total size = %ld\n", csi_mem_alloc_debug_map.total_size);
    for (int i = 0; i <= csi_mem_alloc_debug_map.index; i++) {
        struct csi_mem_alloc_debug_element_ *e = csi_mem_alloc_debug_map.element + i;
        printf("element %d: ptr = %p, size = %ld, is_free = %d\n", i, e->ptr, e->size, e->is_free);
    }
}

static int csi_mem_map_insert(void *ptr, uint64_t size)
{
    int element_number = csi_mem_alloc_debug_map.element_number;
    int index = csi_mem_alloc_debug_map.index;
    if (element_number == 0 || index == element_number - 1) {
        csi_mem_alloc_debug_map.element_number += 512;
        csi_mem_alloc_debug_map.element = realloc(csi_mem_alloc_debug_map.element,
                                            csi_mem_alloc_debug_map.element_number *
                                            sizeof(struct csi_mem_alloc_debug_element_));
    }
    csi_mem_alloc_debug_map.element[index].ptr = ptr;
    csi_mem_alloc_debug_map.element[index].size = size;
    csi_mem_alloc_debug_map.element[index].is_free = 0;
    csi_mem_alloc_debug_map.index++;
}

void *csi_mem_alloc(int64_t size)
{
    void *ret = calloc(1, size);
    if (ret == NULL) {
        csi_debug_error("cannot alloc memory\n");
    }
#ifdef CSI_MEM_DEBUG
    csi_mem_map_insert(ret, size);
    csi_mem_alloc_debug_map.total_size += size;
    printf("csi_mem_alloc: total size = %ld\n", csi_mem_alloc_debug_map.total_size);
#endif
    return ret;
}

void csi_mem_free(void *ptr)
{
#ifdef CSI_MEM_DEBUG
    for (int i = 0; i < csi_mem_alloc_debug_map.index; i++) {
        struct csi_mem_alloc_debug_element_ *e = csi_mem_alloc_debug_map.element + i;
        if (e->ptr == ptr && e->is_free == 0) {
            e->is_free = 1;
            csi_mem_alloc_debug_map.total_size -= e->size;
            printf("csi_mem_free: total size = %ld\n", csi_mem_alloc_debug_map.total_size);
            break;
        }
    }
#endif
    free(ptr);
}
