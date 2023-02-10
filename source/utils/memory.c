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
#include <unistd.h>

#include "csi_nn.h"

// #define SHL_MEM_DEBUG
// #define SHL_MEM_DEBUG_VALID_WRITE
// #define SHL_USE_ATAT_MALLOC
struct shl_mem_alloc_debug_element_ {
    void *ptr;
    int64_t size;
    int is_free;
};

struct shl_mem_alloc_debug_map_ {
    struct shl_mem_alloc_debug_element_ *element;
    int element_number;
    int index;
    int64_t total_size;
};

static struct shl_mem_alloc_debug_map_ shl_mem_alloc_debug_map;

void shl_mem_print_map()
{
    printf("total size = %ld\n", shl_mem_alloc_debug_map.total_size);
    for (int i = 0; i <= shl_mem_alloc_debug_map.index; i++) {
        struct shl_mem_alloc_debug_element_ *e = shl_mem_alloc_debug_map.element + i;
        printf("element %d: ptr = %p, size = %ld, is_free = %d\n", i, e->ptr, e->size, e->is_free);
    }
}

static int shl_mem_map_insert(void *ptr, uint64_t size)
{
    int element_number = shl_mem_alloc_debug_map.element_number;
    int index = shl_mem_alloc_debug_map.index;
    if (element_number == 0 || index == element_number - 1) {
        shl_mem_alloc_debug_map.element_number += 512;
        shl_mem_alloc_debug_map.element = realloc(
            shl_mem_alloc_debug_map.element,
            shl_mem_alloc_debug_map.element_number * sizeof(struct shl_mem_alloc_debug_element_));
    }
    shl_mem_alloc_debug_map.element[index].ptr = ptr;
    shl_mem_alloc_debug_map.element[index].size = size;
    shl_mem_alloc_debug_map.element[index].is_free = 0;
    shl_mem_alloc_debug_map.index++;
    return 0;
}

__attribute__((weak)) void *shl_mem_alloc(int64_t size)
{
    void *ret;
    if (size == 0) {
        shl_debug_info("alloc 0 byte\n");
        return NULL;
    }
#ifdef SHL_MEM_DEBUG_VALID_WRITE
    ret = calloc(1, size + 8);
    int8_t *check_ptr = ret + size;
    /* magic number */
    check_ptr[0] = 0xff;
    check_ptr[1] = 0x23;
    check_ptr[2] = 0x33;
    check_ptr[3] = 0x44;
    check_ptr[4] = 0x45;
    check_ptr[5] = 0x55;
    check_ptr[6] = 0x67;
    check_ptr[7] = 0xff;
#else
#ifdef SHL_USE_ATAT_MALLOC
    void *shl_atat_calloc(size_t n, size_t m);
    ret = shl_atat_calloc(1, size);
#else
    ret = calloc(1, size);
#endif
#endif
    if (ret == NULL) {
        shl_debug_error("cannot alloc memory\n");
    }
#ifdef SHL_MEM_DEBUG
    shl_mem_map_insert(ret, size);
    shl_mem_alloc_debug_map.total_size += size;
    printf("shl_mem_alloc: total size = %ld, get size is %lld\n",
           shl_mem_alloc_debug_map.total_size, size);
#endif
    return ret;
}

void *shl_mem_calloc(size_t nmemb, size_t size) { return shl_mem_alloc(nmemb * size); }

void *shl_mem_realloc(void *ptr, size_t size, size_t orig_size)
{
    void *ret = shl_mem_alloc(size);
    if (!ptr) {
        return ret;
    }
    if (orig_size == 0) {
        shl_debug_warning(
            "New size(instead of original size) will be applied into memcpy, which may cause "
            "problems.\n");
        memcpy(ret, ptr, size);
    } else {
        memcpy(ret, ptr, orig_size);
    }
    shl_mem_free(ptr);
    return ret;
}

void *shl_mem_alloc_aligned(int64_t size, int aligned_bytes)
{
    void *ptr = NULL;
#ifdef SHL_BUILD_RTOS
    size_t real_size = size + aligned_bytes;
    void *tptr = shl_mem_alloc(real_size);
#ifdef SHL_BUILD_C906
    long mask = ~(aligned_bytes - 1);
    long addr = ((long)tptr + aligned_bytes) & mask;
#else
    int mask = ~(aligned_bytes - 1);
    int addr = ((int)tptr + aligned_bytes) & mask;
#endif
    ptr = (void *)addr;
#else
    if (aligned_bytes == 0) {
        aligned_bytes = getpagesize();
    }
    int ret = posix_memalign(&ptr, aligned_bytes, size);
    if (ret || ptr == NULL) shl_debug_error("cannot alloc aligned memory\n");
#endif
    return ptr;
}

__attribute__((weak)) void shl_mem_free(void *ptr)
{
#ifdef SHL_MEM_DEBUG
    for (int i = 0; i < shl_mem_alloc_debug_map.index; i++) {
        struct shl_mem_alloc_debug_element_ *e = shl_mem_alloc_debug_map.element + i;
        if (e->ptr == ptr && e->is_free == 0) {
            e->is_free = 1;
            shl_mem_alloc_debug_map.total_size -= e->size;
            printf("shl_mem_free: total size = %ld\n", shl_mem_alloc_debug_map.total_size);
#ifdef SHL_MEM_DEBUG_VALID_WRITE
            uint8_t *cptr = ptr + e->size;
            if ((cptr[0] == 0xff) && (cptr[1] == 0x23) && (cptr[2] == 0x33) && (cptr[3] == 0x44) &&
                (cptr[4] == 0x45) && (cptr[5] == 0x55) && (cptr[6] == 0x67) && (cptr[7] == 0xff)) {
                break;
            } else {
                printf("shl_mem_free: invalid write %p\n", ptr);
            }
#else
            break;
#endif
        }
    }
#endif
#ifdef SHL_USE_ATAT_MALLOC
    void shl_atat_free(void *f);
    shl_atat_free(ptr);
#else
    free(ptr);
#endif
}
