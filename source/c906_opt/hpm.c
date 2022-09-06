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

/* CSI-NN2 version 2.0.x */

#include "shl_c906.h"

/*
    hpm: hardware performance monitor
    note: Refer to the hpm sample program in the c906 user manual, Enable related status first.
*/
struct shl_c906_hpm shl_c906_get_hw_perf()
{
    struct shl_c906_hpm tmp;
    asm volatile(
                "csrr %0, instret\n\t"
                "csrr %1, cycle\n\t"
                "csrr %2, hpmcounter3\n\t"
                "csrr %3, hpmcounter4\n\t"
                "csrr %4, hpmcounter13\n\t"
                "csrr %5, hpmcounter14\n\t"
                "csrr %6, hpmcounter15\n\t"
                "csrr %7, hpmcounter16\n\t"
                "csrr %8, hpmcounter17\n\t"

                 : "=r"(tmp.inst),
                   "=r"(tmp.cycle),
                   "=r"(tmp.l1_icache_access),
                   "=r"(tmp.l1_icache_miss),
                   "=r"(tmp.store_inst),
                   "=r"(tmp.l1_dcache_raccess),
                   "=r"(tmp.l1_dcache_rmiss),
                   "=r"(tmp.l1_dcache_waccess),
                   "=r"(tmp.l1_dcache_wmiss)
                 :
                 : "memory");
    return tmp;
}

uint64_t shl_c906_get_inst()
{
    uint64_t inst = 0;
    asm volatile("csrr %0, instret"
                 : "=r"(inst)
                 :
                 : "memory");
    // asm volatile("csrr %[inst], minstret"
    //              :  [inst]"=r"(inst)
    //              :
    //              : "memory");
    return inst;
}

uint64_t shl_c906_get_cycle()
{
    uint64_t a = 0;
    asm volatile("csrr %0, cycle"
                 : "=r"(a)
                 :
                 : "memory");
    return a;
}


/*
    index       event                                       counter
    0x1         L1 ICache Access Counter                    mhpmcounter3
    0x2         L1 ICache Miss Counter                      mhpmcounter4
    0x3         I-uTLB Miss Counter                         mhpmcounter5
    0x4         D-uTLB Miss Counter                         mhpmcounter6
    0x5         jTLB Miss Counter                           mhpmcounter7
    0x6         Conditional Branch Mispredict Counter       mhpmcounter8
    0x7         Conditional Branch instruction counter      mhpmcounter9
    0x9         undefine                                    mhpmcounter10-12
    0xb         Store Instruction Counter                   mhpmcounter13
    0xc         L1 DCache read access Counter               mhpmcounter14
    0xd         L1 DCache read miss Counter                 mhpmcounter15
    0xe         L1 DCache write access Counter              mhpmcounter16
    0xf         L1 DCache write miss Counter                mhpmcounter17
    >=0x10      Reserve                                     mhpmcounter18-31
*/

uint64_t shl_c906_get_l1_icache_access()
{
    uint64_t a = 0;
    asm volatile("csrr %0, hpmcounter3"
                 : "=r"(a)
                 :
                 : "memory");
    return a;
}

uint64_t shl_c906_get_l1_icache_miss()
{
    uint64_t a = 0;
    asm volatile("csrr %0, hpmcounter4"
                 : "=r"(a)
                 :
                 : "memory");
    return a;
}

uint64_t shl_c906_get_cb_miss()
{
    uint64_t a = 0;
    asm volatile("csrr %0, hpmcounter8"
                 : "=r"(a)
                 :
                 : "memory");
    return a;
}

uint64_t shl_c906_get_cb_inst()
{
    uint64_t a = 0;
    asm volatile("csrr %0, hpmcounter9"
                 : "=r"(a)
                 :
                 : "memory");
    return a;
}

uint64_t shl_c906_get_store_inst()
{
    uint64_t a = 0;
    asm volatile("csrr %0, hpmcounter13"
                 : "=r"(a)
                 :
                 : "memory");
    return a;
}

uint64_t shl_c906_get_l1_dcache_raccess()
{
    uint64_t a = 0;
    asm volatile("csrr %0, hpmcounter14"
                 : "=r"(a)
                 :
                 : "memory");
    return a;
}

uint64_t shl_c906_get_l1_dcache_rmiss()
{
    uint64_t a = 0;
    asm volatile("csrr %0, hpmcounter15"
                 : "=r"(a)
                 :
                 : "memory");
    return a;
}

uint64_t shl_c906_get_l1_dcache_waccess()
{
    uint64_t a = 0;
    asm volatile("csrr %0, hpmcounter16"
                 : "=r"(a)
                 :
                 : "memory");
    return a;
}

uint64_t shl_c906_get_l1_dcache_wmiss()
{
    uint64_t a = 0;
    asm volatile("csrr %0, hpmcounter17"
                 : "=r"(a)
                 :
                 : "memory");
    return a;
}
