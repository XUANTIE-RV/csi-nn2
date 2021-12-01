/*
*  Copyright (C) 2016-2021 PingTouGe Semiconductor Co., Ltd  Limited. All rights reserved.
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

#ifndef _UTILS_H

#define weak_alias(name, aliasname)  extern __typeof(name) aliasname __attribute__((weak, alias(#name)))

#endif /* _UTILS_H */

/**
 *
 * End of file.
 */
