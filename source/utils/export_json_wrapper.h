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

#ifndef INCLUDE_EXPORT_JSON_WRAPPER_H_
#define INCLUDE_EXPORT_JSON_WRAPPER_H_

#ifdef SHL_EXPORT_MODEL

#ifdef __cplusplus
extern "C" {
#endif

int shl_export_json_internal(struct csinn_session* sess, char* path);

#ifdef __cplusplus
}
#endif

#endif

#endif  // INCLUDE_EXPORT_JSON_WRAPPER_H_