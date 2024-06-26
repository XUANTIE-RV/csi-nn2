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

#include <math.h>
#include <stdlib.h>

#include "shl_profiler.h"

#define TEST_WRAPPER(func, msg)                  \
    {                                            \
        printf("Testing: %s\n", msg);            \
        int fail = func;                         \
        if (fail > 0) {                          \
            printf("Testing: %s - fail\n", msg); \
        } else {                                 \
            printf("Testing: %s - Pass\n", msg); \
        }                                        \
    }

int _test_shl_trace_create_string(const char *data)
{
    int ret = 0;
    struct shl_trace_value *value = shl_trace_create_string(data);
    if (value->type != SHL_TRACE_VALUE_TYPE_STRING || strcmp(value->content.str, data) != 0) {
        printf("%s: %s fail\n", __func__, data);
        ret = 1;
    }
    shl_trace_release_value(value);
    return ret;
}

int _test_shl_trace_create_int64(int64_t data)
{
    int ret = 0;
    struct shl_trace_value *value = shl_trace_create_int64(data);
    if (value->type != SHL_TRACE_VALUE_TYPE_INT64 || value->content.i64 != data) {
        printf("%s: %ld fail\n", __func__, data);
        ret = 1;
    }
    shl_trace_release_value(value);
    return ret;
}

int _test_shl_trace_create_uint64(uint64_t data)
{
    int ret = 0;
    struct shl_trace_value *value = shl_trace_create_uint64(data);
    if (value->type != SHL_TRACE_VALUE_TYPE_UINT64 || value->content.u64 != data) {
        printf("%s: %lu fail\n", __func__, data);
        ret = 1;
    }
    shl_trace_release_value(value);
    return ret;
}

int _test_shl_trace_create_double(double data)
{
    int ret = 0;
    struct shl_trace_value *value = shl_trace_create_double(data);
    if (value->type != SHL_TRACE_VALUE_TYPE_DOUBLE || fabs(value->content.f64 - data) > 1e-10) {
        printf("%s: %f fail\n", __func__, data);
        ret = 1;
    }
    shl_trace_release_value(value);
    return ret;
}

int _test_shl_trace_create_list()
{
    int ret = 0;
    struct shl_trace_value *value =
        shl_trace_create_list(4, shl_trace_create_string("data"), shl_trace_create_int64(-100),
                              shl_trace_create_uint64(100), shl_trace_create_double(3.3));
    if (value->type != SHL_TRACE_VALUE_TYPE_LIST || value->content.list->size != 4) {
        printf("%s: wrong type(%d) or size(%d) fail\n", __func__, value->type,
               value->content.list->size);
        ret = 1;
    }
    struct shl_trace_value_list *list = value->content.list;
    ret += _test_shl_trace_create_string(list->value[0]->content.str);
    ret += _test_shl_trace_create_int64(list->value[0]->content.i64);
    ret += _test_shl_trace_create_uint64(list->value[0]->content.u64);
    ret += _test_shl_trace_create_double(list->value[0]->content.f64);

    shl_trace_release_value(value);
    return ret > 0;
}

int test_shl_trace_value()
{
    int fail_num = 0;
    fail_num += _test_shl_trace_create_string("data");
    fail_num += _test_shl_trace_create_int64(-1);
    fail_num += _test_shl_trace_create_uint64(2);
    fail_num += _test_shl_trace_create_double(2.5);
    fail_num += _test_shl_trace_create_list();

    return fail_num;
}

int test_shl_trace_dict()
{
    int fail_num = 0;
    struct shl_trace_dict *dict;
    dict = shl_trace_create_dict(
        5, "string", shl_trace_create_string("string"), "int64", shl_trace_create_int64(-5),
        "uint64", shl_trace_create_uint64(4), "double", shl_trace_create_double(4.6), "list",
        shl_trace_create_list(2, shl_trace_create_int64(256), shl_trace_create_int64(256)));
    if (dict && dict->items_size == 5) {
        if (strcmp(dict->items[0]->key, "string") != 0 ||
            strcmp(dict->items[1]->key, "int64") != 0 ||
            strcmp(dict->items[2]->key, "uint64") != 0 ||
            strcmp(dict->items[3]->key, "double") != 0 ||
            strcmp(dict->items[4]->key, "list") != 0) {
            printf("Wrong item key...\n");
            fail_num++;
        }
    } else {
        fail_num++;
    }

    shl_trace_release_dict(dict);
    return fail_num;
}

int test_shl_trace_begin_end()
{
    int fail_num = 0;
    struct shl_trace *trace = (struct shl_trace *)shl_mem_alloc(sizeof(struct shl_trace));

    shl_trace_begin(trace, "trace.json");
    if (trace->events != NULL || trace->events_capacity != 0 || trace->events_size != 0 ||
        strcmp(trace->filename, "trace.josn") == 0) {
        printf("should'n initialize trace while enable_trace is false...\n");
        fail_num++;
    }

    trace->enable_trace = true;
    shl_trace_begin(trace, "trace.json");
    if (trace->is_init == false || strcmp(trace->filename, "trace.json") != 0 ||
        trace->events == NULL || trace->events_capacity == 0 || trace->events_size != 0) {
        printf("fail to initialize trace...\n");
        fail_num++;
    }

    shl_trace_end(trace);
    if (trace->is_init != false || trace->events_capacity != 0 || trace->events_size != 0 ||
        trace->events != NULL) {
        printf("fail to deinit trace...\n");
        fail_num++;
    }
    shl_mem_free(trace);
    return fail_num;
}

void func_procedure()
{
    // simulate doing something
    sleep(1);
}

int test_shl_trace_end2end()
{
    int fail_num = 0;

    struct shl_trace *trace = (struct shl_trace *)shl_mem_alloc(sizeof(struct shl_trace));
    trace->enable_trace = true;

    SHL_TRACE_CALL(shl_trace_begin(trace, NULL));

    // generate custom data into otherData
    SHL_TRACE_CALL(shl_trace_other_data(
        trace,
        shl_trace_create_dict(2, "hardware", SHL_TRACE_STRING("x86"), "key", SHL_TRACE_INT64(10))));

    int tmp[] = {1, 3, 224, 224};
    SHL_TRACE_CALL(shl_trace_duration_begin(
        trace, "func_procedure", SHL_TRACE_EVENT_CPU_OPERATOR,
        shl_trace_create_dict(4, "type", shl_trace_create_string("csinn cpu"), "shape",
                              shl_trace_create_list(3, SHL_TRACE_INT64(3), SHL_TRACE_INT64(112),
                                                    SHL_TRACE_INT64(112)),
                              "dim", SHL_TRACE_LIST_INT(4, tmp), "input_shape",
                              SHL_TRACE_LIST(1, SHL_TRACE_LIST_INT(4, tmp)))));

    func_procedure();

    SHL_TRACE_CALL(
        shl_trace_duration_end(trace, "func_procedure", SHL_TRACE_EVENT_CPU_OPERATOR, NULL));

    SHL_TRACE_CALL(shl_trace_end(trace));

    shl_mem_free(trace);
    return fail_num;
}

int main()
{
    TEST_WRAPPER(test_shl_trace_value(), "shl_trace_value: create/release");
    TEST_WRAPPER(test_shl_trace_dict(), "shl_trace_dict: create/release dict_item/dict");
    TEST_WRAPPER(test_shl_trace_begin_end(), "shl_trace_begin_end");

    TEST_WRAPPER(test_shl_trace_end2end(), "shl_trace_end2end");

    return 0;
}