# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import pytest



g_board = ""
g_dtype = ""
g_accuracy = ""
g_flow = ""




def pytest_addoption(parser):
    parser.addoption(
        "--board", action="store", default="c906", help="board option: c906|c908|anole|x86_ref|c920|c920v2"
    )
    parser.addoption(
        "--accuracy", action="store", default="0.99", help="error measures accuracy"
    )
    parser.addoption(
        "--dtype", action="store", default="8", help="8|16|32"
    )
    parser.addoption(
        "--flow", action="store", default="", help=" Run case on Flow with id, Example: Flow_ID"
    )


def pytest_configure(config):
    global g_board
    global g_dtype
    global g_accuracy, g_flow
    g_board = config.getoption("--board")
    g_dtype = config.getoption("--dtype")
    g_accuracy = config.getoption("--accuracy")
    g_flow = config.getoption("--flow")



def id_builder(arg):
    return arg.split("/")[-1].split(".")[0]


def custom_parametrize(*args, **kwargs):
    kwargs.setdefault('ids', id_builder)
    return pytest.mark.parametrize(*args, **kwargs)


def pytest_collection_modifyitems(items):
    # print('pytest cases: \n',items)
    for item in items:
        case_id = item.name
        case_id = case_id.split("[")[-1].split("]")[0]
        # if case_id.isdigit():
        item.user_properties.append(("id", case_id))
