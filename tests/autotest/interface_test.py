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

import sys
import os
import pytest
import numpy as np
import shutil
import conftest
import subprocess
import itertools
import requests
import json

# TOPDIR is the tests directory
TOPDIR = os.path.dirname(__file__) + "/../"

sys.path.append(TOPDIR + "/onnx_ref/")

from ref import *

python_path = "{TOPDIR_}/python_ref".format(TOPDIR_=TOPDIR)
elf_path = "{TOPDIR_}/validation_layer".format(TOPDIR_=TOPDIR)
valid_dir = "{TOPDIR_}/valid_datas".format(TOPDIR_=TOPDIR)
unit_test_elf_path = "{TOPDIR_}/unit_test".format(TOPDIR_=TOPDIR)


CPU_TYPE = conftest.g_board
DTYPE = conftest.g_dtype
ACC = conftest.g_accuracy
FLOW_ID = conftest.g_flow

Test_Plan_Url = "https://flow.eng.t-head.cn/kamala/api/v1/getPlanDetail?planId"


def mkdir(path):
    """
    :path new folder path
    :return new folder absolute path
    """
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)
    return path


def numberOffile(dirname, suffix):
    """
    :dirname search folder path
    :suffix search suffix
    :return list of files with the specified suffix under the folder
    """
    dirPATH = []
    for path in os.listdir(dirname):
        childDir = os.path.join(dirname, path)
        if os.path.isdir(childDir):
            numberOffile(childDir, suffix)
        else:
            if childDir.split(".")[-1] == suffix:
                dirPATH.append(childDir)
    return dirPATH


def caseParams(dirname, dtype, cpu_type, flow=FLOW_ID):
    temp_case = []
    if flow:
        response = requests.get(Test_Plan_Url + "=" + flow)
        if response.json()['success']:
            cases = response.json()['result']['cases']

            print(f"case num :{len(cases)}")

            for i in range (len(cases)):
                case = cases[i]
                id_flag = str(case["id"])
                case_type = case["filtersConditionDOMap"]["shl_type"][0]["conditionContent"]
                try:
                    case_type = json.loads(case_type)
                    for key, value in case_type.items():
                        case_name = key
                        if isinstance(value, dict):
                            for k,v in value.items():                         
                                if v.get("layout", "nchw") == "nhwc":
                                    elf_name = f"{case_name}_nhwc"
                                else:
                                    elf_name = case_name
                                elf_data = os.path.join(elf_path, f"{elf_name}.o.elf")
                                temp_case.append(pytest.param((elf_data, case_name, v), id=id_flag))
                        else:
                            continue
                except:
                    return temp_case
    else:
        return temp_case
    return temp_case


def run_base(
        cmd_execute,
        elf_data,
        python_data,
        test_accuracy,
        python_cmd,
):
    cmd = (
        f"{cmd_execute} "
        f"{elf_data} "
        f"{python_data} "
        f"{test_accuracy} "

    )
    if python_cmd != "":
        print(cmd)
        ret = subprocess.run(python_cmd, shell=True, timeout=300, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        print(python_cmd)
        p_out = ret.stdout.decode("utf-8")
        assert ret.returncode == 0
    else:
        p_out = ""

    ret = subprocess.run(cmd, shell=True, timeout=300, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out = ret.stdout.decode("utf-8")

    if ret != 0:
        print(ret)
        err = ret.stderr.decode("utf-8")
    # out = out
    assert ret.returncode == 0, f"\nexecute cmd:\n{cmd}\ngenerate python:\n{python_cmd}\n{p_out}out:\n{out}\nerr:\n{err}"


class Test_CSINN_Base:
    def setup_class(self):
        vlen = os.environ.get("vlen", "128")
        self.accuracy = ACC
        self.dtype = DTYPE
        if CPU_TYPE == "c906":
            qemu = f"qemu-riscv64 -cpu c906fdv,vlen={vlen}"
        elif CPU_TYPE == "c920":
            qemu = f"qemu-riscv64 -cpu c920,vlen={vlen}"
        elif CPU_TYPE == "c920v2":
            qemu = f"qemu-riscv64 -cpu c920v2,vlen={vlen}"
        elif CPU_TYPE == "c908":
            qemu = f"qemu-riscv64 -cpu c908v,vlen={vlen}"
        elif CPU_TYPE == "rvm":
            qemu = "qemu-riscv64 -cpu c907fdvm,rlen=128"
        elif CPU_TYPE == "rvv":
            qemu = f"qemu-riscv64 -cpu max,vlen={vlen}"

        self.qemu = qemu



class TestCSINN(Test_CSINN_Base):
    #####TODO fix###########
    @pytest.mark.parametrize('test_data', caseParams(elf_path, DTYPE, CPU_TYPE))
    def test_layer(self, test_data):   
        python_data = test_data[1:]
        if test_data[1] == "convolution" or test_data[1] == "group_convolution" or test_data[1] == "depthwise_convolution":
            convolution(python_data)
        elif test_data[1] == "convolution_relu":
            convolution_relu(python_data)
        elif test_data[1] == "maxpool":
            maxpool(python_data)
        elif test_data[1] == "averagepool":
            averagepool(python_data)
        elif test_data[1] == "pad":
            pad(python_data)
        elif test_data[1] == "add" or test_data[1] == "div" or test_data[1] == "mul" or test_data[1] == "sub":
            binary_b(python_data)
        elif test_data[1] == "deconvolution" or test_data[1] == "depthwise_deconvolution":
            deconvolution(python_data)
        elif test_data[1] == "global_avgpool":
            global_avgpool(python_data)
        elif test_data[1] == "global_maxpool":
            global_maxpool(python_data)
        elif test_data[1] == "fullyconnected":
            fullyconnected(python_data)
        elif test_data[1] == "abs" or test_data[1] == "relu" or test_data[1] == "erf" or test_data[1] == "sigmoid":
            unary(python_data)
        elif test_data[1] == "relu1" or test_data[1] == "relu6":
            thresholdedrelu(python_data)  
        elif test_data[1] == "minimum":
            muti_min(python_data)  
        elif test_data[1] == "strided_slice":
            strided_slice(python_data)
        elif test_data[1] == "reduce_sum":
            reduce_op(python_data) 
        elif test_data[1] == "reshape":
            reshape(python_data) 
        elif test_data[1] == "silu":
            silu(python_data)
        elif test_data[1] == "clip":
            clip(python_data)
        elif test_data[1] == "concat":
            concat(python_data)
        elif test_data[1] == "leaky_relu":
            leaky_relu(python_data)
        elif test_data[1] == "gather":
            gather(python_data)
        elif test_data[1] == "matmul":
            matmul(python_data)
        elif test_data[1] == "prelu":
            prelu(python_data)
        elif test_data[1] == "rms_norm":
            rms_norm(python_data)
        elif test_data[1] == "split":
            split(python_data)
        elif test_data[1] == "transpose":
            transpose(python_data)
        elif test_data[1] == "lrn":
            lrn(python_data)
        elif test_data[1] == "convolution1d":
            convolution1d(python_data)
        elif test_data[1] == "softmax":
            softmax(python_data)
        elif test_data[1] == "layer_norm":
            layer_norm(python_data)
        else:
            return 
        
        run_base(self.qemu, test_data[0], TOPDIR + test_data[1] + "_test_data_f32.bin", self.accuracy, "")


    @pytest.mark.parametrize('unit_test_elf_data', numberOffile(unit_test_elf_path, "elf"))
    def test_opt_interface(self, unit_test_elf_data):
        run_base(self.qemu, unit_test_elf_data, "", self.accuracy, "")


class TestHeterogeneous:
    def test_subgraph_fuse(self):
        hlight_test_dir = os.path.join(TOPDIR, "validation_graph", "hlight")
        compile_cmd = f"make -C {hlight_test_dir}"

        ret = os.system(compile_cmd)
        assert ret == 0, "Compiling subgraph fusion test fails."

        os.chdir(hlight_test_dir)
        exec_cmd = f"./run.sh"
        ret = os.system(exec_cmd)
        assert ret == 0, "Execute subgraph fusion test fails"


class TestTVMGen:
    def test_tvmgen(self):
        tvmgen_test_dir = os.path.join(TOPDIR, "validation_graph", "tvmgen")
        compile_cmd = f"make -C {tvmgen_test_dir}"

        ret = os.system(compile_cmd)
        assert ret == 0, "Compiling tvmgen tests fails."

        os.chdir(tvmgen_test_dir)
        exec_cmd = f"./run.sh"
        ret = os.system(exec_cmd)
        assert ret == 0, "Execute tvmgen tests fails"

