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

# TOPDIR is the tests directory
TOPDIR = os.path.dirname(__file__) + "/../"

python_path = "{TOPDIR_}/python_ref".format(TOPDIR_=TOPDIR)
elf_path = "{TOPDIR_}/validation_layer".format(TOPDIR_=TOPDIR)
valid_dir = "{TOPDIR_}/valid_datas".format(TOPDIR_=TOPDIR)
unit_test_elf_path = "{TOPDIR_}/unit_test".format(TOPDIR_=TOPDIR)


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


def run_base(
        cmd_execute,
        elf_data,
        python_data,
        test_accuracy
):
    hhb_cmd = (
        f"{cmd_execute} "
        f"{elf_data} "
        f"{python_data} "
        f"{test_accuracy} "

    )
    print(hhb_cmd)

    ret = os.system(hhb_cmd)
    assert ret == 0


@pytest.fixture(scope='module')
def compile_execute(cmdopt):
    board = cmdopt["board"]
    accuracy = cmdopt["accuracy"]
    if board == "c860":
        qemu = "qemu-cskyv2 -cpu ck860v"
    elif board == "c906":
        qemu = "qemu-riscv64 -cpu c906fdv"
    elif board == "c910":
        qemu = "qemu-riscv64 -cpu c910v"
    mkdir(valid_dir)
    return qemu, accuracy


@conftest.custom_parametrize('elf_data', numberOffile(elf_path, "c"))
def test_inference(cmdopt, elf_data, compile_execute):
    elf_data = elf_data.replace(".c", ".o.elf")
    if "nchw" or "nhwc" in elf_data:
        python_data = "_".join(elf_data.split("/")[-1].split("_")[:-1])
    else:
        python_data = "_".join(elf_data.split("/")[-1].split("_"))
    os.chdir(valid_dir)
    cmd = "python " + python_path + "/" + python_data + ".py"
    ret = os.system(cmd)
    assert ret == 0
    run_base(compile_execute[0], elf_data, valid_dir + "/" + python_data + "_data_f32.bin", compile_execute[1])


def get_testtype(op_type):
    if "averagepool" in op_type or "maxpool" in op_type:
        test_type = ["random","2x2s2","2x2s2_p1","3x3s2","3x3s2_p1","3x3s1_p1"]
    elif op_type == "convolution":
        test_type = ["random","gemm_conv1x1s1","conv3x3s1_im2col_sgemm","conv3x3s1_winograd64","conv3x3s1_winograd64","gemm_random"]
    elif op_type == "depthwise_convolution":
        test_type = ["random","3x3s1","3x3s2"]
    else:
        test_type =[]
    return test_type

    

@pytest.mark.usefixtures("compile_execute")
class TestCSINN:
    @pytest.mark.parametrize('elf_data', numberOffile(elf_path, "elf"))
    def test_layer(self,elf_data,compile_execute):
        flag = 0
        data = elf_data.split("/")[-1].split(".")[0]
        if "argmax" in data or "argmin" in data:
            path = os.path.join(python_path, data + "_stride.py")
        elif "roipool" in data:
            path = os.path.join(python_path, data + "_caffe.py")
        else:
            path = os.path.join(python_path, data + "_nchw.py")
        if not os.path.exists(path):
            path = os.path.join(python_path, data + ".py")
            flag = 1

        os.chdir(valid_dir)
        if "roipool" in data:
            cmd = f'docker run --rm -v {valid_dir}:mnt tvm_caffe:rfcn sh -c "cd mnt && python3 {path}"'
        else:
            cmd = f"python {path}"
        ret = os.system(cmd)
        assert ret == 0
        if flag == 1:
            run_base(compile_execute[0], elf_data, valid_dir + "/" + data + "_data_f32.bin", compile_execute[1])
        else:
            if "argmax" in data or "argmin" in data:
                run_base(compile_execute[0], elf_data, valid_dir + "/" + data + "_stride_data_f32.bin", compile_execute[1])
            else:
                run_base(compile_execute[0], elf_data, valid_dir + "/" + data + "_nchw_data_f32.bin", compile_execute[1])

    
    @pytest.mark.parametrize('elf_data', numberOffile(elf_path, "elf"))
    def test_rvv_layer(self,elf_data,compile_execute):
        flag = 0
        data = elf_data.split("/")[-1].split(".")[0]
        test_type = get_testtype(data)
        path = os.path.join(python_path, data + "_nchw.py")
        if not os.path.exists(path):
            path = os.path.join(python_path, data + ".py")
            flag = 1 
        if test_type != []:
            for i in test_type:                             
                cmd = f"python {path} {i}"
                ret = os.system(cmd)
                assert ret == 0
                if flag == 1:
                    run_base(compile_execute[0], elf_data, TOPDIR + data + "_data_f32.bin", compile_execute[1])
                else:
                    run_base(compile_execute[0], elf_data, TOPDIR + data + "_nchw_data_f32.bin", compile_execute[1])
        else:             
            cmd = f"python {path}"
            ret = os.system(cmd)
            assert ret == 0
            if flag == 1:
                run_base(compile_execute[0], elf_data, TOPDIR + data + "_data_f32.bin", compile_execute[1])
            else:
                run_base(compile_execute[0], elf_data, TOPDIR + data + "_nchw_data_f32.bin", compile_execute[1])



    @pytest.mark.parametrize('unit_test_elf_data', numberOffile(unit_test_elf_path, "elf"))
    def test_opt_interface(self, unit_test_elf_data, compile_execute):
        run_base(compile_execute[0], unit_test_elf_data, "", compile_execute[1])


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

