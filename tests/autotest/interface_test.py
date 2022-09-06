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
        test_accuracy,
        python_cmd,
):
    hhb_cmd = (
        f"{cmd_execute} "
        f"{elf_data} "
        f"{python_data} "
        f"{test_accuracy} "

    )
    print(hhb_cmd)

    ret = os.system(hhb_cmd)
    pytest.assume(ret == 0, f"{hhb_cmd}\n{python_cmd}")


@pytest.fixture(scope='module')
def compile_execute(cmdopt):
    board = cmdopt["board"]
    accuracy = cmdopt["accuracy"]
    vlen = cmdopt["vlen"]
    if board == "c860":
        qemu = "qemu-cskyv2 -cpu ck860v"
    elif board == "c906":
        qemu = "qemu-riscv64 -cpu c906fdv"
    elif board == "c910":
        qemu = "qemu-riscv64 -cpu c910v"
    elif board == "c908":
        qemu = "qemu-riscv64 -cpu c908v"
    mkdir(valid_dir)
    return qemu, accuracy, vlen


####TODO rm ###########
# def get_testtype(op_type):
#     if "averagepool" in op_type or "maxpool" in op_type:
#         test_type = ["random","2x2s2","2x2s2_p1","3x3s2","3x3s2_p1","3x3s1_p1"]
#     elif op_type == "convolution":
#         test_type = ["random","gemm_conv1x1s1","conv3x3s1_im2col_sgemm","conv3x3s1_winograd64","conv3x3s1_winograd64","conv3x3s1_winograd64_pack","gemm_random"]
#     elif op_type == "depthwise_convolution":
#         test_type = ["random","3x3s1","3x3s2"]
#     elif op_type == "group_convolution":
#         test_type = ["random", "conv3x3s1d1"]
#     elif op_type == "relu":
#         test_type = ["random", "16x3_8_4_2_1"]
#     elif op_type == "add":
#         test_type = ["", "vector", "size1", "flag0"]
#     else:
#         test_type =[]
#     return test_type

import itertools
def get_testvlen(op_type, vlen):
    list_dtype = [int(vlen)]
    list_vlen = [128, 256, 512]
    if op_type == "convolution":
        list_type = ["pack1_com", "pack1_gemm", "packnto1", "packnto1_conv1x1s1", "pack1ton", "pack1ton_conv1x1s1", "packn_com", "packn_conv1x1s1", "packn_conv3x3s1", "packn_conv3x3s1_linput"]
        test_type = list(itertools.product(list_dtype, list_vlen, list_type))
    elif op_type == "group_convolution":
        list_type = ["pack1ton_conv1x1s1"]
        test_type = list(itertools.product(list_dtype, list_vlen, list_type))
    elif op_type == "depthwise_convolution":
        list_type = ["pack1_common", "pack1_conv3x3s2", "pack1_conv3x3s1", "packnto1", "pack1ton", "packn_com", "packn_conv3x3s2", "packn_conv3x3s1"]
        test_type = list(itertools.product(list_dtype, list_vlen, list_type))
    elif op_type == "global_avgpool" or op_type == "global_maxpool":
        list_type = ["packn", "pack1"]
        test_type = list(itertools.product(list_dtype, list_vlen, list_type))
    elif op_type == "averagepool" or op_type == "maxpool":
        list_type = ["packn_global", "global", "packn_2x2s2", "pack1_2x2s2", "packn_2x2s2p0", "pack1_2x2s2p0", "packn_2x2s2p1", "pack1_2x2s2p1", "packn_3x3s2", "pack1_3x3s2", "packn_3x3s2p0", "pack1_3x3s2p0", "packn_3x3s2p1", "pack1_3x3s2p1", "packn_3x3s1_p1", "pack1_3x3s1_p1"]
        test_type = list(itertools.product(list_dtype, list_vlen, list_type))
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
            cmd = f"python3 {path}"
        ret = os.system(cmd)
        assert ret == 0
        if flag == 1:
            run_base(compile_execute[0], elf_data, valid_dir + "/" + data + "_data_f32.bin", compile_execute[1], cmd)
        else:
            if "argmax" in data or "argmin" in data:
                run_base(compile_execute[0], elf_data, valid_dir + "/" + data + "_stride_data_f32.bin", compile_execute[1], cmd)
            else:
                run_base(compile_execute[0], elf_data, valid_dir + "/" + data + "_nchw_data_f32.bin", compile_execute[1], cmd)


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
                cmd = f"python3 {path} {i}"
                print(cmd)
                ret = os.system(cmd)
                assert ret == 0
                if flag == 1:
                    run_base(compile_execute[0], elf_data, TOPDIR + data + "_data_f32.bin", compile_execute[1], cmd)
                else:
                    run_base(compile_execute[0], elf_data, TOPDIR + data + "_nchw_data_f32.bin", compile_execute[1], cmd)
        else:
            cmd = f"python3 {path}"
            ret = os.system(cmd)
            assert ret == 0
            if flag == 1:
                run_base(compile_execute[0], elf_data, TOPDIR + data + "_data_f32.bin", compile_execute[1], cmd)
            else:
                run_base(compile_execute[0], elf_data, TOPDIR + data + "_nchw_data_f32.bin", compile_execute[1], cmd)


    @pytest.mark.parametrize('elf_data', numberOffile(elf_path, "elf"))
    def test_c908_layer(self,elf_data,compile_execute):
        flag = 0
        data = elf_data.split("/")[-1].split(".")[0]
        test_type = get_testvlen(data, compile_execute[2])
        compile_option = compile_execute[0]
        path = os.path.join(python_path, data + "_nchw.py")
        if not os.path.exists(path):
            path = os.path.join(python_path, data + ".py")
            flag = 1
        elif "convolution" in path or "averagepool" in path or "maxpool" in path:
            path = os.path.join(python_path, data + "_vlen.py")
        if test_type != []:
            for i in test_type:
                cmd = f"python3 {path} {i[0]} {i[1]} {i[2]}"
                print(cmd)
                ret = os.system(cmd)
                pytest.assume(ret == 0)
                if str(i[1]) == "256":
                    compile_option = "qemu-riscv64  -cpu rv64,x-v=true,vext_spec=v1.0,vlen=256,x-thead=true"
                elif str(i[1]) == "512":
                    compile_option = "qemu-riscv64  -cpu rv64,x-v=true,vext_spec=v1.0,vlen=512,x-thead=true"

                if flag == 1:
                    run_base(compile_option, elf_data, TOPDIR + data + "_data_f32.bin", compile_execute[1], cmd)
                else:
                    run_base(compile_option, elf_data, TOPDIR + data + "_nchw_data_f32.bin", compile_execute[1], cmd)
        else:
            cmd = f"python3 {path}"
            ret = os.system(cmd)
            pytest.assume(ret == 0)
            if flag == 1:
                run_base(compile_option, elf_data, TOPDIR + data + "_data_f32.bin", compile_execute[1], cmd)
            else:
                run_base(compile_option, elf_data, TOPDIR + data + "_nchw_data_f32.bin", compile_execute[1], cmd)




    @pytest.mark.parametrize('unit_test_elf_data', numberOffile(unit_test_elf_path, "elf"))
    def test_opt_interface(self, unit_test_elf_data, compile_execute):
        run_base(compile_execute[0], unit_test_elf_data, "", compile_execute[1], "")


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

