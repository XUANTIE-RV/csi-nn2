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
    cmd = (
        f"{cmd_execute} "
        f"{elf_data} "
        f"{python_data} "
        f"{test_accuracy} "

    )
    if python_cmd != "":
        print(cmd)
        ret = subprocess.run(python_cmd, shell=True, timeout=10, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        p_out = ret.stdout.decode("utf-8")
        assert ret.returncode == 0
    else:
        p_out = "" 

    ret = subprocess.run(cmd, shell=True, timeout=10, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out = ret.stdout.decode("utf-8")
    err = ret.stderr.decode("utf-8")
    out = out + err
    # print(out.append(err))
    # out = '\n'.join(out)
    pytest.assume(ret.returncode == 0, f"\nexecute cmd:\n{cmd}\ngenerate python:\n{python_cmd}\n{p_out}out:\n{out}")


import itertools
def get_testvlen(op_type, vlen, qemu_info):
    list_dtype = [int(vlen)]
    if "c908" in qemu_info:
        list_vlen = [128, 256, 512]
    else:
        list_vlen = [128]
    if op_type == "convolution":
        list_type = ["pack1_com", "pack1_gemm", "packnto1", "packnto1_conv1x1s1", "pack1ton", "pack1ton_conv1x1s1", "packn_com", "packn_conv1x1s1", "packn_conv3x3s1", "packn_conv3x3s1_linput"]
        test_type = list(itertools.product(list_dtype, list_vlen, list_type))
    elif op_type == "group_convolution":
        list_type = ["packn_conv3x3s1d1"]
        test_type = list(itertools.product(list_dtype, list_vlen, list_type))
    elif op_type == "depthwise_convolution":
        list_type = ["pack1_common", "pack1_conv3x3s2", "pack1_conv3x3s1", "packn_com", "packn_conv3x3s2", "packn_conv3x3s1"]
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


class Test_CSINN_Base:
    def setup_class(self):        

        board = conftest.board
        self.accuracy = conftest.accuracy
        self.dtype = conftest.dtype
        if board == "c860":
            qemu = "qemu-cskyv2 -cpu ck860v"
        elif board == "c906":
            qemu = "qemu-riscv64 -cpu c906fdv"
        elif board == "c920":
            qemu = "qemu-riscv64 -cpu c920"
        elif board == "c908":
            qemu = "qemu-riscv64 -cpu c908v"
        
        self.qemu = qemu



class TestCSINN(Test_CSINN_Base):
    #####TODO rm###########
    @pytest.mark.parametrize('elf_data', numberOffile(elf_path, "elf"))
    def test_layer(self,elf_data):
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

        mkdir(valid_dir)
        os.chdir(valid_dir)
        if "roipool" in data:
            cmd = f'docker run --rm -v {valid_dir}:mnt tvm_caffe:rfcn sh -c "cd mnt && python3 {path}"'
        else:
            cmd = f"python3 {path}"
        if flag == 1:
            run_base(self.qemu, elf_data, valid_dir + "/" + data + "_data_f32.bin", self.accuracy, cmd)
        else:
            if "argmax" in data or "argmin" in data:
                run_base(self.qemu, elf_data, valid_dir + "/" + data + "_stride_data_f32.bin", self.accuracy, cmd)
            else:
                run_base(self.qemu, elf_data, valid_dir + "/" + data + "_nchw_data_f32.bin", self.accuracy, cmd)

    
    @pytest.mark.parametrize('elf_data', numberOffile(elf_path, "elf"))
    def test_rvv_layer(self,elf_data):
        flag = 0
        data = elf_data.split("/")[-1].split(".")[0]
        test_type = get_testvlen(data, self.dtype, self.qemu)
        compile_option = self.qemu
        path = os.path.join(python_path, data + "_nchw.py")
        if not os.path.exists(path):
            path = os.path.join(python_path, data + ".py")
            flag = 1
        elif "convolution" in path or "pool" in path:
            path = os.path.join(python_path, data + "_vlen.py")
        if test_type != []:
            for i in test_type:
                cmd = f"python3 {path} {i[0]} {i[1]} {i[2]}"
                if str(i[1]) == "256":
                    compile_option = "qemu-riscv64  -cpu rv64,x-v=true,vext_spec=v1.0,vlen=256,x-thead=true"
                elif str(i[1]) == "512":
                    compile_option = "qemu-riscv64  -cpu rv64,x-v=true,vext_spec=v1.0,vlen=512,x-thead=true"

                if flag == 1:
                    run_base(compile_option, elf_data, TOPDIR + data + "_data_f32.bin", self.accuracy, cmd)
                else:
                    run_base(compile_option, elf_data, TOPDIR + data + "_nchw_data_f32.bin", self.accuracy, cmd)
        else:
            cmd = f"python3 {path}"
            if flag == 1:
                run_base(compile_option, elf_data, TOPDIR + data + "_data_f32.bin", self.accuracy, cmd)
            else:
                run_base(compile_option, elf_data, TOPDIR + data + "_nchw_data_f32.bin", self.accuracy, cmd)




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

