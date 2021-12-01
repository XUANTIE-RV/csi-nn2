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
elf_path = "{TOPDIR_}/validation".format(TOPDIR_=TOPDIR)
valid_dir = "{TOPDIR_}/valid_datas".format(TOPDIR_=TOPDIR)


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
        qemu = "qemu-riscv64"
    os.system("make clean;make -j16 test_" + board)
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
