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
""" SHL Command Line Tools """
import argparse
import sys
import os

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--whereis',
                        choices=[
                            "base",
                            "c906",
                            "c908",
                            "c920",
                            "rvm",
                            "th1520",
                        ],
                        default='base',
                        help='SHL install directory')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()

    return opt


def get_execute_path():
    if hasattr(sys, "_MEIPASS"):
        execute_path = os.path.dirname(os.path.realpath(sys.executable))
    else:
        execute_path, _ = os.path.split(os.path.abspath(__file__))
        execute_path = os.path.join(execute_path)
    return execute_path


def show_location(target):
    execute_path = get_execute_path()
    if target == "base":
        print("{base}/install_nn2".format(base=execute_path))
    else:
        print("{base}/install_nn2/{lib}".format(base=execute_path, lib=target))


def main(opt):
    show_location(opt.whereis)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
