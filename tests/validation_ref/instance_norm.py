import os
import ctypes

import numpy as np
from torch import tensor
from torch.nn import functional as fn
from utils import BuildSo
from utils import csinn_tensor, csinn_instance_norm_params
from utils import csinn_quant_info, csinn_params_base

TEST_ERROR = Exception("test instance_norm error!")


def instance_norm_py():
    # init the input data and parameters
    batch = int(np.random.randint(1, high=4, size=1))
    in_channel = int(np.random.randint(1, high=32, size=1))
    in_size_x = int(np.random.randint(32, high=64, size=1))
    in_size_y = int(np.random.randint(32, high=64, size=1))
    zero_point1 = int(np.random.randint(-6, high=6, size=1))
    std1 = int(np.random.randint(1, high=20, size=1))
    zero_point2 = int(np.random.randint(-6, high=6, size=1))
    std2 = int(np.random.randint(1, high=20, size=1))
    zero_point3 = int(np.random.randint(-6, high=6, size=1))
    std3 = int(np.random.randint(1, high=20, size=1))

    inputs = np.random.normal(
        zero_point1, std1, (batch, in_channel, in_size_y, in_size_x)
    )
    inputs = inputs.astype(np.float32)

    scales = np.random.normal(zero_point2, std2, (in_channel))
    scales = scales.astype(np.float32)

    bias = np.random.normal(zero_point3, std3, (in_channel))
    bias = bias.astype(np.float32)

    torch_out = fn.instance_norm(
        tensor(inputs), weight=tensor(scales), bias=tensor(bias)
    ).numpy()

    return inputs, scales, bias, torch_out


def instance_norm_ref(inputs, scales, bias, lib_path):
    # load share lib
    lib = ctypes.cdll.LoadLibrary(lib_path)

    # create test params
    shl_gref_instance_norm = lib.shl_ref_instance_norm_f32
    shl_gref_instance_norm.restype = ctypes.c_int
    shl_gref_instance_norm.argtypes = [
        ctypes.POINTER(csinn_tensor),
        ctypes.POINTER(csinn_tensor),
        ctypes.POINTER(csinn_tensor),
        ctypes.POINTER(csinn_tensor),
        ctypes.POINTER(csinn_instance_norm_params),
    ]

    ouptut = np.zeros_like(inputs, dtype=np.float32)
    qinfo = csinn_quant_info(scale=1, zero_point=0)

    inputs_tensor = csinn_tensor(
        data=inputs.ctypes.data,
        dtype=10,  # CSINN_DTYPE_FLOAT32
        mtype=0,  # CSINN_MEM_TYPE_CPU_NOT_ALIGNED
        dim=tuple(inputs.shape),
        dim_count=len(inputs.shape),
        is_const=0,
        name=b"input_tensor",
        layout=4,  # CSINN_LAYOUT_NCHW
        quant_channel=1,
        qinfo=ctypes.pointer(qinfo),
        sess=None,
    )

    scales_tensor = csinn_tensor(
        data=scales.ctypes.data,
        dtype=10,  # CSINN_DTYPE_FLOAT32
        mtype=0,  # CSINN_MEM_TYPE_CPU_NOT_ALIGNED
        dim=tuple(inputs.shape),
        dim_count=len(inputs.shape),
        is_const=0,
        name=b"scale_tensor",
        layout=4,  # CSINN_LAYOUT_NCHW
        quant_channel=1,
        qinfo=ctypes.pointer(qinfo),
        sess=None,
    )

    bias_tensor = csinn_tensor(
        data=bias.ctypes.data,
        dtype=10,  # CSINN_DTYPE_FLOAT32
        mtype=0,  # CSINN_MEM_TYPE_CPU_NOT_ALIGNED
        dim=tuple(inputs.shape),
        dim_count=len(inputs.shape),
        is_const=0,
        name=b"bias_tensor",
        layout=4,  # CSINN_LAYOUT_NCHW
        quant_channel=1,
        qinfo=ctypes.pointer(qinfo),
        sess=None,
    )
    output_tensor = csinn_tensor(
        data=ouptut.ctypes.data,
        dtype=10,  # CSINN_DTYPE_FLOAT32
        mtype=0,  # CSINN_MEM_TYPE_CPU_NOT_ALIGNED
        dim=tuple(inputs.shape),
        dim_count=len(inputs.shape),
        is_const=0,
        name=b"output_tensor",
        layout=4,  # CSINN_LAYOUT_NCHW
        quant_channel=1,
        qinfo=ctypes.pointer(qinfo),
        sess=None,
    )

    base = csinn_params_base()
    params = csinn_instance_norm_params(
        base=base,
        epsilon=1e-5,
    )

    # run interface
    result = shl_gref_instance_norm(
        ctypes.byref(inputs_tensor),
        ctypes.byref(scales_tensor),
        ctypes.byref(bias_tensor),
        ctypes.byref(output_tensor),
        ctypes.byref(params),
    )

    if result != 1:
        assert 0, TEST_ERROR

    return ouptut


def _test_instance_norm(so_path):
    # get inputs and outputs
    inputs, scales, bias, torch_out = instance_norm_py()
    output = instance_norm_ref(inputs, scales, bias, so_path)

    np.testing.assert_array_almost_equal(output, torch_out, 3)


def test_instance_norm():
    src_file = os.path.abspath("../../source/reference/instance_norm.c")

    with BuildSo(src_file) as builder:
        for _ in range(10):
            _test_instance_norm(builder.so_path)


if __name__ == "__main__":
    test_instance_norm()
