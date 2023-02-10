import os
import ctypes

import numpy as np
from torch import tensor
from torch.nn import functional as fn
from utils import BuildSo
from utils import csinn_tensor, csinn_conv2d_params
from utils import csinn_quant_info, csinn_params_base

TEST_ERROR = Exception("test deconv2d error!")

np.random.seed(0)


def deconv2d_py(in_channel, stride, group):
    # init the input data and parameters
    batch = 1
    in_size_x = 10
    in_size_y = 10

    zero_point1 = int(np.random.randint(-6, high=6, size=1))
    std1 = int(np.random.randint(1, high=20, size=1))
    zero_point2 = int(np.random.randint(-6, high=6, size=1))
    std2 = int(np.random.randint(1, high=20, size=1))
    zero_point3 = int(np.random.randint(-6, high=6, size=1))
    std3 = int(np.random.randint(1, high=20, size=1))

    weight_n = 3
    weight_x = 3
    weight_y = 3

    inputs = np.random.normal(
        zero_point1, std1, (batch, in_channel, in_size_y, in_size_x)
    )
    inputs = inputs.astype(np.float32)

    weight = np.random.normal(
        zero_point2, std2, (in_channel, weight_n, weight_y, weight_x)
    )
    weight = weight.astype(np.float32)

    bias = np.random.normal(zero_point3, std3, (weight_n * group))
    bias = bias.astype(np.float32)

    torch_out = fn.conv_transpose2d(
        tensor(inputs),
        weight=tensor(weight),
        bias=tensor(bias),
        stride=stride,
        groups=group,
    ).numpy()

    return inputs, weight, bias, torch_out


def deconv2d_ref(inputs, weight, bias, out_shape, lib_path, stride, group):
    # load share lib
    lib = ctypes.cdll.LoadLibrary(lib_path)

    # create test params
    shl_gref_deconv2d = lib.shl_ref_group_deconv2d_f32
    shl_gref_deconv2d.restype = ctypes.c_int
    shl_gref_deconv2d.argtypes = [
        ctypes.POINTER(csinn_tensor),
        ctypes.POINTER(csinn_tensor),
        ctypes.POINTER(csinn_tensor),
        ctypes.POINTER(csinn_tensor),
        ctypes.POINTER(csinn_conv2d_params),
    ]

    ouptut = np.zeros(out_shape, dtype=np.float32)
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

    weight_tensor = csinn_tensor(
        data=weight.ctypes.data,
        dtype=10,  # CSINN_DTYPE_FLOAT32
        mtype=0,  # CSINN_MEM_TYPE_CPU_NOT_ALIGNED
        dim=tuple(weight.shape),
        dim_count=len(weight.shape),
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
        dim=tuple(bias.shape),
        dim_count=len(bias.shape),
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
        dim=tuple(out_shape),
        dim_count=len(out_shape),
        is_const=0,
        name=b"output_tensor",
        layout=4,  # CSINN_LAYOUT_NCHW
        quant_channel=1,
        qinfo=ctypes.pointer(qinfo),
        sess=None,
    )

    base = csinn_params_base(
        layout=4,  # CSINN_LAYOUT_NCHW
    )
    params = csinn_conv2d_params(
        base=base,
        group=group,
        stride_height=stride,
        stride_width=stride,
        pad_left=0,
        pad_top=0,
    )

    # run interface
    result = shl_gref_deconv2d(
        ctypes.byref(inputs_tensor),
        ctypes.byref(output_tensor),
        ctypes.byref(weight_tensor),
        ctypes.byref(bias_tensor),
        ctypes.byref(params),
    )

    if result != 1:
        assert 0, TEST_ERROR

    return ouptut


def _test_deconv2d(so_path, in_channel, group, stride):
    # get inputs and outputs
    inputs, weight, bias, torch_out = deconv2d_py(in_channel, stride, group)
    output = deconv2d_ref(inputs, weight, bias, torch_out.shape, so_path, stride, group)
    np.testing.assert_array_almost_equal(output, torch_out, 4)


def test_deconv2d():
    src_file = os.path.abspath("../../source/reference/deconvolution.c")

    with BuildSo(src_file) as builder:
        _test_deconv2d(builder.so_path, 10, 2, 2)
        _test_deconv2d(builder.so_path, 12, 3, 2)
        _test_deconv2d(builder.so_path, 12, 3, 3)


if __name__ == "__main__":
    test_deconv2d()
