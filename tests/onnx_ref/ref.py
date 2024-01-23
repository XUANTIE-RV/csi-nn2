import onnx
# from onnx import helper, save_model
from onnx import AttributeProto, TensorProto, GraphProto
import onnxruntime
import numpy as np
from onnx_utlis import run
import os
import struct
import json
import math
import torch
from torch import tensor
from torch.nn import functional as fn
import tensorflow as tf

# TOPDIR is the scripts directory
TOPDIR = os.path.dirname(__file__) + "/../"


def convolution(test_data):
    data = test_data[1]
    out_name = test_data[0]
    para = []
    x_shape = data["X"]
    W_shape = data["W"]
    p_shape = data["pads"]
    s_shape = data["strides"]
    d_shape = data["dilations"]
    g = data["group"]
    auto_pad = data["auto_pad"]

    x = np.random.randn(*x_shape).astype(np.float32)
    W = np.random.randn(*W_shape).astype(np.float32)

    if data.get("B",""):
        B_shape = [data.get("B","")[0]]
        B = np.random.randn(*B_shape).astype(np.float32)
        i_des = ['x', 'W', 'B']
        i_data = [x, W, B]
    else:
        B = []
        i_des = ['x', 'W']
        i_data = [x, W]

    k_shape = W_shape[2:4]


    node = onnx.helper.make_node(
        'Conv',
        inputs=i_des,
        outputs=['y'],
        kernel_shape=k_shape,
        pads=p_shape,
        strides=s_shape,  # Default values for other attributes: dilations=[1, 1], groups=1
        dilations=d_shape,
        group = g,
        auto_pad = auto_pad
    )

    y = onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, [])


    out = run(node, inputs=i_data, outputs=[y], name='test')

    out = np.reshape(out, np.shape(out)[1:])

    if data["layout"] == "nhwc":
        x = np.transpose(x, [0, 2, 3, 1])
        if out_name == "depthwise_convolution":
            W = np.transpose(W, [1, 2, 3, 0])
        else:
            W = np.transpose(W, [0, 2, 3, 1])
        out = np.transpose(out, [0, 2, 3, 1])

    o_shape = np.shape(out)

    src_in_1   = x.flatten()
    weight_1   = W.flatten()
    src_out_1  = np.array(out).flatten()

    if out_name == "group_convolution":
        total_size = (len(src_in_1) + len(src_out_1)) + len(weight_1) + len(B) + 18
    else:
        total_size = (len(src_in_1) + len(src_out_1)) + len(weight_1) + len(B) + 17

    para.append(total_size)
    if data["layout"] == "nhwc":
        para.append(x_shape[0])
        para.append(x_shape[2]) #height
        para.append(x_shape[3]) #width
        para.append(x_shape[1])
    else:
        para.append(x_shape[0])
        para.append(x_shape[1])
        para.append(x_shape[2]) #height
        para.append(x_shape[3]) #width
    para.append(s_shape[0])
    para.append(s_shape[1])
    para.append(k_shape[0])
    para.append(k_shape[1])
    para.append(p_shape[1])
    para.append(p_shape[3])
    para.append(p_shape[0])
    para.append(p_shape[2])
    para.append(W_shape[0])
    para.append(d_shape[1])
    para.append(d_shape[0])
    if data["layout"] == "nhwc":
        if out_name == "depthwise_convolution":
            para.append(o_shape[1]) #height
            para.append(o_shape[2]) #width
        else:
            para.append(o_shape[2]) #width
            para.append(o_shape[1]) #height
    else:
        if out_name == "depthwise_convolution":
            para.append(o_shape[2]) #height
            para.append(o_shape[3]) #width
        else:
            para.append(o_shape[3]) #width
            para.append(o_shape[2]) #weight
    if out_name == "group_convolution":
       para.append(g) #height

    print(para)


    with open(f"{out_name}_test_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(weight_1)), *weight_1)
        fp.write(data)
        data = struct.pack(('%df' % len(B)), *B)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()



def convolution1d(test_data):
    data = test_data[1]
    out_name = test_data[0]
    para = []
    x_shape = data["X"]
    W_shape = data["W"]
    p_shape = data["pads"]
    s_shape = data["strides"][0]
    d_shape = data["dilations"][0]
    g = data["group"]

    x = np.random.randn(*x_shape).astype(np.float32)
    W = np.random.randn(*W_shape).astype(np.float32)

    if data.get("B",""):
        B_shape = [data.get("B","")[0]]
        B = np.random.randn(*B_shape).astype(np.float32)
        i_des = ['x', 'W', 'B']
        i_data = [x, W, B]
    else:
        B = []
        i_des = ['x', 'W']
        i_data = [x, W]

    
    t_src_in  = tensor(x)
    t_weight  = tensor(W)
    t_bias    = tensor(B)

    t_src_in  = fn.pad(t_src_in, p_shape, 'constant', 0)
    t_src_out1 = fn.conv1d(t_src_in, t_weight, bias=t_bias, stride=s_shape, dilation=d_shape, groups=g).numpy()
    out_size_x = np.shape(t_src_out1)[2]

    src_in_1   = x.flatten()
    weight_1   = W.flatten()
    src_out_1  = t_src_out1.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + len(weight_1) + len(B) + 17
    para.append(total_size)
    para.append(x_shape[0])
    para.append(x_shape[1])
    para.append(x_shape[2])  #width
    para.append(s_shape)
    para.append(W_shape[2])
    para.append(p_shape[0])
    para.append(p_shape[0])
    para.append(W_shape[0])
    para.append(d_shape)
    para.append(out_size_x)

    print(para)


    with open(f"{out_name}_test_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(weight_1)), *weight_1)
        fp.write(data)
        data = struct.pack(('%df' % len(B)), *B)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()



def layer_norm(test_data):
    data = test_data[1]
    out_name = test_data[0]
    para = []
    x_shape = data["X"]
    axis = data["axis"]
    gamma = data["gamma"]
    beta = data["beta"]

    x = np.random.randn(*x_shape).astype(np.float32)


    tf.enable_eager_execution()
    layer_norm = tf.keras.layers.LayerNormalization(
        axis=axis,
        gamma_initializer=tf.constant_initializer(gamma),
        beta_initializer=tf.constant_initializer(beta),
    )
    src_out = layer_norm(tf.convert_to_tensor(x)).numpy()

 
    src_in_1  = x.flatten()
    src_out_1 = src_out.flatten()


    norm_num = data["X"][1] * data["X"][2] * data["X"][3]
    gamma = np.full(norm_num, fill_value=gamma)
    beta = np.full(norm_num, fill_value=beta)

    total_size = len(src_in_1) + len(src_out_1) + len(gamma) + len(beta) + 5
    para.append(total_size)
    para.append(data["X"][0])
    para.append(data["X"][1])
    para.append(data["X"][2])
    para.append(data["X"][3])
    para.append(axis)
    print(para)

    with open(f"{out_name}_test_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        data = struct.pack(('%df' % len(gamma)), *gamma)
        fp.write(data)
        data = struct.pack(('%df' % len(beta)), *beta)
        fp.write(data)
        fp.close()


def convolution_relu(test_data):
    data = test_data[1]
    out_name = test_data[0]
    para = []
    x_shape = data["X"]
    W_shape = data["W"]
    p_shape = data["pads"]
    s_shape = data["strides"]
    d_shape = data["dilations"]
    g = data["group"]
    auto_pad = data["auto_pad"]

    x = np.random.randn(*x_shape).astype(np.float32)
    W = np.random.randn(*W_shape).astype(np.float32)

    if data.get("B",""):
        B_shape = [data.get("B","")[0]]
        B = np.random.randn(*B_shape).astype(np.float32)
        i_des = ['x', 'W', 'B']
        i_data = [x, W, B]
    else:
        B = []
        i_des = ['x', 'W']
        i_data = [x, W]

    k_shape = W_shape[2:4]

    y_tmp = onnx.helper.make_tensor_value_info('y_tmp', TensorProto.FLOAT, [])
    y = onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, [])


    node_conv = onnx.helper.make_node(
        'Conv',
        inputs=i_des,
        outputs=['y_tmp'],
        kernel_shape=k_shape,
        pads=p_shape,
        strides=s_shape,  # Default values for other attributes: dilations=[1, 1], groups=1
        dilations=d_shape,
        group = g,
        auto_pad = auto_pad
    )


    node_relu = onnx.helper.make_node(
        'Relu',
        inputs=['y_tmp'],
        outputs=['y'],
    )

    out = run(node_conv, inputs=i_data, outputs=[y_tmp], name='test')
    i_relu = [np.reshape(out, np.shape(out)[1:])]
    out = run(node_relu, inputs=i_relu, outputs=[y], name='test')

    out = np.reshape(out, np.shape(out)[1:])

    if data["layout"] == "nhwc":
        x = np.transpose(x, [0, 2, 3, 1])
        W = np.transpose(W, [0, 2, 3, 1])
        out = np.transpose(out, [0, 2, 3, 1])

    o_shape = np.shape(out)

    src_in_1   = x.flatten()
    weight_1   = W.flatten()
    src_out_1  = np.array(out).flatten()

    if out_name == "group_convolution":
        total_size = (len(src_in_1) + len(src_out_1)) + len(weight_1) + len(B) + 18
    else:
        total_size = (len(src_in_1) + len(src_out_1)) + len(weight_1) + len(B) + 17

    para.append(total_size)
    if data["layout"] == "nhwc":
        para.append(x_shape[0])
        para.append(x_shape[2]) #height
        para.append(x_shape[3]) #width
        para.append(x_shape[1])
    else:
        para.append(x_shape[0])
        para.append(x_shape[1])
        para.append(x_shape[2]) #height
        para.append(x_shape[3]) #width
    para.append(s_shape[0])
    para.append(s_shape[1])
    para.append(k_shape[0])
    para.append(k_shape[1])
    para.append(p_shape[1])
    para.append(p_shape[3])
    para.append(p_shape[0])
    para.append(p_shape[2])
    para.append(W_shape[0])
    para.append(d_shape[1])
    para.append(d_shape[0])
    if data["layout"] == "nhwc":
        para.append(o_shape[2]) #width
        para.append(o_shape[1]) #height
    else:
        para.append(o_shape[3]) #width
        para.append(o_shape[2]) #height
    if out_name == "group_convolution":
       para.append(g)

    print(para)


    with open(f"{out_name}_test_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(weight_1)), *weight_1)
        fp.write(data)
        data = struct.pack(('%df' % len(B)), *B)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()



def pad(test_data):
    data = test_data[1]
    out_name = test_data[0]
    para = []
    x_shape = data["X"]
    p_shape = np.array(data["pads"]).astype(np.int64)

    x = np.random.randn(*x_shape).astype(np.float32)

    i_des = ['x', 'pads']
    i_data = [x, p_shape]


    node = onnx.helper.make_node(
        'Pad',
        inputs=i_des,
        outputs=['y'],
        mode='constant'
    )

    y = onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, [])

    out = run(node, inputs=i_data, outputs=[y], name='test')

    src_in_1   = x.flatten()
    src_out_1  = np.array(out).flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 8

    para.append(total_size)
    para.append(x_shape[0])
    para.append(x_shape[1])
    para.append(x_shape[2]) #height
    para.append(x_shape[3]) #width
    para.append(p_shape[3])
    para.append(p_shape[7])
    para.append(p_shape[2])
    para.append(p_shape[6])
    print(para)


    with open(f"{out_name}_test_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()


def maxpool(test_data):
    data = test_data[1]
    out_name = test_data[0]
    para = []
    x_shape = data["X"]
    k_shape = data["kernel_shape"]
    p_shape = data["pads"]
    s_shape = data["strides"]
    d_shape = data["dilations"]
    storage_order = data["storage_order"]
    ceil_mode = data["ceil_mode"]
    auto_pad = data["auto_pad"]

    x = np.random.randn(*x_shape).astype(np.float32)

    i_des = ['x']
    i_data = [x]

    node = onnx.helper.make_node(
        'MaxPool',
        inputs=i_des,
        outputs=['y'],
        kernel_shape=k_shape,
        pads=p_shape,
        strides=s_shape,
        dilations=d_shape,
        storage_order=storage_order,
        auto_pad=auto_pad,
        ceil_mode=ceil_mode
    )

    y = onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, [])

    out = run(node, inputs=i_data, outputs=[y], name='test')
    out = np.reshape(out, np.shape(out)[1:])
    o_shape = np.shape(out)

    if data["layout"] == "nhwc":
        #permute nchw to nhwc
        x = np.transpose(x, [0, 2, 3, 1])
        out = np.transpose(out, [0, 2, 3, 1])

    src_in_1 = x.flatten()
    src_out_1  = np.array(out).flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 15

    para.append(total_size)
    para.append(x_shape[0])
    if data["layout"] == "nhwc":
        para.append(x_shape[2]) #height
        para.append(x_shape[3]) #width
        para.append(x_shape[1])
    else:
        para.append(x_shape[1])
        para.append(x_shape[2]) #height
        para.append(x_shape[3]) #width

    para.append(s_shape[0]) #height
    para.append(s_shape[1]) #width

    para.append(k_shape[0]) #height
    para.append(k_shape[1]) #width

    para.append(p_shape[1])
    para.append(p_shape[3])
    para.append(p_shape[0])
    para.append(p_shape[2])

    para.append(o_shape[2]) #height
    para.append(o_shape[3]) #width
    para.append(ceil_mode)


    print(para)

    with open(f"{out_name}_test_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()


def averagepool(test_data):
    data = test_data[1]
    out_name = test_data[0]
    para = []
    x_shape = data["X"]
    k_shape = data["kernel_shape"]
    p_shape = data["pads"]
    s_shape = data["strides"]
    ceil_mode = data["ceil_mode"]
    count_include_pad = data["count_include_pad"]
    auto_pad = data["auto_pad"]

    x = np.random.randn(*x_shape).astype(np.float32)

    i_des = ['x']
    i_data = [x]

    node = onnx.helper.make_node(
        'AveragePool',
        inputs=i_des,
        outputs=['y'],
        kernel_shape=k_shape,
        pads=p_shape,
        strides=s_shape,
        auto_pad=auto_pad,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad
    )

    y = onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, [])

    out = run(node, inputs=i_data, outputs=[y], name='test')
    out = np.reshape(out, np.shape(out)[1:])
    o_shape = np.shape(out)

    if data["layout"] == "nhwc":
        #permute nchw to nhwc
        x = np.transpose(x, [0, 2, 3, 1])
        out = np.transpose(out, [0, 2, 3, 1])

    src_in_1   = x.flatten()
    src_out_1  = np.array(out).flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 16

    para.append(total_size)
    para.append(x_shape[0])
    if data["layout"] == "nhwc":
        para.append(x_shape[2]) #height
        para.append(x_shape[3]) #width
        para.append(x_shape[1])
    else:
        para.append(x_shape[1])
        para.append(x_shape[2]) #height
        para.append(x_shape[3]) #width

    para.append(s_shape[0]) #height
    para.append(s_shape[1]) #width

    para.append(k_shape[0]) #height
    para.append(k_shape[1]) #width

    para.append(p_shape[1])
    para.append(p_shape[3])
    para.append(p_shape[0])
    para.append(p_shape[2])

    para.append(o_shape[2]) #height
    para.append(o_shape[3]) #width
    para.append(count_include_pad)
    para.append(ceil_mode)


    print(para)

    with open(f"{out_name}_test_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()



def binary_b(test_data):
    data = test_data[1]
    out_name = test_data[0]
    para = []
    a_shape = data["A"]
    b_shape = data["B"]

    in0_dim = len(a_shape)
    in1_dim = len(b_shape)

    if out_name=="div":
        a = np.random.uniform(1, 2, a_shape).astype(np.float32)
        b = np.random.uniform(1, 2, b_shape).astype(np.float32)
    else:
        a = np.random.randn(*a_shape).astype(np.float32)
        b = np.random.randn(*b_shape).astype(np.float32)

    i_des = ['A', 'B']
    i_data = [a, b]


    node = onnx.helper.make_node(
        f"{out_name.capitalize()}",
        inputs=i_des,
        outputs=['y'],
    )

    y = onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, [])

    out = run(node, inputs=i_data, outputs=[y], name='test')

    out = np.reshape(out, np.shape(out)[1:])
    o_shape = np.shape(out)
    out_dim = len(o_shape)

    src_in_1 = a.flatten()
    src_in_2 = b.flatten()
    src_out_1  = np.array(out).flatten()

    total_size = (len(src_in_1) + len(src_in_2) + len(src_out_1)) + 3 + in0_dim + in1_dim + out_dim

    para.append(total_size)         # 0
    para.append(in0_dim)            # 1
    para.append(in1_dim)            # 2
    para.append(out_dim)            # 3
    for i in range(0, in0_dim):
        para.append(a_shape[i])   # 3 ~ 3+in0_dim-1
    for i in range(0, in1_dim):
        para.append(b_shape[i])   # 3+in0_dim ~ 3+in0_dim+in1_dim-1
    for i in range(0, out_dim):
        para.append(o_shape[i])   # 3+in0_dim+in1_dim ~ 3+in0_dim+in1_dim+out_dim-1
    print(para)


    with open(f"{out_name}_test_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_2)), *src_in_2)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()


def muti_min(test_data):
    data = test_data[1]
    out_name = test_data[0]
    para = []
    data_shape = data["data_0"]
    if len(data_shape) > 2:
        print("Unsupport shape!!")
        return
    a_shape = data_shape[0]
    b_shape = data_shape[1]

    in0_dim = len(a_shape)
    in1_dim = len(b_shape)


    a = np.random.randn(*a_shape).astype(np.float32)
    b = np.random.randn(*b_shape).astype(np.float32)

    i_des = ['A', 'B']
    i_data = [a, b]


    node = onnx.helper.make_node(
        'Min',
        inputs=i_des,
        outputs=['y'],
    )

    y = onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, [])

    out = run(node, inputs=i_data, outputs=[y], name='test')

    out = np.reshape(out, np.shape(out)[1:])
    o_shape = np.shape(out)
    out_dim = len(o_shape)

    src_in_1 = a.flatten()
    src_in_2 = b.flatten()
    src_out_1  = np.array(out).flatten()

    total_size = (len(src_in_1) + len(src_in_2) + len(src_out_1)) + 3 + in0_dim + in1_dim + out_dim

    para.append(total_size)         # 0
    para.append(in0_dim)            # 1
    para.append(in1_dim)            # 2
    para.append(out_dim)            # 3
    for i in range(0, in0_dim):
        para.append(a_shape[i])   # 3 ~ 3+in0_dim-1
    for i in range(0, in1_dim):
        para.append(b_shape[i])   # 3+in0_dim ~ 3+in0_dim+in1_dim-1
    for i in range(0, out_dim):
        para.append(o_shape[i])   # 3+in0_dim+in1_dim ~ 3+in0_dim+in1_dim+out_dim-1
    print(para)


    with open(f"{out_name}_test_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_2)), *src_in_2)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()


def reshape(test_data):
    data = test_data[1]
    out_name = test_data[0]
    para = []
    data_shape = data["data"]
    reshape = data["shape"]

    in0_dim = len(data_shape)
    in1_dim = len(reshape)


    a = np.random.randn(*data_shape).astype(np.float32)
    b = np.array(reshape, dtype=np.int64)

    i_des = ['data', 'shape']
    i_data = [a, b]

    node = onnx.helper.make_node(
        'Reshape',
        inputs=i_des,
        outputs=['y'],
    )

    y = onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, [])

    out = run(node, inputs=i_data, outputs=[y], name='test')

    out = np.reshape(out, np.shape(out)[1:])
    # o_shape = np.shape(out)
    # out_dim = len(o_shape)

    src_in_1 = a.flatten()
    src_out_1  = np.array(out).flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 3 + in0_dim + in1_dim

    para.append(total_size)         # 0
    para.append(in0_dim)            # 1
    para.append(in1_dim)            # 2
    for i in range(0, in0_dim):
        para.append(data_shape[i])   # 2 ~ 2+in0_dim-1
    for i in range(0, in1_dim):
        para.append(reshape[i])   # 2+in0_dim ~ 2+in0_dim+in1_dim-1
    print(para)


    with open(f"{out_name}_test_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()


def strided_slice(test_data):
    data = test_data[1]
    out_name = test_data[0]
    para = []
    in_shape = data["data"]
    starts = data["starts"]
    ends = data["ends"]
    axes = data["axes"]
    steps = data["steps"]

    in_dim = len(in_shape)
    slice_count = len(starts)

    a = np.random.randn(*in_shape).astype(np.float32)
    starts = np.array(starts, dtype=np.int64)
    ends = np.array(ends, dtype=np.int64)
    axes = np.array(axes, dtype=np.int64)
    steps = np.array(steps, dtype=np.int64)

    i_des = ['x', 'starts', 'ends', 'axes', 'steps']
    i_data = [a, starts, ends, axes, steps]

    node = onnx.helper.make_node(
        'Slice',
        inputs=i_des,
        outputs=['y'],
    )

    y = onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, [])

    out = run(node, inputs=i_data, outputs=[y], name='test')

    out = np.reshape(out, np.shape(out)[1:])
    # o_shape = np.shape(out)
    # out_dim = len(o_shape)

    src_in_1 = a.flatten()
    src_out_1  = np.array(out).flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 3 + in_dim + 3 * in_dim

    para.append(total_size)
    para.append(in_dim)
    for i in range(0, in_dim):
        para.append(in_shape[i])
    para.append(slice_count)
    for i in range(0, in_dim):
        para.append(starts[i])
        para.append(ends[i])
        para.append(steps[i])
    para.append(len(src_out_1))
    print(para)


    with open(f"{out_name}_test_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()


def deconvolution(test_data):
    data = test_data[1]
    out_name = test_data[0]
    para = []
    x_shape = data["X"]
    W_shape = data["W"]
    kernel_shape = data["kernel_shape"]
    p_shape = data["pads"]
    opadding_shape = data["output_padding"]
    output_shape = data.get("output_shape","")
    s_shape = data["strides"]
    d_shape = data["dilations"]
    g = data["group"]
    auto_pad = data["auto_pad"]

    x = np.random.randn(*x_shape).astype(np.float32)
    W = np.random.randn(*W_shape).astype(np.float32)

    if data.get("B",""):
        B_shape = [data.get("B","")[0]]
        B = np.random.randn(*B_shape).astype(np.float32)
        i_des = ['x', 'W', 'B']
        i_data = [x, W, B]
    else:
        B = []
        i_des = ['x', 'W']
        i_data = [x, W]

    k_shape = kernel_shape


    node = onnx.helper.make_node(
        'ConvTranspose',
        inputs=i_des,
        outputs=['y'],
        kernel_shape=k_shape,
        output_padding=opadding_shape,
        pads=p_shape,
        strides=s_shape,  # Default values for other attributes: dilations=[1, 1], groups=1
        dilations=d_shape,
        group = g,
        auto_pad = auto_pad
    )

    y = onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, [])


    out = run(node, inputs=i_data, outputs=[y], name='test')

    out = np.reshape(out, np.shape(out)[1:])


    o_shape = np.shape(out)

    src_in_1   = x.flatten()
    weight_1   = W.flatten()
    src_out_1  = np.array(out).flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + len(weight_1) + len(B) + 17

    para.append(total_size)
    para.append(x_shape[0])
    para.append(x_shape[1])
    para.append(x_shape[2]) #height
    para.append(x_shape[3]) #width
    para.append(s_shape[0])
    para.append(s_shape[1])
    para.append(k_shape[0])
    para.append(k_shape[1])
    para.append(p_shape[1])
    para.append(p_shape[3])
    para.append(p_shape[0])
    para.append(p_shape[2])
    para.append(W_shape[1]) ## output_channel  #12
    para.append(d_shape[1]) ## 13
    para.append(d_shape[0]) ## 14
    if out_name == "depthwise_deconvolution":
        if output_shape:
            para.append(output_shape[0]) #height
            para.append(output_shape[1]) #width
        else:
            para.append(o_shape[2]) #height
            para.append(o_shape[3]) #width
    else:
        if output_shape:
            para.append(output_shape[1]) #height
            para.append(output_shape[0]) #width
        else:
            para.append(o_shape[3]) #width
            para.append(o_shape[2]) #weight

    print(para)

    with open(f"{out_name}_test_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(weight_1)), *weight_1)
        fp.write(data)
        data = struct.pack(('%df' % len(B)), *B)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()



def global_avgpool(test_data):
    data = test_data[1]
    out_name = test_data[0]
    para = []
    x_shape = data["X"]

    x = np.random.randn(*x_shape).astype(np.float32)

    i_des = ['x']
    i_data = [x]

    node = onnx.helper.make_node(
        'GlobalAveragePool',
        inputs=i_des,
        outputs=['y'],
    )

    y = onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, [])

    out = run(node, inputs=i_data, outputs=[y], name='test')
    out = np.reshape(out, np.shape(out)[1:])
    o_shape = np.shape(out)

    if data["layout"] == "nhwc":
        #permute nchw to nhwc
        x = np.transpose(x, [0, 2, 3, 1])
        out = np.transpose(out, [0, 2, 3, 1])


    src_in_1   = x.flatten()
    src_out_1  = np.array(out).flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 6

    para.append(total_size)
    para.append(x_shape[0])
    if data["layout"] == "nhwc":
        para.append(x_shape[2]) #height
        para.append(x_shape[3]) #width
        para.append(x_shape[1])
    else:
        para.append(x_shape[1])
        para.append(x_shape[2]) #height
        para.append(x_shape[3]) #width

    para.append(o_shape[2]) #height
    para.append(o_shape[3]) #width

    print(para)

    with open(f"{out_name}_test_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()


def global_maxpool(test_data):
    data = test_data[1]
    out_name = test_data[0]
    para = []
    x_shape = data["X"]

    x = np.random.randn(*x_shape).astype(np.float32)

    i_des = ['x']
    i_data = [x]

    node = onnx.helper.make_node(
        'GlobalMaxPool',
        inputs=i_des,
        outputs=['y'],
    )

    y = onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, [])

    out = run(node, inputs=i_data, outputs=[y], name='test')
    out = np.reshape(out, np.shape(out)[1:])
    o_shape = np.shape(out)

    if data["layout"] == "nhwc":
        #permute nchw to nhwc
        x = np.transpose(x, [0, 2, 3, 1])
        out = np.transpose(out, [0, 2, 3, 1])

    src_in_1   = x.flatten()
    src_out_1  = np.array(out).flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 6

    para.append(total_size)
    para.append(x_shape[0])
    if data["layout"] == "nhwc":
        para.append(x_shape[2]) #height
        para.append(x_shape[3]) #width
        para.append(x_shape[1])
    else:
        para.append(x_shape[1])
        para.append(x_shape[2]) #height
        para.append(x_shape[3]) #width

    para.append(o_shape[2]) #height
    para.append(o_shape[3]) #width

    print(para)

    with open(f"{out_name}_test_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()


def fullyconnected(test_data):
    data = test_data[1]
    out_name = test_data[0]
    para = []
    a_shape = data["A"]
    b_shape = data["B"]
    c_shape = data["C"]

    a = np.random.randn(*a_shape).astype(np.float32)
    b = np.random.randn(*b_shape).astype(np.float32)
    c = np.random.randn(*c_shape).astype(np.float32)

    i_des = ['a', 'b', 'c']
    i_data = [a, b, c]

    node = onnx.helper.make_node(
        'Gemm',
        inputs=i_des,
        outputs=['y'],
    )

    y = onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, [])

    out = run(node, inputs=i_data, outputs=[y], name='test')
    out = np.reshape(out, np.shape(out)[1:])

    src_in_1   = a.flatten()
    weight = np.transpose(b, [1, 0])
    weight = weight.flatten()
    bias = c.flatten()

    o_shape = np.shape(out)
    src_out_1  = np.array(out).flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + len(bias) + + len(weight) + 3

    para.append(total_size)
    para.append(a_shape[0]) ## batch
    para.append(a_shape[1]) ## in_size
    para.append(b_shape[1]) ## out_size

    print(para)

    with open(f"{out_name}_test_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(weight)), *weight)
        fp.write(data)
        data = struct.pack(('%df' % len(bias)), *bias)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()


def unary(test_data):
    data = test_data[1]
    out_name = test_data[0]
    para = []
    x_shape = data["X"]

    x = np.random.randn(*x_shape).astype(np.float32)
    in_dim = len(x_shape)

    i_des = ['x']
    i_data = [x]

    node = onnx.helper.make_node(
        f"{out_name.capitalize()}",
        inputs=i_des,
        outputs=['y'],
    )

    y = onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, [])

    out = run(node, inputs=i_data, outputs=[y], name='test')
    out = np.reshape(out, np.shape(out)[1:])
    o_shape = np.shape(out)


    src_in_1   = x.flatten()
    src_out_1  = np.array(out).flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 1 + in_dim

    para.append(total_size)
    para.append(in_dim)
    for i in range(0, in_dim):
        para.append(x_shape[i])

    print(para)
    with open(f"{out_name}_test_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()


def reduce_op(test_data):
    data = test_data[1]
    out_name = test_data[0]
    para = []
    x_shape = data["X"]
    axes_data = data["axes"]
    keepdims = data["keepdims"]
    noop_with_empty_axes = data["noop_with_empty_axes"]

    x = np.random.randn(*x_shape).astype(np.float32)
    in_dim = len(x_shape)
    axes = np.array([axes_data], dtype=np.int64)

    i_des = ['x', 'axes']
    i_data = [x, axes]

    op_name = ""

    for i in  out_name.split("_"):
        op_name += i.capitalize()

    node = onnx.helper.make_node(
        f"{op_name}",
        inputs=i_des,
        outputs=['y'],
        keepdims=keepdims,
        noop_with_empty_axes=noop_with_empty_axes
    )

    y = onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, [])

    out = run(node, inputs=i_data, outputs=[y], name='test')
    out = np.reshape(out, np.shape(out)[1:])
    o_shape = np.shape(out)

    src_in_1   = x.flatten()
    src_out_1  = np.array(out).flatten()
    if axes_data < 0:
        axes_data = in_dim - abs(axes_data)

    total_size = (len(src_in_1) + len(src_out_1)) + 3 + in_dim

    para.append(total_size)
    para.append(in_dim)
    para.append(axes_data)
    para.append(keepdims)
    for i in range(0, in_dim):
        para.append(x_shape[i])

    print(para)
    with open(f"{out_name}_test_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()


def silu(test_data):
    data = test_data[1]
    out_name = test_data[0]
    para = []
    x_shape = data["X"]

    x = np.random.randn(*x_shape).astype(np.float32)
    in_dim = len(x_shape)

    t_src_in = tensor(x)
    out = fn.silu(t_src_in).numpy()
    o_shape = np.shape(out)


    src_in_1   = x.flatten()
    src_out_1  = np.array(out).flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 1 + in_dim

    para.append(total_size)
    para.append(in_dim)
    for i in range(0, in_dim):
        para.append(x_shape[i])

    print(para)
    with open(f"{out_name}_test_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()


def softmax(test_data):
    data = test_data[1]
    out_name = test_data[0]
    para = []
    x_shape = data["X"]
    axis = data["axis"]

    x = np.random.randn(*x_shape).astype(np.float32)
    in_dim = len(x_shape)

    i_des = ['x']
    i_data = [x]

    node = onnx.helper.make_node(
        f"{out_name.capitalize()}",
        inputs=i_des,
        outputs=['y'],
        axis=axis,
    )

    y = onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, [])

    out = run(node, inputs=i_data, outputs=[y], name='test')
    out = np.reshape(out, np.shape(out)[1:])
    o_shape = np.shape(out)


    src_in_1   = x.flatten()
    src_out_1  = np.array(out).flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 2 + in_dim

    para.append(total_size)
    para.append(in_dim)
    para.append(axis)
    for i in range(0, in_dim):
        para.append(x_shape[i])

    print(para)
    with open(f"{out_name}_test_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()


def clip(test_data):
    data = test_data[1]
    out_name = test_data[0]
    para = []
    x_shape = data["input"]
    x = np.random.randn(*x_shape).astype(np.float32)

    clip_min_val = np.array(data["min"]).astype(np.float32)
    clip_max_val = np.array(data["max"]).astype(np.float32)

    i_des = ['x', 'min', 'max']
    i_data = [x, clip_min_val, clip_max_val]

    node = onnx.helper.make_node(
        'Clip',
        inputs=i_des,
        outputs=['y'],
    )

    y = onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, [])

    out = run(node, inputs=i_data, outputs=[y], name='test')
    out = np.reshape(out, np.shape(out)[1:])
    o_shape = np.shape(out)


    src_in_1   = x.flatten()
    src_out_1  = np.array(out).flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 6

    para.append(total_size)
    para.append(x_shape[0])
    para.append(x_shape[1])
    para.append(x_shape[2])
    para.append(x_shape[3])
    para.append(int(clip_min_val))
    para.append(int(clip_max_val))
    print(para)

    with open(f"{out_name}_test_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()


def concat(test_data):
    data = test_data[1]
    out_name = test_data[0]
    para = []
    src_in_1 = []
    x_shape = data["inputs"]
    x = [np.random.randn(*v).astype(np.float32) for v in x_shape]
    in_args = ['value' + str(k) for k in range(len(x_shape))]

    axis = int(data["axis"])

    i_data = [v for v in x]

    node = onnx.helper.make_node(
        'Concat',
        inputs=[s for s in in_args],
        outputs=['y'],
        axis=axis,
    )

    y = onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, [])

    out = run(node, inputs=i_data, outputs=[y], name='test')
    out = np.reshape(out, np.shape(out)[1:])
    o_shape = np.shape(out)

    for i in range(0, len(x_shape)):
        src_in_1.append(x[i].flatten())
    src_out_1  = np.array(out).flatten()

    total_size = (len(src_in_1[0]) * len(x_shape) + len(src_out_1)) + 6

    para.append(total_size)
    para.append(x_shape[0][0])
    para.append(x_shape[0][1])
    para.append(x_shape[0][2])
    para.append(x_shape[0][3])
    para.append(int(len(x_shape)))
    para.append(int(axis))
    print(para)

    with open(f"{out_name}_test_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        for i in range(0, len(x_shape)):
            data = struct.pack(('%df' % len(src_in_1[i])), *src_in_1[i])
            fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()


def leaky_relu(test_data):
    data = test_data[1]
    out_name = test_data[0]
    para = []
    x_shape = data["X"]
    alpha = np.float32([data["alpha"]])

    x = np.random.randn(*x_shape).astype(np.float32)
    out = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * alpha

    src_in_1   = x.flatten()
    src_out_1  = np.array(out).flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 5

    para.append(total_size)
    para.append(x_shape[0])
    para.append(x_shape[1])
    para.append(x_shape[2])
    para.append(x_shape[3])
    print(para)

    with open(f"{out_name}_test_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(alpha)), *alpha)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()


def thresholdedrelu(test_data):
    data = test_data[1]
    out_name = test_data[0]
    para = []
    x_shape = data["X"]
    if out_name == "relu1":
        alpha = 1.0
    elif out_name == "relu6":
        alpha = 6.0

    x = np.random.randn(*x_shape).astype(np.float32)
    in_dim = len(x_shape)


    out = torch.clamp(tensor(x), 0, alpha).numpy()
    o_shape = np.shape(out)

    src_in_1   = x.flatten()
    src_out_1  = np.array(out).flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 1 + in_dim

    para.append(total_size)
    para.append(in_dim)
    for i in range(0, in_dim):
        para.append(x_shape[i])

    print(para)
    with open(f"{out_name}_test_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()


def gather(test_data):
    data = test_data[1]
    out_name = test_data[0]
    para = []
    x_shape = data["data"]
    indices = data["indices"]
    axis = int(data["axis"])
    in_dim = len(x_shape)
    indices_shape = np.shape(indices)
    indices_dim = len(indices_shape)

    x = np.random.randn(*x_shape).astype(np.float32)
    indices = np.array(indices).astype(np.int64)

    i_des = ['data', 'indices']
    i_data = [x, indices]

    node = onnx.helper.make_node(
        'Gather',
        inputs=i_des,
        outputs=['y'],
        axis=axis
    )

    y = onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, [])

    out = run(node, inputs=i_data, outputs=[y], name='test')
    out = np.reshape(out, np.shape(out)[1:])
    o_shape = np.shape(out)


    src_in_1   = x.flatten()
    indices_in_1 = indices.flatten()
    src_out_1  = np.array(out).flatten()

    total_size = (len(src_in_1) + len(src_out_1) + len(indices_in_1)) + 3 + in_dim + indices_dim

    para.append(total_size)
    para.append(axis)
    para.append(in_dim)
    for i in range(0, in_dim):
        para.append(x_shape[i])

    para.append(indices_dim)
    for i in range(0, indices_dim):
        para.append(indices_shape[i])
    print(para)

    with open(f"{out_name}_test_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%di' % len(indices_in_1)), *indices_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

def split(test_data):
    data = test_data[1]
    out_name = test_data[0]
    para = []
    x_shape = data["input"]
    split = data["split"]
    axis = int(data["axis"])
    num_split = len(split)


    x = np.random.randn(*x_shape).astype(np.float32)
    split = np.array(split).astype(np.int64)

    i_des = ['input', 'split']
    i_data = [x, split]
    o_des = ['y_' + str(k) for k in range(num_split)]

    node = onnx.helper.make_node(
        'Split',
        inputs=i_des,
        outputs=o_des,
        axis=axis,
    )

    y = [onnx.helper.make_tensor_value_info(f"y_{y}", TensorProto.FLOAT, []) for y in range(num_split)]

    out = run(node, inputs=i_data, outputs=y, name='test')

    src_in_1   = x.flatten()
    src_out_1 = []
    for i in range(0, num_split):
        src_out_1.append(out[i].flatten())

    total_size = (len(src_in_1) + len(src_out_1[0]) * num_split) + 6

    para.append(total_size)
    para.append(x_shape[0])
    para.append(x_shape[1])
    para.append(x_shape[2])
    para.append(x_shape[3])
    para.append(axis)
    para.append(num_split)
    print(para)

    with open(f"{out_name}_test_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        for i in range(0, num_split):
            data = struct.pack(('%df' % len(src_out_1[i])), *src_out_1[i])
            fp.write(data)
        fp.close()



def transpose(test_data):
    data = test_data[1]
    out_name = test_data[0]
    para = []
    x_shape = data["data"]
    perm = data["perm"]

    input_dim_count = len(x_shape)

    x = np.random.randn(*x_shape).astype(np.float32)

    i_des = ['x']
    i_data = [x]

    node = onnx.helper.make_node(
        'Transpose',
        inputs=i_des,
        outputs=['y'],
        perm=perm
    )

    y = onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, [])

    out = run(node, inputs=i_data, outputs=[y], name='test')
    src_out = np.reshape(out, np.shape(out)[1:])
    o_shape = np.shape(src_out)

    src_in_1 = x.flatten()
    src_out_1 = src_out.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 1 + input_dim_count*3

    para.append(total_size)
    para.append(input_dim_count)
    for i in range(0, input_dim_count):
        para.append(x_shape[i])
    for i in range(0, input_dim_count):
        para.append(perm[i])
    for i in range(0, input_dim_count):
        para.append(o_shape[i])
    print(para)

    with open(f"{out_name}_test_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()


def matmul(test_data):
    data = test_data[1]
    out_name = test_data[0]
    para = []
    a_shape = data["A"]
    b_shape = data["B"]
    transpose_a = data["transpose_a"]
    transpose_b = data["transpose_b"]

    tran = []

    src_in0 = np.random.randn(*a_shape).astype(np.float32)
    src_in1 = np.random.randn(*b_shape).astype(np.float32)
    if len(a_shape) >= 2:
        for i in range(0, len(a_shape)):
            if i == len(a_shape) - 1:
                tran.append(i-1)
            elif i == len(a_shape) - 2:
                tran.append(i+1)
            else:
                tran.append(i)

        if transpose_a == 1:
            a = src_in0.transpose(tran)
        else:
            a = src_in0
        if transpose_b == 1:
            b = src_in1.transpose(tran)
        else:
            b = src_in1


    i_data = [a, b]

    node = onnx.helper.make_node(
        'MatMul',
        inputs=['a', 'b'],
        outputs=['y'],
    )

    y = onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, [])

    out = run(node, inputs=i_data, outputs=[y], name='test')
    src_out = np.reshape(out, np.shape(out)[1:])
    o_shape = np.shape(out)

    src_in_0  = src_in0.flatten()
    src_in_1  = src_in1.flatten()

    src_out_1 = src_out.flatten()

    total_size = len(src_in_0) + len(src_in_1) + len(src_out_1) + 3 * len(a_shape) + 3

    para.append(total_size)
    para.append(transpose_a)
    para.append(transpose_b)
    para.append(len(a_shape))
    print(para)

    with open(f"{out_name}_test_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%di' % len(a_shape)), *a_shape)
        fp.write(data)
        data = struct.pack(('%di' % len(b_shape)), *b_shape)
        fp.write(data)
        data = struct.pack(('%di' % len(np.shape(src_out))), *np.shape(src_out))
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_0)), *src_in_0)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()


def prelu(test_data):
    data = test_data[1]
    out_name = test_data[0]
    para = []
    x_shape = data["X"]
    slope_shape = data["slope"]

    x = np.random.randn(*x_shape).astype(np.float32)
    slope = np.random.randn(*slope_shape).astype(np.float32)

    t_src_in  = tensor(x)
    t_weight  = tensor(slope)
    src_out = fn.prelu(t_src_in, t_weight).numpy()

    src_in_1  = x.flatten()
    slope  = slope.flatten()

    src_out_1 = src_out.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + len(slope) + 4

    para.append(total_size)
    para.append(x_shape[0])
    para.append(x_shape[1])
    para.append(x_shape[2])
    para.append(x_shape[3])
    print(para)

    with open(f"{out_name}_test_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(slope)), *slope)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()



def rms_norm(test_data):
    def rms_norm(x: np.array, weight: np.array, axis, eps=1e-8) -> np.array:
        axis = axis if axis >= 0 else axis + len(x.shape)
        batches = np.prod(x.shape[:axis])
        norm_size = np.prod(x.shape[axis:])

        x = x.flatten()
        weight = weight.flatten()
        output = np.empty(x.shape, dtype=np.float32).flatten()
        for b in range(batches):
            avg = np.sum([x[b * norm_size + i]**2 for i in range(norm_size)]) / norm_size
            scale = 1.0 / math.sqrt(avg + eps)
            for i in range(norm_size):
                output[b * norm_size + i] = x[b * norm_size + i] * scale * weight[i]

        return output

    data = test_data[1]
    out_name = test_data[0]
    para = []
    x_shape = data["X"]
    axis = -1
    eps = 1e-8
    norm_shape = np.array(x_shape[axis:])

    x = np.random.randn(*x_shape).astype(np.float32)
    weight = np.random.randn(*norm_shape).astype(np.float32)

    src_out = rms_norm(x, weight, axis, eps)

    src_in_1  = x.flatten()
    weight_1  = weight.flatten()
    src_out_1 = src_out.flatten()

    total_size = len(src_in_1) + len(weight_1) + len(src_out_1) + 6

    para.append(total_size)
    para.append(x_shape[0])
    para.append(x_shape[1])
    para.append(x_shape[2])
    para.append(x_shape[3])
    para.append(axis)
    para_f = [eps]
    print(para + para_f)

    with open(f"{out_name}_test_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(para_f)), *para_f)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(weight_1)), *weight_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()




def lrn(test_data):
    data = test_data[1]
    out_name = test_data[0]
    para_int = []
    para_float = []
    x_shape = data["X"]
    alpha = data["alpha"]
    beta = data["beta"]
    bias = data["bias"]
    size = data["size"]


    x = np.random.randn(*x_shape).astype(np.float32)

    i_data = [x]

    node = onnx.helper.make_node(
        'LRN',
        inputs=['x'],
        outputs=['y'],
        alpha=alpha,
        beta=beta,
        bias=bias,
        size=size
    )

    y = onnx.helper.make_tensor_value_info('y', TensorProto.FLOAT, [])

    out = run(node, inputs=i_data, outputs=[y], name='test')
    out = np.reshape(out, np.shape(out)[1:])
    o_shape = np.shape(out)


    src_in_1   = x.flatten()
    src_out_1  = np.array(out).flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 8

    para_int.append(total_size)
    para_int.append(x_shape[0])
    para_int.append(x_shape[1])
    para_int.append(x_shape[2])
    para_int.append(x_shape[3])
    para_int.append(size)
    para_float.append(bias)
    para_float.append(alpha)
    para_float.append(beta)
    print(para_int)
    print(para_float)

    with open(f"{out_name}_test_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para_int)), *para_int)
        fp.write(data)
        data = struct.pack(('%df' % len(para_float)), *para_float)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

