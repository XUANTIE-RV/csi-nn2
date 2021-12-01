
import numpy as np
import tensorflow as tf
import struct

def get_stride(pos, shape):
    size = 1
    for i in range(pos+1,len(shape)):
        size *= shape[i]
    return size

def get_reduce(input, axis):
    inner_extend = []
    inner_stride = []
    out_extend = []
    out_stride = []

    for i in range(len(np.shape(input))):
        flag = 0
        for j in range(len(axis)):
            tmp = axis[j]
            if (i == tmp):
                flag =1
        if flag:
            inner_extend.append(np.shape(input)[i])
            stride = get_stride(i, np.shape(input))
            inner_stride.append(stride)
        else:
            out_extend.append(np.shape(input)[i])
            stride = get_stride(i, np.shape(input))
            out_stride.append(stride)
    return out_stride, out_extend, inner_stride, inner_extend,




def prod_f32():
    para = []
    # init the input data and parameters
    batch       = int(np.random.randint(1, high=4, size=1))
    in_size_x   = int(np.random.randint(16, high=32, size=1))
    in_size_y   = int(np.random.randint(16, high=32, size=1))
    in_channel  = int(np.random.randint(1, high=16, size=1))

    zero_point1 = int(np.random.randint(-3, high=3, size=1))
    std1        = int(np.random.randint(1, high=10, size=1))

    src_in1 = np.random.normal(zero_point1, std1, (batch, in_channel, in_size_y, in_size_x, ))
    src_in1 = src_in1.astype(np.float32)
    axis = int(np.random.randint(0, high=4, size=1))


    prod_a_nokeepdims = tf.keras.backend.prod(src_in1, axis=axis, keepdims=False)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        result = sess.run(prod_a_nokeepdims)

    a, b, c, d = get_reduce(src_in1, [axis, ])

    inner_stride = np.array(c).astype(np.int32)
    inner_extend = np.array(d).astype(np.int32)
    out_extend = np.array(b).astype(np.int32)
    out_stride = np.array(a).astype(np.int32)


    m = len(inner_extend)
    n = len(out_extend)

    src_in_1 = src_in1.flatten()
    src_out_1 = result.flatten()


    total_size = (len(src_in_1) + len(src_out_1) + len(inner_extend) + len(out_extend) + len(inner_stride) + len(out_stride)) + 7

    print(total_size)

    para.append(total_size)
    para.append(batch) #batch
    para.append(in_channel) #channel
    para.append(in_size_y) #height
    para.append(in_size_x) #width
    para.append(axis)
    para.append(m)
    para.append(n)


    with open("prod_stride_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%di' % len(out_stride)), *out_stride)
        fp.write(data)
        data = struct.pack(('%di' % len(out_extend)), *out_extend)
        fp.write(data)
        data = struct.pack(('%di' % len(inner_stride)), *inner_stride)
        fp.write(data)
        data = struct.pack(('%di' % len(inner_extend)), *inner_extend)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()


    return 0


if __name__ == '__main__':
    prod_f32()
    print("end")

