#!/bin/sh -x

qemu-riscv64 ./add.o.elf add_graph_data_f32.bin
qemu-riscv64 ./avgpool.o.elf avgpool2d_nchw_data_f32.bin
qemu-riscv64 ./concat.o.elf concat_data_f32.bin
qemu-riscv64 ./convolution.o.elf convolution_nchw_data_f32.bin
qemu-riscv64 ./deconvolution.o.elf deconvolution_nchw_data_f32.bin
qemu-riscv64 ./depth_to_space.o.elf depth_to_space_data_f32.bin
qemu-riscv64 ./depthwise_convolution.o.elf depthwise_convolution_nchw_data_f32.bin
qemu-riscv64 ./div.o.elf div_data_f32.bin
qemu-riscv64 ./flatten.o.elf flatten_data_f32.bin
qemu-riscv64 ./fullyconnected.o.elf fullyconnected_data_f32.bin
qemu-riscv64 ./global_avgpool.o.elf global_avgpool2d_nchw_data_f32.bin
qemu-riscv64 ./global_maxpool.o.elf global_maxpool2d_nchw_data_f32.bin
qemu-riscv64 ./group_convolution.o.elf group_convolution_nchw_data_f32.bin
qemu-riscv64 ./leaky_relu.o.elf leaky_relu_data_f32.bin
# qemu-riscv64 ./lrn.o.elf lrn_data_f32.bin
qemu-riscv64 ./maximum.o.elf maximum_data_f32.bin
qemu-riscv64 ./maxpool.o.elf maxpool2d_nchw_data_f32.bin
# qemu-riscv64 ./mean.o.elf mean_graph_data_f32.bin
qemu-riscv64 ./minimum.o.elf minimum_data_f32.bin
# qemu-riscv64 ./mul.o.elf mul_data_f32.bin
qemu-riscv64 ./pad.o.elf pad_nchw_data_f32.bin
# qemu-riscv64 ./prelu.o.elf prelu_nhwc_data_f32.bin
qemu-riscv64 ./relu.o.elf relu_data_f32.bin
qemu-riscv64 ./relu1.o.elf relu1_data_f32.bin
qemu-riscv64 ./relu6.o.elf relu6_data_f32.bin
qemu-riscv64 ./reshape.o.elf reshape_data_f32.bin
qemu-riscv64 ./resize_bilinear.o.elf resize_bilinear_nchw_data_f32.bin
qemu-riscv64 ./resize_nearest_neighbor.o.elf resize_nearestneighbor_data_f32.bin
qemu-riscv64 ./sigmoid.o.elf sigmoid_data_f32.bin
qemu-riscv64 ./space_to_depth.o.elf space_to_depth_data_f32.bin
qemu-riscv64 ./split.o.elf split_graph_data_f32.bin
qemu-riscv64 ./squeeze.o.elf squeeze_data_f32.bin
qemu-riscv64 ./sub.o.elf sub_data_f32.bin
qemu-riscv64 ./tanh.o.elf tanh_data_f32.bin
qemu-riscv64 ./transpose.o.elf transpose_data_f32.bin
