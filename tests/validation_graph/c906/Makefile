CC = riscv64-unknown-linux-gnu-gcc
INCLUDE = -I../../../include -I../../utils
CFLAGS = -O2 -g3 -march=rv64gcv0p7_zfh_xtheadc -mabi=lp64d -static

test_objs =

test_objs += add.o
# test_objs += argmax.o
test_objs += avgpool.o
# test_objs += batch_normalization.o
# test_objs += batch_to_space_nd.o
test_objs += concat.o
test_objs += convolution.o
# test_objs += crop.o
test_objs += deconvolution.o
test_objs += depthwise_convolution.o
test_objs += depth_to_space.o
test_objs += div.o
test_objs += flatten.o
test_objs += fullyconnected.o
test_objs += global_avgpool.o
test_objs += global_maxpool.o
test_objs += group_convolution.o
# test_objs += l2_normalization.o
test_objs += leaky_relu.o
# test_objs += lrn.o
test_objs += maximum.o
test_objs += maxpool.o
# test_objs += mean.o
test_objs += minimum.o
# test_objs += mul.o
# test_objs += negative.o
test_objs += pad.o
# test_objs += prelu.o
test_objs += relu.o
test_objs += relu1.o
test_objs += relu6.o
test_objs += reshape.o
test_objs += resize_bilinear.o
test_objs += resize_nearest_neighbor.o
test_objs += sigmoid.o
# test_objs += softmax.o
# test_objs += space_to_batch_nd.o
test_objs += space_to_depth.o
test_objs += split.o
test_objs += squeeze.o
# test_objs += strided_slice.o
test_objs += sub.o
test_objs += tanh.o
test_objs += transpose.o


utils_objs =
utils_objs += ../../utils/test_utils.o


all: csi

csi: $(utils_objs) $(test_objs)

$(utils_objs): %.o: %.c
	$(CC) -c $(CFLAGS) $(INCLUDE) $< -o $@

$(test_objs): %.o: %.c
	$(CC) -c $(CFLAGS) $(INCLUDE) $< -o $@
	$(CC) -c $(CFLAGS) $(INCLUDE) ../$< -o ../$@
	$(CC) $@ ../$@ $(CFLAGS) $(BOARD) $(utils_objs) -L../../../c906_static_build/ \
	../../../c906_static_build/libshl_c906.a -lpthread -lc -lm -lstdc++ -o $@.elf

clean:
	rm -rf  $(test_objs) $(utils_objs) *.a *.asm *.elf *.bin *.asm
