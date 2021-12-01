CC = csky-abiv2-linux-gcc
INCLUDE = -I../../../include -I../../utils
CFLAGS += -O2 -g3 -mhard-float -mcpu=c860

test_objs =

test_objs += add.o
test_objs += avgpool.o
# test_objs += batch_normalization.o
test_objs += concat.o
# test_objs += crop.o
# test_objs += depth_to_space.o
test_objs += flatten.o
test_objs += global_avgpool.o
test_objs += global_maxpool.o
test_objs += leaky_relu.o
test_objs += lrn.o
test_objs += maximum.o
test_objs += maxpool.o
test_objs += mean.o
test_objs += minimum.o
# test_objs += negative.o
test_objs += pad.o
test_objs += prelu.o
test_objs += relu.o
test_objs += relu1.o
test_objs += relu6.o
test_objs += reshape.o
test_objs += resize_bilinear.o
test_objs += resize_nearest_neighbor.o
test_objs += sigmoid.o
test_objs += space_to_depth.o
test_objs += split.o
test_objs += squeeze.o
# test_objs += strided_slice.o
test_objs += sub.o
test_objs += tanh.o
test_objs += transpose.o


# test_objs += abs.o
# test_objs += and.o
# test_objs += argmax.o
# test_objs += argmin.o
# test_objs += batch_to_space.o
# test_objs += clip.o
test_objs += convolution.o
# test_objs += convolution_relu.o
# test_objs += convolution_relu6.o
test_objs += deconvolution.o
test_objs += depthwise_convolution.o
# test_objs += depthwise_deconvolution.o
test_objs += div.o
# test_objs += elu.o
# test_objs += equal.o
# test_objs += exp.o
# test_objs += expand_dims.o
# test_objs += floor.o
# test_objs += floor_divide.o
test_objs += fullyconnected.o
# test_objs += gather.o
# test_objs += gather_nd.o
# test_objs += greater.o
# test_objs += greater_equal.o
test_objs += group_convolution.o
# test_objs += l2_normalization.o
# test_objs += l2_pool.o
# test_objs += less.o
# test_objs += less_equal.o
# test_objs += log.o
# test_objs += log_softmax.o
# test_objs += matmul.o
# test_objs += max.o
# test_objs += maxpool2d_locat.o
# test_objs += min.o
# test_objs += mul.o
# test_objs += not_equal.o
# test_objs += or.o
# test_objs += pow.o
# test_objs += prod.o
# test_objs += psroipooling.o
# test_objs += relun.o
# test_objs += reorg.o
# test_objs += reverse.o
# test_objs += roipooling.o
# test_objs += rsqrt.o
# test_objs += select.o
# test_objs += shuffle_channel.o
# test_objs += slice.o
# test_objs += sin.o
# test_objs += softmax.o
# test_objs += softplus.o
# test_objs += softrelu.o
# test_objs += space_to_batch.o
# test_objs += sqrt.o
# test_objs += square.o
# test_objs += stack.o
# test_objs += sum.o
# test_objs += tile.o
# test_objs += topk.o
# test_objs += unpooling.o
# test_objs += unstack.o


utils_objs =

utils_objs += ../../utils/math_snr.o
utils_objs += ../../utils/test_utils.o

all: csi

csi: $(utils_objs) $(test_objs)

$(utils_objs): %.o: %.c
	$(CC) -c $(CFLAGS) $(INCLUDE) $< -o $@

$(test_objs): %.o: %.c
	$(CC) -c $(CFLAGS) $(INCLUDE) $< -o $@
	$(CC) -c $(CFLAGS) $(INCLUDE) ../$< -o ../$@
	$(CC) $@ ../$@ $(CFLAGS) $(BOARD) $(utils_objs) -L../../../lib -L../../../module/acuity-driver/lib/acuity-ovxlib-dev/lib/ \
	-ljpeg -lpng -lz -lcsi_nn2_openvx -Wl,-unresolved-symbols=ignore-in-shared-libs -lm -o $@.elf

clean:
	rm -rf  $(test_objs) $(utils_objs) *.a *.asm *.elf *.bin *.asm
