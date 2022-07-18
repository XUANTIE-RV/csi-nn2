#! /bin/bash

ROOT_PATH=$PWD
QEMU_LINK=https://github.com/T-head-Semi/csi-nn2/releases/download/v1.12/xuantie-qemu-x86_64-Ubuntu-18.04-20220402-0353.tar.gz
TAR_FILE=$ROOT_PATH/tools/Xuantie-Qemu-x86_64-Ubuntu-18.04.tar.gz

# create dir
QEMU_PATH=tools/qemu
mkdir -p $QEMU_PATH

# download XuanTie-QEMU
if [ ! -f "$TAR_FILE" ]; then
	wget -O $TAR_FILE $QEMU_LINK
fi

# unzip file
if [ -f "$TAR_FILE" ]; then
	if [ ! -d $ROOT_PATH/$QEMU_PATH/bin ]; then
		tar -zxvf $TAR_FILE -C $ROOT_PATH/$QEMU_PATH --strip-components 1
		echo "remove $TAR_FILE"
	fi
	rm -f $TAR_FILE
fi

if [ -d "$ROOT_PATH/$QEMU_PATH/bin" ]; then
	echo "remove usless file"
	ls $ROOT_PATH/$QEMU_PATH/bin |grep -v qemu-riscv64 |xargs -i rm -rf $ROOT_PATH/$QEMU_PATH/bin/{}
	ls $ROOT_PATH/$QEMU_PATH |grep -v bin |xargs -i rm -rf $ROOT_PATH/$QEMU_PATH/{}
fi


echo "Download toolchain in $ROOT_PATH/tools/"
echo "Set env variable:
	export PATH=$ROOT_PATH/$QEMU_PATH/bin:$ \bPATH"
