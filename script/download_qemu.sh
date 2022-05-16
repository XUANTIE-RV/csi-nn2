#! /bin/bash

ROOT_PATH=$PWD
QEMU_LINK=https://github.com/T-head-Semi/csi-nn2/releases/download/v1.12/xuantie-qemu-x86_64-Ubuntu-18.04-20220402-0353.tar.gz
TAR_FILE=$ROOT_PATH/tools/Xuantie-Qemu-x86_64-Ubuntu-18.04.tar.gz
COLOR="\\E[5;33m"

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
	fi
fi

echo -e  $COLOR"Download toolchain in $ROOT_PATH/tools/"
echo -e $COLOR"Set env variable:
	export PATH=$ROOT_PATH/$QEMU_PATH/bin:$ \bPATH"
