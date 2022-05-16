#! /bin/bash

ROOT_PATH=$PWD
GCC_LINK=https://github.com/T-head-Semi/csi-nn2/releases/download/v1.12/Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.2.5-20220323.tar.gz
TAR_FILE=$ROOT_PATH/tools/Xuantie-900-gcc-linux-toolchain.tar.gz
COLOR="\\E[5;33m"

# create dir
GCC_PATH=tools/gcc-toolchain
mkdir -p $GCC_PATH

# download XuanTie-GNU-toolchain
if [ ! -f "$TAR_FILE" ]; then
	wget -O $TAR_FILE $GCC_LINK
fi

# unzip file
if [ -f "$TAR_FILE" ]; then
	if [ ! -d $ROOT_PATH/$GCC_PATH/bin ]; then
		tar -zxvf $TAR_FILE -C $ROOT_PATH/$GCC_PATH --strip-components 1
	fi
fi

echo -e  $COLOR"Download toolchain in $ROOT_PATH/tools/"
echo -e $COLOR"Set env variable:
	export PATH=$ROOT_PATH/$GCC_PATH/bin:$ \bPATH"
