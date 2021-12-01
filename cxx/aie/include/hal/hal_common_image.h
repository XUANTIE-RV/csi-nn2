/*
 * Copyright (C) 2020-2021 Alibaba Group Holding Limited
 */
#pragma once

#include <cstdint>
#include "hal_attributes.h"

/* IS_CAPABLE(VideoDecoderCap_t.codec_type, VIDEO_CODEC_TYPE_H264) */
#define IS_CAPABLE(value, cap) ((value & cap) == cap)

namespace HALImage {

typedef enum {
	PIXEL_FORMAT_YUV420P	= 1 << 0,	///< planar YUV 4:2:0, 12bpp, (1 Cr & Cb sample per 2x2 Y samples)
	PIXEL_FORMAT_NV12		= 1 << 1,	///< semi planar YUV 4:2:0, 12bpp, (1 Cr & Cb sample per 2x2 Y samples)
	PIXEL_FORMAT_YUV422P	= 1 << 2,	///< planar YUV 4:2:2, 16bpp, (1 Cr & Cb sample per 2x1 Y samples)
	PIXEL_FORMAT_YUYV422	= 1 << 3,	///< packed YUV 4:2:2, 16bpp, Y0 Cb Y1 Cr
	PIXEL_FORMAT_RGB888		= 1 << 4,	///< packed RGB 8:8:8, 24bpp, RGBRGB...
	PIXEL_FORMAT_GRAY8		= 1 << 5,	///< packed GRAY 8, 8bpp
	PIXEL_FORMAT_GRAY16		= 1 << 6,	///< packed GRAY 16, 16bpp
	PIXEL_FORMAT_JPEG		= 1 << 7,	///< packed JPEG
	PIXEL_FORMAT_RGB888P	= 1 << 8,	///< planar RGB 8:8:8
	PIXEL_FORMAT_NONE		= 1 << 9,	///< data format
} PixelFormat_e;

typedef enum {
	QVGA	= 1 << 0,	//320x240
	VGA		= 1 << 1,	//640x480
	XGA		= 1 << 2,	//1024x768
	HD720P	= 1 << 3,	//1280x720
	HD1080P	= 1 << 4,	//1920x1080
} Resolution_e;


typedef struct {
	uint32_t	width;
	uint32_t	height;
} ImageSize_t;

typedef struct {
	uint32_t	left;
	uint32_t	top;
	uint32_t	width;
	uint32_t	height;
} ImageRect_t;

typedef struct {
	uint32_t	left;
	uint32_t	top;
} ImagePos_t;

typedef struct {
	uint64_t		pts;
	uint32_t		width;
	uint32_t		height;
	uint32_t		size;
	PixelFormat_e	pixel_format;
	uint8_t*		phy_address[3];	/* YUV420 has 3 planars */
	uint8_t*		vir_address[3];
	attribute_deprecated
	int32_t			fd_of_device;//for DEV MEM, set with phy_address
} ImageInfo_t;

typedef enum {
	IMAGE_ROTATION_0	= 1 << 0,
	IMAGE_ROTATION_90	= 1 << 1,
	IMAGE_ROTATION_180	= 1 << 2,
	IMAGE_ROTATION_270	= 1 << 3,
} ImageRotation_e;

typedef struct {
	uint8_t	red; /* 0 - 255 */
	uint8_t	green;
	uint8_t	blue;
} ColorScalar_t;

} // namespace HALImage
