/*
 * Copyright (C) 2020-2021 Alibaba Group Holding Limited
 */
#pragma once

#include <string>
#include <vector>

#include "hal/hal_common_image.h"

typedef struct {
	int batch;
	int size;
	void **data; //data[batch][size]
} MatrixData_t;

typedef struct {
	int rows;
	int cols;
	int depth;
	HALImage::PixelFormat_e format;
} MatrixSpec_t;

typedef struct {
	int zero_point;
	float scale;
} MatrixQuant_t;

typedef struct {
	MatrixSpec_t mspec;
	MatrixData_t mdata;
	MatrixQuant_t mquant;
} Matrix_t;

typedef struct {
	Matrix_t *mtx;
	int mtx_num;
} Tensor_t;

typedef struct {
	int batch;
	int sched_mode;
	int resize_type;
} NetConfig_t;

class AiNet
{
public:
	AiNet(const std::string model) {
		this->model = model;
	}
	AiNet(const std::string model, const std::string weights) {
		this->model = model;
		this->weights = weights;
	}

public:
	std::string model;
	std::string weights;
};

typedef struct {
	uint32_t max_channels; // [0, max_channels - 1]
} AiEngineCap_t;

class AiEngine
{
public:
	virtual ~AiEngine(){}

	virtual int Open(int idx) = 0;
	virtual int Close() = 0;
	virtual int LoadNet(AiNet *net) = 0;
	virtual int UnLoadNet() = 0;
	virtual int SetNetConfig(NetConfig_t *cfg) = 0;
	virtual int GetNetConfig(NetConfig_t *cfg) = 0;
	virtual int GetPerfProfile(std::vector<float>& timings) = 0;

	virtual int SendInputTensor(Tensor_t *input, int32_t timeout) = 0;
	virtual int Predict(Tensor_t **output, int32_t timeout) = 0;
	virtual int ReleaseOutputTensor(Tensor_t *output) = 0;

	virtual int GetInputTensor(Tensor_t **input) = 0;
	virtual int GetOutputTensor(Tensor_t **output) = 0;
	virtual int SetInputTensor(Tensor_t *input) = 0;
	virtual int SetOutputTensor(Tensor_t *output) = 0;
	virtual int Run(int32_t timeout) = 0;
	virtual int ReleaseTensor(Tensor_t *tensor) = 0;

public:
	static int GetCapability(AiEngineCap_t *cap);
};

extern "C" {
AiEngine* AiEngineInstance();
}
