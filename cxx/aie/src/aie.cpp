/*
 * Copyright (C) 2020-2021 Alibaba Group Holding Limited
 */
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <stdio.h>

#include "aie.h"
extern "C" {
#include "csi_ovx.h"
extern unsigned char _binary_model_params_start[];
void *csinn_(char *params);
}

AiEngineAlg::AiEngineAlg() : sess(NULL)
{
	std::cout << "AiEngineAlg::AiEngineAlg()" << std::endl;
}

AiEngineAlg::~AiEngineAlg()
{
	std::cout << "AiEngineAlg::~AiEngineAlg()" << std::endl;
}


int AiEngineAlg::Open(int idx) {
	std::cout << "AiEngineAlg::Open(" << idx << ")" << std::endl;
    return 0;
}

int AiEngineAlg::Close() {
    return 0;
}

int AiEngineAlg::LoadNet(AiNet *net) {
	std::cout << "AiEngineAlg::LoadNet()" << std::endl;
    // // read params data
    // std::vector<char> data;
    // std::ifstream file(net->weights.data(), std::ios::binary);
    // file.seekg(0, std::ios::end);
    // data.resize(file.tellg());
    // file.seekg(0);
    // file.read(reinterpret_cast<char*>(data.data()), data.size());
    // char *params = reinterpret_cast<char*>(data.data());
    // if (params == NULL) {
    //     return -1;
    // }
    // // create network
    // sess = (struct csi_session *)csinn_(params);

    char *params = reinterpret_cast<char*>(_binary_model_params_start);
    sess = (struct csi_session *)csinn_(params);

    return 0;
}

int AiEngineAlg::UnLoadNet() {
    csi_session_deinit(sess);
    csi_free_session(sess);

    return 0;
}

int AiEngineAlg::SetNetConfig(NetConfig_t *cfg)
{
    return 0;
}

int AiEngineAlg::GetNetConfig(NetConfig_t *cfg)
{
    return 0;
}

int AiEngineAlg::GetPerfProfile(std::vector<float>& timings)
{
    return 0;
}

int AiEngineAlg::SendInputTensor(Tensor_t *input, int32_t timeout)
{
	return 0;
}

int AiEngineAlg::Predict(Tensor_t **output, int32_t timeout)
{
	return 0;
}

int AiEngineAlg::ReleaseOutputTensor(Tensor_t *output)
{
	return 0;
}

int AiEngineAlg::GetInputTensor(Tensor_t **input) {
    *input = (Tensor_t *)malloc(sizeof(Tensor_t));
    int input_num;
    Tensor_t *input_p = *input;
    struct csi_tensor *input_tensor = csi_alloc_tensor(NULL);

    input_num = csi_get_input_number(sess);
    input_p->mtx_num = input_num;
    input_p->mtx = new Matrix_t[input_num];
    for (int i = 0; i < input_num; i++) {
        memset(input_tensor, 0, sizeof(struct csi_tensor));
        csi_get_input(i, input_tensor, sess);
        struct csi_quant_info *qinfo = input_tensor->qinfo;
        input_p->mtx[i].mquant.scale = qinfo->scale;
        input_p->mtx[i].mquant.zero_point = qinfo->zero_point;
        input_p->mtx[i].mdata.batch = 1;
        if (input_tensor->dim_count == 4) {
            input_p->mtx[i].mspec.depth = input_tensor->dim[1];
            input_p->mtx[i].mspec.rows = input_tensor->dim[2];
            input_p->mtx[i].mspec.cols = input_tensor->dim[3];
        } else if (input_tensor->dim_count == 3) {
            input_p->mtx[i].mspec.depth = 1;
            input_p->mtx[i].mspec.rows = input_tensor->dim[1];
            input_p->mtx[i].mspec.cols = input_tensor->dim[2];
        } else if (input_tensor->dim_count == 2) {
            input_p->mtx[i].mspec.depth = 1;
            input_p->mtx[i].mspec.rows = 1;
            input_p->mtx[i].mspec.cols = input_tensor->dim[1];
        }
        input_p->mtx[i].mdata.size = input_p->mtx[i].mspec.depth * input_p->mtx[i].mspec.rows * input_p->mtx[i].mspec.cols;
        free(input_tensor->data);
    }
    return 0;
}

int AiEngineAlg::GetOutputTensor(Tensor_t **output) {
    *output = (Tensor_t*)malloc(sizeof(Tensor_t));
    int output_num;
    Tensor_t *output_p = *output;
    struct csi_tensor output_tensor;

    output_num = csi_get_output_number(sess);
    output_p->mtx_num = output_num;
    output_p->mtx = new Matrix_t[output_num];
    for (int i = 0; i < output_num; i++) {
        memset(&output_tensor, 0, sizeof(struct csi_tensor));
        csi_get_output(i, &output_tensor, sess);
        struct csi_quant_info *qinfo = output_tensor.qinfo;
        output_p->mtx[i].mquant.scale = qinfo->scale;
        output_p->mtx[i].mquant.zero_point = qinfo->zero_point;
        if (output_tensor.dtype == CSINN_DTYPE_UINT8) {
            output_p->mtx[i].mdata.batch = 1;
        } else if (output_tensor.dtype == CSINN_DTYPE_FLOAT32) {
            output_p->mtx[i].mdata.batch = 3;
        }
        if (output_tensor.dim_count == 4) {
            output_p->mtx[i].mspec.depth = output_tensor.dim[1];
            output_p->mtx[i].mspec.rows = output_tensor.dim[2];
            output_p->mtx[i].mspec.cols = output_tensor.dim[3];
        } else if (output_tensor.dim_count == 3) {
            output_p->mtx[i].mspec.depth = 1;
            output_p->mtx[i].mspec.rows = output_tensor.dim[1];
            output_p->mtx[i].mspec.cols = output_tensor.dim[2];
        } else if (output_tensor.dim_count == 2) {
            output_p->mtx[i].mspec.depth = 1;
            output_p->mtx[i].mspec.rows = 1;
            output_p->mtx[i].mspec.cols = output_tensor.dim[1];
        }
        output_p->mtx[i].mdata.size = output_p->mtx[i].mspec.depth * output_p->mtx[i].mspec.rows * output_p->mtx[i].mspec.cols;
        free(output_tensor.data);
    }
    return 0;
}

int AiEngineAlg::SetInputTensor(Tensor_t *input) {
    // if (input->mtx[0].mdata.batch != 1) {
    //     std::cout << "do not support for multi batch data..." << std::endl;
    //     return -1;
    // }
    struct csi_tensor *input_tensor;
    for (int i = 0 ; i < input->mtx_num; i++) {
        input_tensor->data = *(input->mtx[i].mdata.data);
        csi_update_input(i, input_tensor, sess);
    }
    return 0;
}

int AiEngineAlg::SetOutputTensor(Tensor_t *output)
{
    output_t = output;
    // struct csi_tensor output_tensor;
    // for (int i = 0; i < output->mtx_num; i++) {
    //     memset(&output_tensor, 0, sizeof(struct csi_tensor));
    //     csi_get_output(i, &output_tensor, sess);
    //     if (output_tensor.dtype == CSINN_DTYPE_UINT8) {
    //         memcpy(output->mtx[i].mdata.data[0], output_tensor.data, output->mtx[i].mdata.size);
    //     } else if (output_tensor.dtype == CSINN_DTYPE_FLOAT32) {
    //         memcpy(output->mtx[i].mdata.data[0], output_tensor.data, output->mtx[i].mdata.size * 4);
    //     }
    //     free(output_tensor.data);
    // }
	return 0;
}

int AiEngineAlg::Run(int32_t timeout) {
    csi_session_run(sess);

    // get output data
    struct csi_tensor output_tensor;
    for (int i = 0; i < output_t->mtx_num; i++) {
        memset(&output_tensor, 0, sizeof(struct csi_tensor));
        csi_get_output(i, &output_tensor, sess);
        if (output_tensor.dtype == CSINN_DTYPE_UINT8) {
            memcpy(output_t->mtx[i].mdata.data[0], output_tensor.data, output_t->mtx[i].mdata.size);
        } else if (output_tensor.dtype == CSINN_DTYPE_FLOAT32) {
            memcpy(output_t->mtx[i].mdata.data[0], output_tensor.data, output_t->mtx[i].mdata.size * 4);
        }
        free(output_tensor.data);
    }
}

int AiEngineAlg::ReleaseTensor(Tensor_t *tensor) {
    delete tensor->mtx;
    return 0;
}

extern "C" {
AiEngine* AiEngineInstance() { return new AiEngineAlg(); }
}

int AiEngine::GetCapability(AiEngineCap_t *cap)
{
	if (cap == NULL) {
		return HAL_ERRNO_COMMON(COMMON_ERRNO_PARAMETER);
	}
	cap->max_channels = -1;
	return AI_SUCCESS;
}
