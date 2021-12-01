/*
 * Copyright (C) 2020-2021 Alibaba Group Holding Limited
 */
#ifndef AIE_SIM_H
#define AIE_SIM_H

#include <vector>

#include "hal/hal_errno.h"
#include "hal/hal_aie.h"

extern "C" {
#include "csi_ovx.h"
}

class AiEngineAlg:public AiEngine
{
public:
	AiEngineAlg();
	~AiEngineAlg();

	virtual int Open(int idx);
	virtual int Close();
	virtual int LoadNet(AiNet *net);
	virtual int UnLoadNet();
	virtual int SetNetConfig(NetConfig_t *cfg);
	virtual int GetNetConfig(NetConfig_t *cfg);
	virtual int GetPerfProfile(std::vector<float>& timings);

	virtual int SendInputTensor(Tensor_t *input, int32_t timeout);
	virtual int Predict(Tensor_t **output, int32_t timeout);
	virtual int ReleaseOutputTensor(Tensor_t *output);

	virtual int GetInputTensor(Tensor_t **input);
	virtual int GetOutputTensor(Tensor_t **output);
	virtual int SetInputTensor(Tensor_t *input);
	virtual int SetOutputTensor(Tensor_t *output);
	virtual int Run(int32_t timeout);
	virtual int ReleaseTensor(Tensor_t *tensor);

private:
	struct csi_session *sess;
	Tensor_t *output_t;
};

#endif
