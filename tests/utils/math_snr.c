/*
 * Copyright (C) 2016-2021 C-SKY Limited. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* CSI-NN2 version 1.8.x */

/* ----------------------------------------------------------------------
*        Include project header files
* -------------------------------------------------------------------- */
#include"math_snr.h"

/**
 * @brief  Caluclation of SNR for float data
 * @param  pRef     Pointer to the reference buffer
 * @param  pTest    Pointer to the test buffer
 * @param  buffSize    total number of samples
 * @return SNR
 * The function Caluclates signal to noise ratio for the reference output
 * and test output
 */

float csi_snr_f32(float *pRef, float *pTest, uint32_t buffSize)
{
    double EnergySignal = 0.0, EnergyError = 0.0;
    uint32_t i;
    float SNR;
    int temp;
    int *test;

    for (i = 0; i < buffSize; i++)
    {
        /* Checking for a NAN value in pRef array */
        test =   (int *)(&pRef[i]);
        temp =  *test;

        if(temp == 0x7FC00000)
        {
            return(0);
        }

        /* Checking for a NAN value in pTest array */
        test =   (int *)(&pTest[i]);
        temp =  *test;

        if(temp == 0x7FC00000)
        {
            return(0);
        }
        EnergySignal += (double)pRef[i] * (double)pRef[i];
        EnergyError += ((double)pRef[i] - (double)pTest[i]) * ((double)pRef[i] - (double)pTest[i]);
    }

    /* Checking for a NAN value in EnergyError */
    test =   (int *)(&EnergyError);
    temp =  *test;

    if(temp == 0x7FC00000)
    {
        return(0);
    }

    if(EnergyError == 0.0)
    {
        return(MAX_SNR_VALUE);
    }

    SNR = (float)10 * log10 (EnergySignal / EnergyError);

    return (SNR);

}

/**
 * @brief  Caluclation of SNR for Q31 data
 * @param  pRef     Pointer to the reference buffer
 * @param  pTest    Pointer to the test buffer
 * @param  buffSize    total number of samples
 * @return SNR
 * The function Caluclates signal to noise ratio for the reference output
 * and test output
 */
float csi_snr_q31(int32_t *pRef, int32_t *pTest, uint32_t buffSize)
{
    double  EnergySignal = 0.0, EnergyError = 0.0;
    double tempRef, tempTest;
    uint32_t i;
    float SNR;

    for (i = 0; i < buffSize; i++)
    {
        tempRef       = (double)pRef[i] / pow(2, 31);
        tempTest      = (double)pTest[i] / pow(2, 31);
        EnergySignal += tempRef * tempRef;
        EnergyError  += (tempTest - tempRef) * (tempTest - tempRef);
    }


    if(EnergyError == 0.0)
    {
        return(MAX_SNR_VALUE);
    }

    SNR = (float)10 * log10 (EnergySignal / EnergyError);

    return (SNR);

}


/**
 * @brief  Caluclation of SNR for Q7 data
 * @param  pRef     Pointer to the reference buffer
 * @param  pTest    Pointer to the test buffer
 * @param  buffSize    total number of samples
 * @return SNR
 * The function Caluclates signal to noise ratio for the reference output
 * and test output
 */
float csi_snr_q7(int8_t *pRef, int8_t *pTest, uint32_t buffSize)
{
    double  EnergySignal = 0.0, EnergyError = 0.0;
    double tempRef, tempTest;
    uint32_t i;
    float SNR;

    for (i = 0; i < buffSize; i++)
    {
        tempRef       = pRef[i] / pow(2, 6);
        tempTest      = pTest[i] / pow(2, 6);
        EnergySignal += tempRef * tempRef;
        EnergyError  += (tempTest - tempRef) * (tempTest - tempRef);
    }

    SNR = (float)10 * log10 (EnergySignal / EnergyError);

    return (SNR);

}
