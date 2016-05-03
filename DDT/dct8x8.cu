//----------------------
// INCLUDES
//----------------------
#include "dct8x8.cuh"
#include "math.h"
#include "device_launch_parameters.h"

//----------------------
// IMPLEMENTATION
//----------------------

//*********************************
// Alpha for computing forward 2Dct
//*********************************
__device__ float alpha(int i)
{
    if (i == 0)
        return sqrt(0.125);
    else
        return sqrt(0.25);
}

//***********************************************************
// Compute DCT coef of value (u,v) in matrix
// @param u       : row index
// @param v       : col index
// @param aMatrix : source matrix
// @param offset  : offset of first element 
// @return        : dct coeff for
//***********************************************************
__device__ float computeDCTCoef(int u, int v, float* aMatrix, int offset)
{
    float res = alpha(u)*alpha(v);
    float tmp = 0.0f;

    for (int i = 0; i < ROW_NUMBER; ++i)
    {
        for (int j = 0; j < COL_NUMBER; ++j)
        {
            tmp += cosf(ROW_COEF*u*(2 * i + 1))*cosf(COL_COEF*v*(2 * j + 1))*aMatrix[offset + i * ROW_NUMBER + j];
        }
    }
    return res*tmp;
}

//*************************************************************
// Compute complete DCT of a 8x8 matrix
// See header file for details
//*************************************************************
__global__ void computeDCT2(float* srcMatrixes, float* dstMatrixes, int numberOfMatrixes)
{
    // sub matrix starting index and offset
    int matrixIndex = threadIdx.x*DCT_MATRIX_SIZE;
    int offset = matrixIndex;

    if (matrixIndex < (numberOfMatrixes*DCT_MATRIX_SIZE))
    {
        for (int u = 0; u < ROW_NUMBER; ++u)
        {
            for (int v = 0; v <  COL_NUMBER; ++v)
            {
                dstMatrixes[matrixIndex] = computeDCTCoef(u, v, srcMatrixes, offset);
                ++matrixIndex;
            }
        }
    }
}

//*******************************************************************
// Wrapper for calling kernel from Cpp source file
// See header file for details
//*******************************************************************
cudaError_t wrapperDCT2(float* srcMatrixes, float* dstMatrixes, int numberOfMatrixes)
{
    cudaError_t ret = cudaSuccess;

    // allocate data on device
    float* devSrcMatrixes;
    float* devDstMatrixes;
    size_t matrixesSize = numberOfMatrixes*DCT_MATRIX_SIZE*sizeof(float);
    ret = cudaMalloc(&devSrcMatrixes, matrixesSize);
    if (ret != cudaSuccess)
        return ret;

    ret = cudaMalloc(&devDstMatrixes, matrixesSize);
    if (ret != cudaSuccess)
        return ret;

    // copy source matrixes on device
    ret = cudaMemcpy(devSrcMatrixes, srcMatrixes, matrixesSize, cudaMemcpyHostToDevice);
    if (ret != cudaSuccess)
        return ret;

    // run kernel
    computeDCT2 << <1, DCT_KERNEL_THREADS >> >(devSrcMatrixes, devDstMatrixes, numberOfMatrixes);
    
    // copy destination matrixes back on host
    ret = cudaMemcpy(dstMatrixes, devDstMatrixes, matrixesSize, cudaMemcpyDeviceToHost);
    if (ret != cudaSuccess)
        return ret;

    cudaFree(devSrcMatrixes);
    cudaFree(devDstMatrixes);
    return ret;
}