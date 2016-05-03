//**********************************************
// 8x8 DCT on Cuda types and declarations
//**********************************************
#ifndef H_DCT_8X8
#define H_DCT_8X8
#include "cuda_runtime.h"
#include "utilities.h"

//----------------------
// DEFINES
//----------------------
// 10 digits Pi
#define pi 3.1415926535f

// Pi/2N coeff
#define ROW_COEF pi/16.0f
#define COL_COEF pi/16.0f 

// Matrix dims
#define ROW_NUMBER 8
#define COL_NUMBER 8

// Matrix sizes (8x8)
#define DCT_MATRIX_SIZE 64

// Number of thread for the DCT kernel
#define DCT_KERNEL_THREADS 1024

//*************************************************************
// Compute complete DCT of a 8x8 sub matrixes. This kernel is
// desgined to run in a single block of 32x32 threads. Threads
// are identified by matrixes dimensions so the kernel can
// compute at most 1024 DCT in parallel
// @param srcMatrixes : a pointer to the first element of the
//                      the first source matrix
// @param dstMatrixes : a pointer to the first element of the
//                      first dest matrix
// @param matrixesDim : dimensions of matrixes
//*************************************************************
__global__ void computeDCT2(float* srcMatrixes, float* dstMatrixes, int numberOfSubMatrixes);

//*******************************************************************
// Wrapper for calling kernel from Cpp source file
//*******************************************************************
cudaError_t wrapperDCT2(float* srcMatrixes, float* dstMatrixes, int numberOfSubMatrixes);

#endif //H_DCT_8X8