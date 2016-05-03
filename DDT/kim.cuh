//**********************************************
// Cut Image function and types declaration
//**********************************************
#ifndef H_KIM_CUDA
#define H_KIM_CUDA
#include "cuda_runtime.h"
#include "utilities.h"

//----------------------------
// DEFINES
//----------------------------
// Dimensions of a mean matrix (NxN)
#define MEAN_MATRIX_DIM 8

// Size of a mean matrix (8x8)
#define MEAN_MATRIX_SIZE 64

// Dimensions of kim sub matrix (6x6)
#define KIM_SIGN_MATRIX_DIM 6

// Size of kim signature
#define KIM_SIGN_SIZE 35

// Threads dimension for the kernel
#define PH1_KERNEL_THREADS 32
#define PH3_KERNEL_THREADS 1024

//******************************************************************
// Parallel version of Kim signature extraction, phase 1
// this phase consist in preparing image data before DCT processing.
// Each image block is cutted in 8x8 sub blocks. Then we compute
// mean value of these sub blocks and put these values in a matrix
// of size 8x8, one per image block. Each matrix is directly copied
// in meanMatrixes buffer which contains number of image blocks *
// number of element per mean matrix
// @param imageBlocks     : a pointer to an image block buffer
// @param imageBlocksDims : a ROI holding imageBlocks dimensions
// @param meanMatrixes    : a pointer to the first mean matrix 
//*****************************************************************
__global__ void extractKimSignaturePhase1(unsigned char* imageBlocks, utilities::ROI imageBlocksDims, int blockSize, float* meanMatrixes);

//*****************************************************
// Wrapper for lauching the phase 1 kernel 
// from any source file
//*****************************************************
cudaError_t extractKimSignaturePh1(unsigned char* imageBlocks, const utilities::ROI& imageBlocksDims, int elemPerImageBlock, float* meanMatrixes);

//******************************************************************
// Parallel version of Kim signature extraction, phase 3. For each
// mean matrixes got in phase 1, we computed their 8x8 dct (phase2).
// For each dct matrix, only 6x6 part will be kept, added into a 
// line vector without the DC coefficient and sorted in descending
// order. That is the kim signature.
// @param dctMeanMatrixes  : a pointer to the first element of the
//                           first dct matrix to treat
// @param numberOfMatrixes : the number of matrixes holded into
//                           dctMeanMatrixes
// @param kimSignatures    : the kim signature of each matrixes
__global__ void extractKimSignaturePhase3(float* dctMeanMatrixes, int numberOfMatrixes, float* kimSignatures);

//*****************************************************
// Wrapper for lauching the phase 3 kernel 
// from any source file
//*****************************************************
cudaError_t extractKimSignaturePh3(float* dctMeanMatrixes, int numberOfMatrixes, float* kimSignatures);

#endif // H_KIM_CUDA