//**********************************************
// Online Detection Parallel implementation
// Function and type declaration
//**********************************************
#ifndef H_DETECTION_CUDA
#define H_DETECTION_CUDA
#include "cuda_runtime.h"
#include "utilities.h"

//----------------------------
// DEFINES
//----------------------------
// Threads dimension for the kernels
#define CUT_IMG_KERNEL_THREADS 32
#define MARK_IMG_KERNEL_THREADS 32
#define CALC_DISTANCE_KERNEL_THREADS 1024

//*********************************************************************************
// Parallel version of cutImage. Split given image into image blocks of blockSize^2
// this kernel is designed to work with bytes values only
//*********************************************************************************
// @param imageSrc    : source image buffer
// @param imgWidth    : source image width
// @param imgHeight   : source image height
// @param blockSize   : size of image block
// @param imageBlocks : image blocks buffer
//*********************************************************************************
__global__ void cutImageP(unsigned char* imageSrc, utilities::ROI imageDims, int blockSize, unsigned char* imageBlocks);

//*********************************************************
// Wrapper for calling the kermel from standard source file
//*********************************************************
cudaError_t cutImagePWrapper(unsigned char* imageSrc, const utilities::ROI& imageDims, int blockSize, unsigned char* imageBlocks);

//*********************************************************************************
// Parallel version of a part of onlineDetection algorithm. Given a matrix holding 
// distance between each image block signature and reference signature, it computes
// with the given treshold, if the block should be treated as defectious or not.
// If so, the block will be copied in a matrix with other defectious blocks and be
// shown to the user 
//*********************************************************************************
// @param alpha         : treshold used to treat a block defectious
// @param srcDistMatrix : a pointer to the matrix holding euclidian distances
//                        between reference block signature and images block
//                        signature
// @param imageBlocks   : a pointer to the matrix holding image blocks
// @param markedImage   : a pointer to the matrix which will show the defectious
//                        blocks
// @note this kernel is intended to be use with only one block in the grid and a
//       two dimension N*N threads per blocks
//*********************************************************************************
__global__ void markImageDefectsP(float alpha, float* srcDistMatrix, size_t distMatrixSize, utilities::ROI distMatrixDims, int blockSize, unsigned char* imageBlocks, unsigned char* markedImage, utilities::ROI imageDims);

//*********************************************************
// Wrapper for calling the kernel markImageDefectsP from 
// standard source file
//*********************************************************
cudaError_t markImageDefectsPWrapper(float alpha, float* srcDistMatrix, int blockSize, utilities::ROI imgBlockDims, unsigned char* imageBlocks, unsigned char* markedImage, utilities::ROI imageDims);

//**********************************************************************************
// Compute euclidian distance between refSign and signs vectors, stores it in
// distances vector
// @param refSign      : a pointer to the first element of the refernce signature
// @param signs        : a pointer to the first element of the first signature
// @param distances    : a pointer to the first element where to store distances
// @param distanceMean : a pointer to the float which hold distances mean value (in)
//**********************************************************************************
__global__ void computeEuclidianDistanceP(float* refSign, float* signs, int numberOfsigns, float* distances, float* distanceMean);

//*********************************************************
// Wrapper for calling the kernel markImageDefectsP from 
// standard source file
//*********************************************************
cudaError_t computeEuclidianDistancePWrapper(float* refSign, float* signs, int numberOfsigns, float* distances, float* distanceMean);

#endif //H_DETECTION_CUDA