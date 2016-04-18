#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdint.h>

//#include "utilities.h"
//#include "cut_image.h"
//#include "kim.h"
#include "cuda_utils.h"
#include "dct8x8.cuh"

using namespace cv;
//using namespace utilities;
using namespace std;

//*********************************************************************************
// Parallel version of cutImage. Split given image into image blocks of blockSize^2
// this kernel is designed to work with 8 bits values only
//*********************************************************************************
// @param imageSrc    : source image buffer
// @param imgWidth    : source image width
// @param imgHeight   : source image height
// @param blockSize   : size of image block
// @param imageBlocks : image blocks buffer
//*********************************************************************************
__global__ void cutImageP(unsigned char* imageSrc, int imgWidth, int imgHeight, int blockSize, unsigned char* imageBlocks)
{
    int threadX = threadIdx.x;
    int threadY = threadIdx.y;

    // dimensions of block in image
    int blockWidth  = ceilf(imgWidth / blockSize);
    int blockHeight = ceilf(imgHeight / blockSize);
    int numberOfBlocks = blockWidth*blockHeight;

    // number of element per image block
    int elemsPerBlocks = blockSize*blockSize;

    // block index
    int blockIndex = threadY + threadX*blockWidth;

    // index on the image the block starts
    int blockStartIndex = threadX*blockWidth*elemsPerBlocks + threadY*blockSize;
    
    // index(x) on the image block ends
    int blockLimitIndex = blockStartIndex + (blockSize - 1)*imgWidth;

    // index of the blocks destination buffer
    int imageBlocksIndex;

    // align destination buffer index with index of image block
    imageBlocksIndex = blockIndex*elemsPerBlocks;

    if ( blockIndex < numberOfBlocks)
    {
        int i, j;
        
        // fill the image block buffer
        for (i = blockStartIndex; i <= blockLimitIndex; i = i + imgWidth)
        {
            for (j = i; j < (i + 4); ++j)
            {
                imageBlocks[imageBlocksIndex] = imageSrc[j];
                ++imageBlocksIndex;
            }
        }
    }
}

int main(int argc, char** argv)
{
    int blockSize = 4;
    Mat image = imread("E:\\jee\\cours\\GEN5023\\code\\textile_images\\test3.bmp", IMREAD_GRAYSCALE);
    uchar* imageData = image.data;
    size_t imageSize = (size_t)image.size().area();
    
    uchar *devImageData, *devImageBlocks, *hostImageBlocks;

    // allocate data on host and device
    hostImageBlocks = (uchar*)malloc(imageSize);

    cudaError_t allocRes = cudaMalloc(&devImageData, imageSize);

    if (allocRes != cudaSuccess)
    {
        printf("Error while allocating devImageData on device\n");
        exit(EXIT_FAILURE);
    }

    allocRes = cudaMalloc(&devImageBlocks, imageSize);

    if (allocRes != cudaSuccess)
    {
        printf("Error while allocating devImageBlocks on device\n");
        exit(EXIT_FAILURE);
    }

    // copy data on device
    cudaError_t memCpyRes = cudaMemcpy(devImageData, imageData, imageSize, cudaMemcpyHostToDevice);

    if (memCpyRes != cudaSuccess)
    {
        printf("Error while copying devImageData on device\n");
        exit(EXIT_FAILURE);
    }

    // run kernel
    dim3 threads(16, 16);
    cutImageP << <1, threads >> >(devImageData, image.cols, image.rows, blockSize, devImageBlocks);
    getLastCudaError("Kernel execution failed");

    // copy data back on host
    memCpyRes = cudaMemcpy(hostImageBlocks, devImageBlocks, imageSize, cudaMemcpyDeviceToHost);

    if (memCpyRes != cudaSuccess)
    {
        printf("Error while copying devImageBlocks on host\n");
        exit(EXIT_FAILURE);
    }

    int elemPerblocs = blockSize*blockSize;
    int i;

    // uncut images from block
    for (i = 0; i < imageSize/blockSize; i = i + elemPerblocs)
    {
        // create a block buffer
        uchar blockBuffer[16];
        int j, k;
        j = 0;
        printf("\n[");
        for (k = i; k < i+elemPerblocs; ++k)
        {
            //blockBuffer[j] = hostImageBlocks[k];
            printf(" %d ", hostImageBlocks[k]);
            ++j;
        }

        printf("]\n");
        //Mat aBlock(blockSize, blockSize, CV_8U, blockBuffer);
    }

    return 0;
}