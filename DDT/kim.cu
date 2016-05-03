//------------------------------------------------
// implementation file of kim signature extraction
//-------------------------------------------------

//----------------------------
// INCLUDES
//----------------------------
#include "kim.cuh"
#include "device_launch_parameters.h"
#include "thrust/sort.h"
#include "thrust/device_vector.h"
#include "math.h"

//----------------------------
// NAMESPACES
//----------------------------
using namespace std;
using namespace utilities;

//----------------------------
// IMPLEMENTATION
//----------------------------

//***********************************************************
// Run phase 1 of kim signature - see header file for details
//***********************************************************
__global__ void extractKimSignaturePhase1(unsigned char* imageBlocks, ROI imageBlocksDims, int blockSize, float* meanMatrixes)
{
    // threads index
    int threadX = threadIdx.x;
    int threadY = threadIdx.y;

    if (threadX < imageBlocksDims.height && threadY < imageBlocksDims.width)
    {
        // block index by thread index
        int blockIndex = threadX*imageBlocksDims.width + threadY;
        
        // elements per blocks
        int elemPerBlocks = blockSize*blockSize;

        // the index of the first element of the block in imageBlocks chunk
        int blockStartIndex = blockIndex*elemPerBlocks;

        // now for the current block defined by blockStartIndex, cut it 64 (8x8) sub blocks.
        // like image blocks, sub blocks should be square matrix nxn
        int subBlockSize = blockSize / MEAN_MATRIX_DIM;

        // element per sub blocks
        int elemPerSubBlocks = subBlockSize*subBlockSize;

        // Index where to 
        int meanMatrixesIndex = blockIndex*MEAN_MATRIX_SIZE;

        // parse current image block. i and j iterates subBlockSize by subBlockSize, 
        //k iterates on sub block rows and l iterates over sub block cols
        for (int i = blockStartIndex; i < blockStartIndex + elemPerBlocks - blockSize; i = i + blockSize*subBlockSize)
        {
            for (int j = i; j < i + blockSize; j = j + subBlockSize)
            {
                float meanBuffer = 0;
                for (int k = j; k < j+subBlockSize*blockSize; k = k + blockSize)
                {
                    for (int l = k; l < k + subBlockSize; ++l)
                    {
                        meanBuffer += imageBlocks[l];
                    }
                }

                //end parsing a block here
                // compute mean value of block
                meanBuffer = meanBuffer / elemPerSubBlocks;

                // insert directly in mean matrixes buffer
                meanMatrixes[meanMatrixesIndex] = meanBuffer;
                ++meanMatrixesIndex;
            }
        }
    }
}

//***************************************
// Launch phase 1 Kernel of kim signature
//***************************************
cudaError_t extractKimSignaturePh1(unsigned char* imageBlocks, const ROI& imageBlocksDims, int blockSize, float* meanMatrixes)
{
    cudaError_t ret = cudaSuccess;

    // allocate data on device
    float* devMeanMatrixes;
    unsigned char* devImageBlocks;

    int imageBlocksCount = imageBlocksDims.height*imageBlocksDims.width;
    int elemPerImageBlock = blockSize*blockSize;

    size_t meanMatrixesSizeMem = imageBlocksCount*MEAN_MATRIX_SIZE*sizeof(float);

    ret = cudaMalloc(&devMeanMatrixes, meanMatrixesSizeMem);
    if (ret != cudaSuccess)
        return ret;

    ret = cudaMalloc(&devImageBlocks, imageBlocksCount*elemPerImageBlock);
    if (ret != cudaSuccess)
        return ret;

    // copy data on device
    ret = cudaMemcpy(devImageBlocks, imageBlocks, imageBlocksCount*elemPerImageBlock, cudaMemcpyHostToDevice);
    if (ret != cudaSuccess)
        return ret;

    // run kernel
    dim3 threadsDim(PH1_KERNEL_THREADS, PH1_KERNEL_THREADS, 1);
    extractKimSignaturePhase1<<<1, threadsDim>>>(devImageBlocks, imageBlocksDims, blockSize, devMeanMatrixes);
    
    // copy back data on host
    ret = cudaMemcpy(meanMatrixes, devMeanMatrixes, meanMatrixesSizeMem, cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(devImageBlocks);
    cudaFree(devMeanMatrixes);

    return ret;
}

//***************************************
// Helper function to sort an array with
// insertion method
//***************************************
__device__ void insertionSort(float* arr, int arrLen)
{
    for (int j = 1; j < arrLen; ++j)
    {
        float cle = arr[j];
        int i = j - 1;

        while (i >= 0 && arr[i] > cle)
        {
            arr[i + 1] = arr[i];
            i = i - 1;
        }
        arr[i + 1] = cle;
    }
}

//************************************
// Helper function to reverse an array 
//************************************
__device__ void reverseArr(float* arr, int arrLen)
{
    float tmp = 0;
    int start = 0;
    int end = arrLen - 1;
    while (start < end)
    {
        tmp = arr[start];
        arr[start] = arr[end];
        arr[end] = tmp;
        ++start;
        --end;
    }
}


//***********************************************************
// Run phase 3 of kim signature - see header file for details
//***********************************************************
__global__ void extractKimSignaturePhase3(float* dctMeanMatrixes, int numberOfMatrixes, float* kimSignatures)
{
    int matrixIndex = threadIdx.x*MEAN_MATRIX_SIZE;
    int signIndex = threadIdx.x*KIM_SIGN_SIZE;
    int tmpIndex = signIndex;
    // populate signature vector, get rid of DC coef (1 element)
    if (matrixIndex < numberOfMatrixes*MEAN_MATRIX_SIZE && signIndex < numberOfMatrixes*KIM_SIGN_SIZE)
    {
        for (int i = 0; i < KIM_SIGN_MATRIX_DIM; ++i)
        {
            for (int j = 0; j < KIM_SIGN_MATRIX_DIM; ++j)
            {
                if (i == 0 && j == 0)
                    continue;
                else
                {
                    kimSignatures[tmpIndex] = fabsf(dctMeanMatrixes[matrixIndex + i*MEAN_MATRIX_DIM + j]);
                    ++tmpIndex;
                }
            }
        }
        
        // sort in descending order
        insertionSort(&kimSignatures[signIndex], KIM_SIGN_SIZE);
        reverseArr(&kimSignatures[signIndex], KIM_SIGN_SIZE);
    }
}

//***************************************
// Launch phase 3 Kernel of kim signature
//***************************************
cudaError_t extractKimSignaturePh3(float* dctMeanMatrixes, int numberOfMatrixes, float* kimSignatures)
{
    cudaError_t ret = cudaSuccess;
    // allocate memory on device for dct matrixes - kim signatures are already allocated
    // because of thrust::device_vector
    float* devDctMeanMatrixes;
    float* devKimSignatures;

    size_t dctMatrixesSize = numberOfMatrixes*MEAN_MATRIX_SIZE*sizeof(float);
    size_t kimSignatureSize = numberOfMatrixes*KIM_SIGN_SIZE*sizeof(float);
    ret = cudaMalloc(&devDctMeanMatrixes, dctMatrixesSize);
    if (ret != cudaSuccess)
        return ret;

    ret = cudaMalloc(&devKimSignatures, kimSignatureSize);
    if (ret != cudaSuccess)
        return ret;

    // copy dct matrixes on device
    ret = cudaMemcpy(devDctMeanMatrixes, dctMeanMatrixes, dctMatrixesSize, cudaMemcpyHostToDevice);
    if (ret != cudaSuccess)
        return ret;

    // run kernel
    extractKimSignaturePhase3 << <1, PH3_KERNEL_THREADS >> >(devDctMeanMatrixes, numberOfMatrixes, devKimSignatures);

    // copy back data on host
    ret = cudaMemcpy(kimSignatures, devKimSignatures, kimSignatureSize, cudaMemcpyDeviceToHost);

    // deallocate data on device
    cudaFree(devDctMeanMatrixes);
    cudaFree(devKimSignatures);
    return ret;
}