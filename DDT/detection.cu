//---------------------------------------------------
// implementation file of cutImage parallel function
//---------------------------------------------------

//----------------------------
// INCLUDES
//----------------------------
#include "detection.cuh"
#include "kim.cuh"
#include "dct8x8.cuh"
#include "device_launch_parameters.h"
#include "math.h"

//----------------------------
// NAMESPACE
//----------------------------
using namespace utilities;
using namespace std;

//----------------------------
// IMPLEMENTATION
//----------------------------

//**********************************************************************
// Parallel version of cutImage. Split given image into image blocks 
// of blockSize^2 this kernel is designed to work with bytes values only
//**********************************************************************
__global__ void cutImageP(unsigned char* imageSrc, ROI imageDims, int blockSize, unsigned char* imageBlocks)
{
    int threadX = threadIdx.x;
    int threadY = threadIdx.y;

    // dimensions of block in image
    int blockWidth = ceilf(imageDims.width / blockSize);
    int blockHeight = ceilf(imageDims.height / blockSize);

    // number of element per image block
    int elemsPerBlocks = blockSize*blockSize;

    // block index
    int blockIndex = threadY + threadX*blockWidth;

    // index on the image the block starts
    int blockStartIndex = threadX*blockWidth*elemsPerBlocks + threadY*blockSize;

    // index(x) on the image block ends
    int blockLimitIndex = blockStartIndex + (blockSize - 1)*imageDims.width;

    // index of the blocks destination buffer
    int imageBlocksIndex;

    // align destination buffer index with index of image block
    imageBlocksIndex = blockIndex*elemsPerBlocks;

    if (threadX < blockHeight && threadY < blockWidth)
    {
        int i, j;

        // fill the image block buffer
        for (i = blockStartIndex; i <= blockLimitIndex; i = i + imageDims.width)
        {
            for (j = i; j < (i + blockSize); ++j)
            {
                imageBlocks[imageBlocksIndex] = imageSrc[j];
                ++imageBlocksIndex;
            }
        }
    }
}

//*******************************************************************
// Wrapper for calling the kernel cutImageP from standard source file
//*******************************************************************
cudaError_t cutImagePWrapper(unsigned char* image, const ROI& imageDims, int blockSize, unsigned char* imageBlocks)
{
    cudaError_t ret = cudaSuccess;

    size_t imageSize = (size_t)(imageDims.width*imageDims.height);

    uchar* devImageData;
    uchar* devImageBlocks;

    ret = cudaMalloc(&devImageData, imageSize);

    if (ret != cudaSuccess)
    {
        printf("Error while allocating devImageData on device\n");
        return ret;
    }

    ret = cudaMalloc(&devImageBlocks, imageSize);

    if (ret != cudaSuccess)
    {
        printf("Error while allocating devImageBlocks on device\n");
        return ret;
    }

    // copy data on device
    ret = cudaMemcpy(devImageData, image, imageSize, cudaMemcpyHostToDevice);

    if (ret != cudaSuccess)
    {
        printf("Error while copying devImageData on device\n");
        return ret;
    }

    // run kernel
    dim3 threadsDim(CUT_IMG_KERNEL_THREADS, CUT_IMG_KERNEL_THREADS, 1);
    cutImageP << <1, threadsDim >> >(devImageData, imageDims, blockSize, devImageBlocks);

    // copy data back on host
    ret = cudaMemcpy(imageBlocks, devImageBlocks, imageSize, cudaMemcpyDeviceToHost);

    if (ret != cudaSuccess)
    {
        printf("Error while copying devImageBlocks on host\n");
        return ret;
    }
    return ret;
}


//**********************************************************************
// Parallel version of a part of processing of OnlineDetection algorithm
// See header file for details
//**********************************************************************
__global__ void markImageDefectsP(float alpha, float* srcDistMatrix, size_t distMatrixSize, ROI distMatrixDims, int blockSize, unsigned char* imageBlocks, unsigned char* markedImage, ROI imageDims)
{
    // thread index, used to index the block which will be processed
    int threadX = threadIdx.x;
    int threadY = threadIdx.y;

    // image indexes x-y
    int imageX = threadX*blockSize;
    int imageY = threadY*blockSize;

    // image index one dimension
    int imageIndex = imageX*imageDims.width + imageY;

    // distance matrix index and size, given by how many blocks
    int distIndex = threadX*distMatrixDims.width + threadY;

    // number of elements per blocks
    int elemPerBlocks = blockSize*blockSize;

    // block index
    int blockIndex = distIndex*elemPerBlocks;

    // image index of the last line of a block
    int imageIndexTo = imageIndex + (blockSize - 1)*imageDims.width;

    if (threadX < distMatrixDims.height && threadY < distMatrixDims.width)
    {
        float distance = srcDistMatrix[distIndex];

        // block is considered defectious, should copy it on 
        // destination iimage
        if (distance > alpha)
        {
            int i, j;

            for (i = imageIndex; i <= imageIndexTo; i = i + imageDims.width)
            {
                for (j = i; j < i + blockSize; ++j)
                {
                    unsigned char pixelValue = imageBlocks[blockIndex];
                    markedImage[j] = pixelValue;
                    ++blockIndex;
                }
            }
        }
    }
}

//***************************************************************************
// Wrapper for calling the kernel markImageDefectsP from standard source file
//***************************************************************************
cudaError_t markImageDefectsPWrapper(float alpha, float* srcDistMatrix, int blockSize, ROI imgBlockDims, unsigned char* imageBlocks, unsigned char* markedImage, ROI imageDims)
{
    cudaError_t ret;

    // data allocation on device
    float *devDistMatrix;
    size_t distMatrixSize = (size_t)(imgBlockDims.width*imgBlockDims.height*sizeof(float));

    unsigned char* devImageBlocks;
    unsigned char* devMarkedImage;
    size_t imageSize = (size_t)(imageDims.width*imageDims.height);

    // distance matrix
    ret = cudaMalloc(&devDistMatrix, distMatrixSize);
    if (ret != cudaSuccess)
    {
        return ret;
    }

    // image blocks
    ret = cudaMalloc(&devImageBlocks, imageSize);
    if (ret != cudaSuccess)
    {
        return ret;
    }

    // marked image buffer
    ret = cudaMalloc(&devMarkedImage, imageSize);
    if (ret != cudaSuccess)
    {
        return ret;
    }

    // data copy on device
    // distance matrix
    ret = cudaMemcpy(devDistMatrix, srcDistMatrix, distMatrixSize, cudaMemcpyHostToDevice);
    if (ret != cudaSuccess)
    {
        return ret;
    }

    // image blocks
    ret = cudaMemcpy(devImageBlocks, imageBlocks, imageSize, cudaMemcpyHostToDevice);
    if (ret != cudaSuccess)
    {
        return ret;
    }

    // matrix size in dimension, was in byte for malloc
    distMatrixSize /= sizeof(float);
    dim3 threadsDim(MARK_IMG_KERNEL_THREADS, MARK_IMG_KERNEL_THREADS, 1);
    markImageDefectsP << <1, threadsDim >> >(alpha, devDistMatrix, distMatrixSize, imgBlockDims, blockSize, devImageBlocks, devMarkedImage, imageDims);

    // data copy back to host
    ret = cudaMemcpy(markedImage, devMarkedImage, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(devDistMatrix);
    cudaFree(devImageBlocks);
    cudaFree(devMarkedImage);
    return ret;
}

__global__ void computeEuclidianDistanceP(float* refSign, float* signs, int numberOfsigns, float* distances, float* distanceMean)
{
    int signIndex = threadIdx.x;
    float distance = 0.0f;

    if (signIndex < numberOfsigns)
    {
        for (int i = 0; i < KIM_SIGN_SIZE; i++)
        {
            distance += fabsf(refSign[i] - signs[signIndex*KIM_SIGN_SIZE + i]);
        }
        distances[signIndex] = distance;
        *distanceMean += distance;
    }
}

//***********************************************************************************
// Wrapper for calling the kernel computeEuclidianDistanceP from standard source file
//***********************************************************************************
cudaError_t computeEuclidianDistancePWrapper(float* refSign, float* signs, int numberOfsigns, float* distances, float* distanceMean)
{
    cudaError_t ret = cudaSuccess;
    // allocate data on device
    float* devRefSign;
    float* devSigns;
    float* devDistances;
    float* devDistanceMean;

    size_t refSignSize = KIM_SIGN_SIZE*sizeof(float);
    size_t signsSize = KIM_SIGN_SIZE*numberOfsigns*sizeof(float);
    size_t distancesSize = numberOfsigns*sizeof(float);

    ret = cudaMalloc(&devRefSign, refSignSize);
    if (ret != cudaSuccess)
        return ret;

    ret = cudaMalloc(&devSigns, signsSize);
    if (ret != cudaSuccess)
        return ret;

    ret = cudaMalloc(&devDistances, distancesSize);
    if (ret != cudaSuccess)
        return ret;

    ret = cudaMalloc(&devDistanceMean, sizeof(float));
    if (ret != cudaSuccess)
        return ret;

    // copy refSignature and Signatures to device
    ret = cudaMemcpy(devRefSign, refSign, refSignSize, cudaMemcpyHostToDevice);
    if (ret != cudaSuccess)
        return ret;

    ret = cudaMemcpy(devSigns, signs, signsSize, cudaMemcpyHostToDevice);
    if (ret != cudaSuccess)
        return ret;

    // run kernel
    computeEuclidianDistanceP << <1, CALC_DISTANCE_KERNEL_THREADS >> >(devRefSign, devSigns, numberOfsigns, devDistances, devDistanceMean);

    // copy data back on host
    ret = cudaMemcpy(distances, devDistances, distancesSize, cudaMemcpyDeviceToHost);
    if (ret != cudaSuccess)
        return ret;

    ret = cudaMemcpy(distanceMean, devDistanceMean, sizeof(float), cudaMemcpyDeviceToHost);
    if (ret != cudaSuccess)
        return ret;

    // deallocate data on device
    cudaFree(devDistances);
    cudaFree(devRefSign);
    cudaFree(devSigns);
    cudaFree(devDistanceMean);
    return ret;
}

cudaError_t detectionWrapperP(unsigned char* image, ROI imageDims, int blockSize, ROI imageBlocksDims, const vector<float>& refSign, float eta, unsigned char* markedImage)
{
    cudaError_t ret = cudaSuccess;
    int imageSize = imageDims.width*imageDims.height;
    int imageblocksCount = imageBlocksDims.width*imageBlocksDims.height;
    
    // 1. call cut image kernel
    // image alloc
    unsigned char* devImage;
    size_t imageSizeMem = (size_t)imageSize;
    ret = cudaMalloc(&devImage, imageSizeMem);
    ret = cudaMemcpy(devImage, image, imageSizeMem, cudaMemcpyHostToDevice);

    // image blocks alloc
    unsigned char* devImageBlocks;
    ret = cudaMalloc(&devImageBlocks, imageSizeMem);

    // run kernel
    dim3 threadsDim(CUT_IMG_KERNEL_THREADS, CUT_IMG_KERNEL_THREADS, 1);
    cutImageP << <1, threadsDim >> >(devImage, imageDims, blockSize, devImageBlocks);

    // 2. kim signature phase 1
    // mean matrixes
    float* devMeanMatrixes;
    size_t meanMatrixesSizeMem = imageblocksCount*MEAN_MATRIX_SIZE*sizeof(float);
    ret = cudaMalloc(&devMeanMatrixes, meanMatrixesSizeMem);

    // run kernel
    threadsDim.x = PH1_KERNEL_THREADS;
    threadsDim.y = PH1_KERNEL_THREADS;
    extractKimSignaturePhase1 << <1, threadsDim >> >(devImageBlocks, imageBlocksDims, blockSize, devMeanMatrixes);

    // 3. kim signature phase 2 (DCT on mean matrix)
    // dct matrixes alloc
    float *devDctMeanMatrixes;
    ret = cudaMalloc(&devDctMeanMatrixes, meanMatrixesSizeMem);

    // run kernel
    computeDCT2 << <1, DCT_KERNEL_THREADS >> >(devMeanMatrixes, devDctMeanMatrixes, imageblocksCount);

    // 4. kim signature phase 3
    // kim signatures
    float* devKimSignatures;
    size_t kimSignatureSizeMem = imageblocksCount*KIM_SIGN_SIZE*sizeof(float);
    ret = cudaMalloc(&devKimSignatures, kimSignatureSizeMem);

    extractKimSignaturePhase3<< <1, PH3_KERNEL_THREADS >> >(devDctMeanMatrixes, imageblocksCount, devKimSignatures);

    // 5. distance euclidian distance compute
    // ref signature change from vector -> float*
    float pRefSign[KIM_SIGN_SIZE];
    float distanceMean = 0.0;
    std::copy(refSign.begin(), refSign.end(), pRefSign);

    // allocate on device
    float* devRefSign;
    size_t refSignSizeMem = KIM_SIGN_SIZE*sizeof(float);
    ret = cudaMalloc(&devRefSign, refSignSizeMem);
    ret = cudaMemcpy(devRefSign, pRefSign, refSignSizeMem, cudaMemcpyHostToDevice);

    // distances matrixe
    float* hostDistances = new float[imageblocksCount];
    float* devDistanceMatrixes;
    size_t distancesSizeMem = imageblocksCount*sizeof(float);
    ret = cudaMalloc(&devDistanceMatrixes, distancesSizeMem);
    
    // mean distance holder
    float* devDistanceMean;
    ret = cudaMalloc(&devDistanceMean, sizeof(float));

    computeEuclidianDistanceP << <1, CALC_DISTANCE_KERNEL_THREADS >> >(devRefSign, devKimSignatures, imageblocksCount, devDistanceMatrixes, devDistanceMean);

    // copy back some needed data on host
    ret = cudaMemcpy(hostDistances, devDistanceMatrixes, distancesSizeMem, cudaMemcpyDeviceToHost);
    if (ret != cudaSuccess) return ret;

    ret = cudaMemcpy(&distanceMean, devDistanceMean, sizeof(float), cudaMemcpyDeviceToHost);
    if (ret != cudaSuccess) return ret;

    // 6. compute alpha treshold
    distanceMean = distanceMean / (float)imageblocksCount;
    vector<float> distanceVec(imageblocksCount);
    std::copy(hostDistances, hostDistances + imageblocksCount, distanceVec.begin());

    float alpha = distanceMean + (eta*iqr<float>(distanceVec));

    // 7. mark defects on image
    unsigned char* devMarkedImage;
    size_t markedImageSizeMem = (size_t)imageSize;
    ret = cudaMalloc(&devMarkedImage, markedImageSizeMem);

    int distMatrixSize = distancesSizeMem /sizeof(float);
    threadsDim.x = MARK_IMG_KERNEL_THREADS;
    threadsDim.y = MARK_IMG_KERNEL_THREADS;
    markImageDefectsP << < 1, threadsDim >> >(alpha, devDistanceMatrixes, distMatrixSize, imageBlocksDims, blockSize, devImageBlocks, devMarkedImage, imageDims);

    // copy final data back to host
    ret = cudaMemcpy(markedImage, devMarkedImage, imageSize, cudaMemcpyDeviceToHost);

    // and clean all device memory allocated
    cudaFree(devImage);
    cudaFree(devImageBlocks);
    cudaFree(devMeanMatrixes);
    cudaFree(devDctMeanMatrixes);
    cudaFree(devKimSignatures);
    cudaFree(devRefSign);
    cudaFree(devDistanceMatrixes);
    cudaFree(devDistanceMean);
    cudaFree(devMarkedImage);

    //delete[] pRefSign;
    delete[] hostDistances;
    return ret;
}