#include "dct8x8.cuh"

#include "common_functions.h"
#include "device_launch_parameters.h"
#include <opencv2/core/core.hpp>

/**
*  Texture reference that is passed through this global variable into device code.
*  This is done because any conventional passing through argument list way results
*  in compiler internal error. 2008.03.11
*/
texture<float, 2, cudaReadModeElementType> TexSrc;

/**
*  This unitary matrix performs discrete cosine transform of rows of the matrix to the left
*/
__constant__ float DCTv8matrix[] =
{
    0.3535533905932738f, 0.4903926402016152f, 0.4619397662556434f, 0.4157348061512726f, 0.3535533905932738f, 0.2777851165098011f, 0.1913417161825449f, 0.0975451610080642f,
    0.3535533905932738f, 0.4157348061512726f, 0.1913417161825449f, -0.0975451610080641f, -0.3535533905932737f, -0.4903926402016152f, -0.4619397662556434f, -0.2777851165098011f,
    0.3535533905932738f, 0.2777851165098011f, -0.1913417161825449f, -0.4903926402016152f, -0.3535533905932738f, 0.0975451610080642f, 0.4619397662556433f, 0.4157348061512727f,
    0.3535533905932738f, 0.0975451610080642f, -0.4619397662556434f, -0.2777851165098011f, 0.3535533905932737f, 0.4157348061512727f, -0.1913417161825450f, -0.4903926402016153f,
    0.3535533905932738f, -0.0975451610080641f, -0.4619397662556434f, 0.2777851165098009f, 0.3535533905932738f, -0.4157348061512726f, -0.1913417161825453f, 0.4903926402016152f,
    0.3535533905932738f, -0.2777851165098010f, -0.1913417161825452f, 0.4903926402016153f, -0.3535533905932733f, -0.0975451610080649f, 0.4619397662556437f, -0.4157348061512720f,
    0.3535533905932738f, -0.4157348061512727f, 0.1913417161825450f, 0.0975451610080640f, -0.3535533905932736f, 0.4903926402016152f, -0.4619397662556435f, 0.2777851165098022f,
    0.3535533905932738f, -0.4903926402016152f, 0.4619397662556433f, -0.4157348061512721f, 0.3535533905932733f, -0.2777851165098008f, 0.1913417161825431f, -0.0975451610080625f
};


// Temporary blocks
__shared__ float CurBlockLocal1[BLOCK_SIZE2];
__shared__ float CurBlockLocal2[BLOCK_SIZE2];

/**
*  JPEG quality=0_of_12 quantization matrix
*/
__constant__ short Q[] =
{
    32, 33, 51, 81, 66, 39, 34, 17,
    33, 36, 48, 47, 28, 23, 12, 12,
    51, 48, 47, 28, 23, 12, 12, 12,
    81, 47, 28, 23, 12, 12, 12, 12,
    66, 28, 23, 12, 12, 12, 12, 12,
    39, 23, 12, 12, 12, 12, 12, 12,
    34, 12, 12, 12, 12, 12, 12, 12,
    17, 12, 12, 12, 12, 12, 12, 12
};

/**
**************************************************************************
*  Performs in-place quantization of given DCT coefficients plane using
*  predefined quantization matrices (for floats plane). Unoptimized.
*
* \param SrcDst         [IN/OUT] - DCT coefficients plane
* \param Stride         [IN] - Stride of SrcDst
*
* \return None
*/
__global__ void CUDAkernelQuantizationFloat(float *SrcDst, int Stride)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index (current coefficient)
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //copy current coefficient to the local variable
    float curCoef = SrcDst[(by * BLOCK_SIZE + ty) * Stride + (bx * BLOCK_SIZE + tx)];
    float curQuant = (float)Q[ty * BLOCK_SIZE + tx];

    //quantize the current coefficient
    float quantized = round(curCoef / curQuant);
    curCoef = quantized * curQuant;

    //copy quantized coefficient back to the DCT-plane
    SrcDst[(by * BLOCK_SIZE + ty) * Stride + (bx * BLOCK_SIZE + tx)] = curCoef;
}

/**
**************************************************************************
*  Performs 1st implementation of 8x8 block-wise Forward Discrete Cosine Transform of the given
*  image plane and outputs result to the array of coefficients.
*
* \param Dst            [OUT] - Coefficients plane
* \param ImgWidth       [IN] - Stride of Dst
* \param OffsetXBlocks  [IN] - Offset along X in blocks from which to perform processing
* \param OffsetYBlocks  [IN] - Offset along Y in blocks from which to perform processing
*
* \return None
*/
__global__ void CUDAkernel1DCT(float *Dst, int ImgWidth, int OffsetXBlocks, int OffsetYBlocks)
{
    // Block index
    const int bx = blockIdx.x + OffsetXBlocks;
    const int by = blockIdx.y + OffsetYBlocks;

    // Thread index (current coefficient)
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Texture coordinates
    const float tex_x = (float)((bx << BLOCK_SIZE_LOG2) + tx) + 0.5f;
    const float tex_y = (float)((by << BLOCK_SIZE_LOG2) + ty) + 0.5f;

    //copy current image pixel to the first block
    CurBlockLocal1[(ty << BLOCK_SIZE_LOG2) + tx] = tex2D(TexSrc, tex_x, tex_y);

    //synchronize threads to make sure the block is copied
    __syncthreads();

    //calculate the multiplication of DCTv8matrixT * A and place it in the second block
    float curelem = 0;
    int DCTv8matrixIndex = 0 * BLOCK_SIZE + ty;
    int CurBlockLocal1Index = 0 * BLOCK_SIZE + tx;
#pragma unroll

    for (int i = 0; i<BLOCK_SIZE; i++)
    {
        curelem += DCTv8matrix[DCTv8matrixIndex] * CurBlockLocal1[CurBlockLocal1Index];
        DCTv8matrixIndex += BLOCK_SIZE;
        CurBlockLocal1Index += BLOCK_SIZE;
    }

    CurBlockLocal2[(ty << BLOCK_SIZE_LOG2) + tx] = curelem;

    //synchronize threads to make sure the first 2 matrices are multiplied and the result is stored in the second block
    __syncthreads();

    //calculate the multiplication of (DCTv8matrixT * A) * DCTv8matrix and place it in the first block
    curelem = 0;
    int CurBlockLocal2Index = (ty << BLOCK_SIZE_LOG2) + 0;
    DCTv8matrixIndex = 0 * BLOCK_SIZE + tx;
#pragma unroll

    for (int i = 0; i<BLOCK_SIZE; i++)
    {
        curelem += CurBlockLocal2[CurBlockLocal2Index] * DCTv8matrix[DCTv8matrixIndex];
        CurBlockLocal2Index += 1;
        DCTv8matrixIndex += BLOCK_SIZE;
    }

    CurBlockLocal1[(ty << BLOCK_SIZE_LOG2) + tx] = curelem;

    //synchronize threads to make sure the matrices are multiplied and the result is stored back in the first block
    __syncthreads();

    //copy current coefficient to its place in the result array
    Dst[FMUL(((by << BLOCK_SIZE_LOG2) + ty), ImgWidth) + ((bx << BLOCK_SIZE_LOG2) + tx)] = CurBlockLocal1[(ty << BLOCK_SIZE_LOG2) + tx];
}


/**
**************************************************************************
*  Wrapper function for 1st CUDA version of DCT, quantization and IDCT implementations
*
* \param ImgSrc         [IN] - Source byte image plane
* \param ImgDst         [IN] - Quantized result byte image plane
* \param Stride         [IN] - Stride for both source and result planes
* \param Size           [IN] - Size of both planes
*
* \return Execution time in milliseconds
*/
void WrapperCUDA1(unsigned char* ImgSrc, unsigned char* ImgDst, int Stride, ROI Size)
{
    //prepare channel format descriptor for passing texture into kernels
    cudaChannelFormatDesc floattex = cudaCreateChannelDesc<float>();

    //allocate device memory
    cudaArray *Src;
    float *Dst;
    size_t DstStride;
    checkCudaErrors(cudaMallocArray(&Src, &floattex, Size.width, Size.height));

    checkCudaErrors(cudaMallocPitch((void **)(&Dst), &DstStride, Size.width * sizeof(float), Size.height));

    DstStride /= sizeof(float);

    //convert source image to float representation
    int ImgSrcFStride;
    float *ImgSrcF = MallocPlaneFloat(Size.width, Size.height, &ImgSrcFStride);
    CopyByte2Float(ImgSrc, Stride, ImgSrcF, ImgSrcFStride, Size);
    AddFloatPlane(-128.0f, ImgSrcF, ImgSrcFStride, Size);

    //copy from host memory to device
    cudaError_t ret = cudaMemcpy2DToArray(Src, 0, 0, ImgSrcF, ImgSrcFStride * sizeof(float), Size.width * sizeof(float), Size.height, cudaMemcpyHostToDevice);

    if (ret != cudaSuccess)
    {
        printf("Error ! : %s", cudaGetErrorName(ret));
        exit(EXIT_FAILURE);
    }

    //setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(Size.width / BLOCK_SIZE, Size.height / BLOCK_SIZE);

    //execute DCT kernel
    ret = cudaBindTextureToArray(TexSrc, Src);

    if (ret != cudaSuccess)
    {
        printf("Error ! : %s", cudaGetErrorName(ret));
        exit(EXIT_FAILURE);
    }

    CUDAkernel1DCT<<<grid, threads>>>(Dst, (int)DstStride, 0, 0);
    ret = cudaDeviceSynchronize();

    if (ret != cudaSuccess)
    {
        printf("Error ! : %s", cudaGetErrorName(ret));
        exit(EXIT_FAILURE);
    }

    ret = cudaUnbindTexture(TexSrc);
    if (ret != cudaSuccess)
    {
        printf("Error ! : %s", cudaGetErrorName(ret));
        exit(EXIT_FAILURE);
    }

    getLastCudaError("Kernel execution failed");

    // execute Quantization kernel
    //CUDAkernelQuantizationFloat <<<grid, threads>>>(Dst, (int)DstStride);
    //getLastCudaError("Kernel execution failed");

    //copy quantized coefficients from host memory to device array
    ret = cudaMemcpy2DToArray(Src, 0, 0, Dst, DstStride *sizeof(float), Size.width *sizeof(float), Size.height, cudaMemcpyDeviceToDevice);
    if (ret != cudaSuccess)
    {
        printf("Error ! : %s", cudaGetErrorName(ret));
        exit(EXIT_FAILURE);
    }

    /**
    // execute IDCT kernel
    checkCudaErrors(cudaBindTextureToArray(TexSrc, Src));
    CUDAkernel1IDCT << < grid, threads >> >(Dst, (int)DstStride, 0, 0);
    checkCudaErrors(cudaUnbindTexture(TexSrc));
    getLastCudaError("Kernel execution failed");
    /**/

    //copy quantized image block to host
    ret = cudaMemcpy2D(ImgSrcF, ImgSrcFStride *sizeof(float), Dst, DstStride *sizeof(float), Size.width *sizeof(float), Size.height, cudaMemcpyDeviceToHost);
    if (ret != cudaSuccess)
    {
        printf("Error ! : %s", cudaGetErrorName(ret));
        exit(EXIT_FAILURE);
    }

    //convert image back to byte representation
    AddFloatPlane(128.0f, ImgSrcF, ImgSrcFStride, Size);
    //CopyFloat2Byte(ImgSrcF, ImgSrcFStride, ImgDst, Stride, Size);

    //clean up memory
    checkCudaErrors(cudaFreeArray(Src));
    checkCudaErrors(cudaFree(Dst));
    FreePlane(ImgSrcF);
}

/**
**************************************************************************
*  Performs in-place DCT of vector of 8 elements.
*
* \param Vect0          [IN/OUT] - Pointer to the first element of vector
* \param Step           [IN/OUT] - Value to add to ptr to access other elements
*
* \return None
*/
__device__ void CUDAsubroutineInplaceDCTvector(float *Vect0, int Step)
{
    float *Vect1 = Vect0 + Step;
    float *Vect2 = Vect1 + Step;
    float *Vect3 = Vect2 + Step;
    float *Vect4 = Vect3 + Step;
    float *Vect5 = Vect4 + Step;
    float *Vect6 = Vect5 + Step;
    float *Vect7 = Vect6 + Step;

    float X07P = (*Vect0) + (*Vect7);
    float X16P = (*Vect1) + (*Vect6);
    float X25P = (*Vect2) + (*Vect5);
    float X34P = (*Vect3) + (*Vect4);

    float X07M = (*Vect0) - (*Vect7);
    float X61M = (*Vect6) - (*Vect1);
    float X25M = (*Vect2) - (*Vect5);
    float X43M = (*Vect4) - (*Vect3);

    float X07P34PP = X07P + X34P;
    float X07P34PM = X07P - X34P;
    float X16P25PP = X16P + X25P;
    float X16P25PM = X16P - X25P;

    (*Vect0) = C_norm * (X07P34PP + X16P25PP);
    (*Vect2) = C_norm * (C_b * X07P34PM + C_e * X16P25PM);
    (*Vect4) = C_norm * (X07P34PP - X16P25PP);
    (*Vect6) = C_norm * (C_e * X07P34PM - C_b * X16P25PM);

    (*Vect1) = C_norm * (C_a * X07M - C_c * X61M + C_d * X25M - C_f * X43M);
    (*Vect3) = C_norm * (C_c * X07M + C_f * X61M - C_a * X25M + C_d * X43M);
    (*Vect5) = C_norm * (C_d * X07M + C_a * X61M + C_f * X25M - C_c * X43M);
    (*Vect7) = C_norm * (C_f * X07M + C_d * X61M + C_c * X25M + C_a * X43M);
}

/**
**************************************************************************
*  Performs in-place IDCT of vector of 8 elements.
*
* \param Vect0          [IN/OUT] - Pointer to the first element of vector
* \param Step           [IN/OUT] - Value to add to ptr to access other elements
*
* \return None
*/
__device__ void CUDAsubroutineInplaceIDCTvector(float *Vect0, int Step)
{
    float *Vect1 = Vect0 + Step;
    float *Vect2 = Vect1 + Step;
    float *Vect3 = Vect2 + Step;
    float *Vect4 = Vect3 + Step;
    float *Vect5 = Vect4 + Step;
    float *Vect6 = Vect5 + Step;
    float *Vect7 = Vect6 + Step;

    float Y04P = (*Vect0) + (*Vect4);
    float Y2b6eP = C_b * (*Vect2) + C_e * (*Vect6);

    float Y04P2b6ePP = Y04P + Y2b6eP;
    float Y04P2b6ePM = Y04P - Y2b6eP;
    float Y7f1aP3c5dPP = C_f * (*Vect7) + C_a * (*Vect1) + C_c * (*Vect3) + C_d * (*Vect5);
    float Y7a1fM3d5cMP = C_a * (*Vect7) - C_f * (*Vect1) + C_d * (*Vect3) - C_c * (*Vect5);

    float Y04M = (*Vect0) - (*Vect4);
    float Y2e6bM = C_e * (*Vect2) - C_b * (*Vect6);

    float Y04M2e6bMP = Y04M + Y2e6bM;
    float Y04M2e6bMM = Y04M - Y2e6bM;
    float Y1c7dM3f5aPM = C_c * (*Vect1) - C_d * (*Vect7) - C_f * (*Vect3) - C_a * (*Vect5);
    float Y1d7cP3a5fMM = C_d * (*Vect1) + C_c * (*Vect7) - C_a * (*Vect3) + C_f * (*Vect5);

    (*Vect0) = C_norm * (Y04P2b6ePP + Y7f1aP3c5dPP);
    (*Vect7) = C_norm * (Y04P2b6ePP - Y7f1aP3c5dPP);
    (*Vect4) = C_norm * (Y04P2b6ePM + Y7a1fM3d5cMP);
    (*Vect3) = C_norm * (Y04P2b6ePM - Y7a1fM3d5cMP);

    (*Vect1) = C_norm * (Y04M2e6bMP + Y1c7dM3f5aPM);
    (*Vect5) = C_norm * (Y04M2e6bMM - Y1d7cP3a5fMM);
    (*Vect2) = C_norm * (Y04M2e6bMM + Y1d7cP3a5fMM);
    (*Vect6) = C_norm * (Y04M2e6bMP - Y1c7dM3f5aPM);
}

/**
**************************************************************************
*  Performs 8x8 block-wise Forward Discrete Cosine Transform of the given
*  image plane and outputs result to the array of coefficients. 2nd implementation.
*  This kernel is designed to process image by blocks of blocks8x8 that
*  utilizes maximum warps capacity, assuming that it is enough of 8 threads
*  per block8x8.
*
* \param SrcDst                     [OUT] - Coefficients plane
* \param ImgStride                  [IN] - Stride of SrcDst
*
* \return None
*/
__global__ void CUDAkernel2DCT(float *dst, float *src, int ImgStride)
{
    __shared__ float block[KER2_BLOCK_HEIGHT * KER2_SMEMBLOCK_STRIDE];

    int OffsThreadInRow = threadIdx.y * BLOCK_SIZE + threadIdx.x;
    int OffsThreadInCol = threadIdx.z * BLOCK_SIZE;
    src += FMUL(blockIdx.y * KER2_BLOCK_HEIGHT + OffsThreadInCol, ImgStride) + blockIdx.x * KER2_BLOCK_WIDTH + OffsThreadInRow;
    dst += FMUL(blockIdx.y * KER2_BLOCK_HEIGHT + OffsThreadInCol, ImgStride) + blockIdx.x * KER2_BLOCK_WIDTH + OffsThreadInRow;
    float *bl_ptr = block + OffsThreadInCol * KER2_SMEMBLOCK_STRIDE + OffsThreadInRow;

#pragma unroll

    for (unsigned int i = 0; i < BLOCK_SIZE; i++)
        bl_ptr[i * KER2_SMEMBLOCK_STRIDE] = src[i * ImgStride];

    //process rows
    CUDAsubroutineInplaceDCTvector(block + (OffsThreadInCol + threadIdx.x) * KER2_SMEMBLOCK_STRIDE + OffsThreadInRow - threadIdx.x, 1);

    //process columns
    CUDAsubroutineInplaceDCTvector(bl_ptr, KER2_SMEMBLOCK_STRIDE);

    for (unsigned int i = 0; i < BLOCK_SIZE; i++)
        dst[i * ImgStride] = bl_ptr[i * KER2_SMEMBLOCK_STRIDE];
}

/**
**************************************************************************
*  Performs 8x8 block-wise Inverse Discrete Cosine Transform of the given
*  coefficients plane and outputs result to the image. 2nd implementation.
*  This kernel is designed to process image by blocks of blocks8x8 that
*  utilizes maximum warps capacity, assuming that it is enough of 8 threads
*  per block8x8.
*
* \param SrcDst                     [OUT] - Coefficients plane
* \param ImgStride                  [IN] - Stride of SrcDst
*
* \return None
*/
__global__ void CUDAkernel2IDCT(float *dst, float *src, int ImgStride)
{
    __shared__ float block[KER2_BLOCK_HEIGHT * KER2_SMEMBLOCK_STRIDE];

    int OffsThreadInRow = threadIdx.y * BLOCK_SIZE + threadIdx.x;
    int OffsThreadInCol = threadIdx.z * BLOCK_SIZE;
    src += FMUL(blockIdx.y * KER2_BLOCK_HEIGHT + OffsThreadInCol, ImgStride) + blockIdx.x * KER2_BLOCK_WIDTH + OffsThreadInRow;
    dst += FMUL(blockIdx.y * KER2_BLOCK_HEIGHT + OffsThreadInCol, ImgStride) + blockIdx.x * KER2_BLOCK_WIDTH + OffsThreadInRow;
    float *bl_ptr = block + OffsThreadInCol * KER2_SMEMBLOCK_STRIDE + OffsThreadInRow;

#pragma unroll

    for (unsigned int i = 0; i < BLOCK_SIZE; i++)
        bl_ptr[i * KER2_SMEMBLOCK_STRIDE] = src[i * ImgStride];

    //process rows
    CUDAsubroutineInplaceIDCTvector(block + (OffsThreadInCol + threadIdx.x) * KER2_SMEMBLOCK_STRIDE + OffsThreadInRow - threadIdx.x, 1);

    //process columns
    CUDAsubroutineInplaceIDCTvector(bl_ptr, KER2_SMEMBLOCK_STRIDE);

    for (unsigned int i = 0; i < BLOCK_SIZE; i++)
        dst[i * ImgStride] = bl_ptr[i * KER2_SMEMBLOCK_STRIDE];
}

/**
**************************************************************************
*  Wrapper function for 2nd CUDA version of DCT, quantization and IDCT implementations
*
* \param ImgSrc         [IN] - Source byte image plane
* \param ImgDst         [IN] - Quantized result byte image plane
* \param Stride         [IN] - Stride for both source and result planes
* \param Size           [IN] - Size of both planes
*
* \return Execution time in milliseconds
*/
void WrapperCUDA2(unsigned char* ImgSrc, unsigned char* ImgDst, int Stride, ROI Size)
{
    //allocate host buffers for DCT and other data
    int StrideF;
    float *ImgF1 = MallocPlaneFloat(Size.width, Size.height, &StrideF);

    //convert source image to float representation
    CopyByte2Float(ImgSrc, Stride, ImgF1, StrideF, Size);
    cv::Mat ogImg(Size.height, Size.width, CV_8U, ImgSrc);
    cv::Mat flImg(Size.height, Size.width, CV_32F, ImgF1);

    AddFloatPlane(-128.0f, ImgF1, StrideF, Size);
    
    //allocate device memory
    float *src, *dst;
    size_t DeviceStride;
    checkCudaErrors(cudaMallocPitch((void **)&src, &DeviceStride, Size.width * sizeof(float), Size.height));
    checkCudaErrors(cudaMallocPitch((void **)&dst, &DeviceStride, Size.width * sizeof(float), Size.height));
    DeviceStride /= sizeof(float);

    //copy from host memory to device
    checkCudaErrors(cudaMemcpy2D(src, DeviceStride * sizeof(float), ImgF1, StrideF * sizeof(float), Size.width * sizeof(float), Size.height, cudaMemcpyHostToDevice));

    //setup execution parameters
    dim3 GridFullWarps(Size.width / KER2_BLOCK_WIDTH, Size.height / KER2_BLOCK_HEIGHT, 1);
    dim3 ThreadsFullWarps(8, KER2_BLOCK_WIDTH / 8, KER2_BLOCK_HEIGHT / 8);

    //perform block-wise DCT processing and benchmarking
    //const int numIterations = 100;

    //for (int i = -1; i < numIterations; i++)
    //{
    CUDAkernel2DCT<<<GridFullWarps, ThreadsFullWarps >>>(dst, src, (int)DeviceStride);
    getLastCudaError("Kernel execution failed");
    //}

    checkCudaErrors(cudaDeviceSynchronize());
    
    //setup execution parameters for quantization
    //dim3 ThreadsSmallBlocks(BLOCK_SIZE, BLOCK_SIZE);
    //dim3 GridSmallBlocks(Size.width / BLOCK_SIZE, Size.height / BLOCK_SIZE);

    // execute Quantization kernel
    //CUDAkernelQuantizationFloat <<<GridSmallBlocks, ThreadsSmallBlocks>>>(dst, (int)DeviceStride);
    //getLastCudaError("Kernel execution failed");

    //perform block-wise IDCT processing
    //CUDAkernel2IDCT<<<GridFullWarps, ThreadsFullWarps>>>(src, dst, (int)DeviceStride);
    //checkCudaErrors(cudaDeviceSynchronize());
    //getLastCudaError("Kernel execution failed");

    //copy quantized image block to host
    checkCudaErrors(cudaMemcpy2D(ImgF1, StrideF*sizeof(float), dst, DeviceStride*sizeof(float), Size.width *sizeof(float), Size.height, cudaMemcpyDeviceToHost));

    //convert image back to byte representation
    AddFloatPlane(128.0f, ImgF1, StrideF, Size);
    CopyFloat2Byte(ImgF1, StrideF, ImgDst, Stride, Size);
    cv::Mat anotherImg(Size.height, Size.width, CV_32F, ImgF1, StrideF*sizeof(float));

    //clean up memory
    checkCudaErrors(cudaFree(dst));
    checkCudaErrors(cudaFree(src));
    FreePlane(ImgF1);
}