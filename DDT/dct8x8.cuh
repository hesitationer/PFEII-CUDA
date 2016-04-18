//**********************************************
// 8x8 DCT on Cuda types and declarations
//**********************************************
#include "cuda_utils.h"
#include "common.h"

//***********************************************
// CUDA Sample code for a 8x8 DCT computing
// allocate memory on device, to the cleaning and
// copy back on host
//***********************************************
// @param ImgSrc pointer to image content
// @param ImgDst pointer to dct coefs image
// @param Stride max length of a image row
// @param Size   image size as a ROI struct
void WrapperCUDA1(unsigned char* ImgSrc, unsigned char* ImgDst, int Stride, ROI Size);

void WrapperCUDA2(unsigned char* ImgSrc, unsigned char* ImgDst, int Stride, ROI Size);