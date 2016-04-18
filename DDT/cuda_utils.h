#ifndef H_CUDA_UTILS
#define H_CUDA_UTILS
//*****************************************
// CUDA image utilities took in dct8x8
// example
//*****************************************

//*****************************************
// Structure holding image width and height
//*****************************************
typedef struct
{
    int width;          //!< ROI width
    int height;         //!< ROI height
} ROI;

float* MallocPlaneFloat(int width, int height, int *pStepBytes);

unsigned char* MallocPlaneByte(int width, int height, int *pStepBytes);

void CopyByte2Float(unsigned char* ImgSrc, int StrideB, float *ImgDst, int StrideF, ROI Size);

void FreePlane(void *ptr);

void AddFloatPlane(float Value, float *ImgSrcDst, int StrideF, ROI Size);

void MulFloatPlane(float Value, float *ImgSrcDst, int StrideF, ROI Size);

void CopyFloat2Byte(float *ImgSrc, int StrideF, unsigned char* ImgDst, int StrideB, ROI Size);

int clamp_0_255(int x);

float round_f(float num);

#endif // H_CUDA_UTILS