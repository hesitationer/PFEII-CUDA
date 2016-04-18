#include "cuda_utils.h"
#include <cmath>
#include <stdlib.h>

/**
**************************************************************************
*  Memory allocator, returns aligned format frame with 32bpp float pixels.
*
* \param width          [IN] - Width of image buffer to be allocated
* \param height         [IN] - Height of image buffer to be allocated
* \param pStepBytes     [OUT] - Step between two sequential rows
*
* \return Pointer to the created plane
*/
float* MallocPlaneFloat(int width, int height, int *pStepBytes)
{
    float *ptr;
    *pStepBytes = ((int)ceil((width*sizeof(float)) / 16.0f)) * 16;
        
    ptr = (float*)malloc(*pStepBytes * height);
    *pStepBytes = *pStepBytes / sizeof(float);
    return ptr;
}

/**
**************************************************************************
*  Memory allocator, returns aligned format frame with 8bpp pixels.
*
* \param width          [IN] - Width of image buffer to be allocated
* \param height         [IN] - Height of image buffer to be allocated
* \param pStepBytes     [OUT] - Step between two sequential rows
*
* \return Pointer to the created plane
*/
unsigned char* MallocPlaneByte(int width, int height, int* pStepBytes)
{
    unsigned char* ptr;
    *pStepBytes = ((int)ceil(width / 16.0f)) * 16;
    ptr = (unsigned char*)malloc(*pStepBytes * height);
    return ptr;
}

/**
**************************************************************************
*  Copies byte plane to float plane
*
* \param ImgSrc             [IN] - Source byte plane
* \param StrideB            [IN] - Source plane stride
* \param ImgDst             [OUT] - Destination float plane
* \param StrideF            [IN] - Destination plane stride
* \param Size               [IN] - Size of area to copy
*
* \return None
*/
void CopyByte2Float(unsigned char* ImgSrc, int StrideB, float *ImgDst, int StrideF, ROI Size)
{
    for (int i = 0; i<Size.height; i++)
    {
        for (int j = 0; j<Size.width; j++)
        {
            float imgSrcValue = (float)ImgSrc[i*StrideB + j];
            ImgDst[i*StrideF + j] = imgSrcValue;
        }
    }
}

/**
**************************************************************************
*  Memory deallocator, deletes aligned format frame.
*
* \param ptr            [IN] - Pointer to the plane
*
* \return None
*/
void FreePlane(void *ptr)
{
    if (ptr)
    {
        free(ptr);
    }
}

/**
**************************************************************************
*  Performs addition of given value to each pixel in the plane
*
* \param Value              [IN] - Value to add
* \param ImgSrcDst          [IN/OUT] - Source float plane
* \param StrideF            [IN] - Source plane stride
* \param Size               [IN] - Size of area to copy
*
* \return None
*/
void AddFloatPlane(float Value, float *ImgSrcDst, int StrideF, ROI Size)
{
    for (int i = 0; i<Size.height; i++)
    {
        for (int j = 0; j<Size.width; j++)
        {
            ImgSrcDst[i*StrideF + j] += Value;
        }
    }
}

/**
**************************************************************************
*  Performs multiplication of given value with each pixel in the plane
*
* \param Value              [IN] - Value for multiplication
* \param ImgSrcDst          [IN/OUT] - Source float plane
* \param StrideF            [IN] - Source plane stride
* \param Size               [IN] - Size of area to copy
*
* \return None
*/
void MulFloatPlane(float Value, float *ImgSrcDst, int StrideF, ROI Size)
{
    for (int i = 0; i<Size.height; i++)
    {
        for (int j = 0; j<Size.width; j++)
        {
            ImgSrcDst[i*StrideF + j] *= Value;
        }
    }
}

/**
**************************************************************************
*  Copies float plane to byte plane (with clamp)
*
* \param ImgSrc             [IN] - Source float plane
* \param StrideF            [IN] - Source plane stride
* \param ImgDst             [OUT] - Destination byte plane
* \param StrideB            [IN] - Destination plane stride
* \param Size               [IN] - Size of area to copy
*
* \return None
*/
void CopyFloat2Byte(float *ImgSrc, int StrideF, unsigned char* ImgDst, int StrideB, ROI Size)
{
    for (int i = 0; i<Size.height; i++)
    {
        for (int j = 0; j<Size.width; j++)
        {
            ImgDst[i*StrideB + j] = (unsigned char)clamp_0_255((int)(round_f(ImgSrc[i*StrideF + j])));
        }
    }
}

/**
**************************************************************************
*  The routine clamps the input value to integer byte range [0, 255]
*
* \param x          [IN] - Input value
*
* \return Pointer to the created plane
*/
int clamp_0_255(int x)
{
    return (x < 0) ? 0 : ((x > 255) ? 255 : x);
}

/**
**************************************************************************
*  Float round to nearest value
*
* \param num            [IN] - Float value to round
*
* \return The closest to the input float integer value
*/
float round_f(float num)
{
    float NumAbs = fabs(num);
    int NumAbsI = (int)(NumAbs + 0.5f);
    float sign = num > 0 ? 1.0f : -1.0f;
    return sign * NumAbsI;
}