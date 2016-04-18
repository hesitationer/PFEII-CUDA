#ifndef H_COMMON_DCT
#define H_COMMON_DCT
/*
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/**
**************************************************************************
* \file Common.h
* \brief Common includes header.
*
* This file contains includes of all libraries used by the project.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>       // helper functions for CUDA timing and initialization
#include <helper_functions.h>  // helper functions for timing, string parsing

/**
*  The dimension of pixels block
*/
#define BLOCK_SIZE          8


/**
*  Square of dimension of pixels block
*/
#define BLOCK_SIZE2         64


/**
*  log_2{BLOCK_SIZE), used for quick multiplication or division by the
*  pixels block dimension via shifting
*/
#define BLOCK_SIZE_LOG2     3


/**
*  log_2{BLOCK_SIZE*BLOCK_SIZE), used for quick multiplication or division by the
*  square of pixels block via shifting
*/
#define BLOCK_SIZE2_LOG2    6


/**
*  This macro states that __mul24 operation is performed faster that traditional
*  multiplication for two integers on CUDA. Please undefine if it appears to be
*  wrong on your system
*/
#define __MUL24_FASTER_THAN_ASTERIX


/**
*  Wrapper to the fastest integer multiplication function on CUDA
*/
#ifdef __MUL24_FASTER_THAN_ASTERIX
#define FMUL(x,y)   (__mul24(x,y))
#else
#define FMUL(x,y)   ((x)*(y))
#endif

#define C_a 1.387039845322148f //!< a = (2^0.5) * cos(    pi / 16);  Used in forward and inverse DCT.  
#define C_b 1.306562964876377f //!< b = (2^0.5) * cos(    pi /  8);  Used in forward and inverse DCT.  
#define C_c 1.175875602419359f //!< c = (2^0.5) * cos(3 * pi / 16);  Used in forward and inverse DCT.  
#define C_d 0.785694958387102f //!< d = (2^0.5) * cos(5 * pi / 16);  Used in forward and inverse DCT.  
#define C_e 0.541196100146197f //!< e = (2^0.5) * cos(3 * pi /  8);  Used in forward and inverse DCT.  
#define C_f 0.275899379282943f //!< f = (2^0.5) * cos(7 * pi / 16);  Used in forward and inverse DCT.  

/**
*  Normalization constant that is used in forward and inverse DCT
*/
#define C_norm 0.3535533905932737f // 1 / (8^0.5)

/**
*  Width of data block (2nd kernel)
*/
#define KER2_BLOCK_WIDTH          32

/**
*  Height of data block (2nd kernel)
*/
#define KER2_BLOCK_HEIGHT         16

/**
*  LOG2 of width of data block (2nd kernel)
*/
#define KER2_BW_LOG2              5

/**
*  LOG2 of height of data block (2nd kernel)
*/
#define KER2_BH_LOG2              4

/**
*  Stride of shared memory buffer (2nd kernel)
*/
#define KER2_SMEMBLOCK_STRIDE     (KER2_BLOCK_WIDTH+1)

#endif //H_COMMON_DCT