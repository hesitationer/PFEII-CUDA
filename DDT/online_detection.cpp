//--------------------------------------------------
// implementation file of online detection functions
//--------------------------------------------------

//----------------------------
// INCLUDES
//----------------------------
#include "online_detection.h"
#include "kim.h"
#include "detection.cuh"
#include "kim.cuh"
#include "dct8x8.cuh"
#include "thrust/device_vector.h"
//----------------------------
// NAMESPACES
//----------------------------
using namespace cv;
using namespace std;
using namespace utilities;

//----------------------------
// IMPLEMENTATION
//----------------------------

//****************************
// See header file for details
//****************************
cudaError_t onlineDetection(unsigned char* imageBlocks, const ROI& imageBlocksDims, int blockSize, vector<float>& refSign, float eta, bool overlap, unsigned char* markedImage, const ROI& markedImageDims)
{
    // overlap not implemented yet
    (void)overlap;
    cudaError_t ret = cudaSuccess;

    int numberOfBlocks = imageBlocksDims.width*imageBlocksDims.height;
    
    // phase 1 of kim signature : extract from each image blocks
    // a 8x8 mean matrix
    float* meanMatrixes = new float[numberOfBlocks * 64];
    ret = extractKimSignaturePh1(imageBlocks, imageBlocksDims, blockSize, meanMatrixes);

    // phase 2 of kim signature : run 2D-DCT of mean matrixes
    float* dctMeanMatrixes = new float[numberOfBlocks * 64];
    ret = wrapperDCT2(meanMatrixes, dctMeanMatrixes, numberOfBlocks);

    // phase 3 of kim signature : compute kim signatures from DCT matrixes
    // allocate a vector on device for all the dct matrixes
    float* kimSignatures = new float[numberOfBlocks*KIM_SIGN_SIZE];
    ret = extractKimSignaturePh3(dctMeanMatrixes, numberOfBlocks, kimSignatures);

    // compute distance between each signature
    float* distances = new float[numberOfBlocks];
    float pRefSign[KIM_SIGN_SIZE];
    float distanceMean = 0.0;
    std::copy(refSign.begin(), refSign.end(), pRefSign);

    ret = computeEuclidianDistancePWrapper(pRefSign, kimSignatures, numberOfBlocks, distances, &distanceMean);
    
    distanceMean = distanceMean / (float)numberOfBlocks;
    vector<float> distanceVec(numberOfBlocks);
    std::copy(distances, distances+numberOfBlocks, distanceVec.begin());
    
    float alpha = distanceMean + (eta*iqr<float>(distanceVec));

    // mark defectious blocks
    markImageDefectsPWrapper(alpha, distances, blockSize, imageBlocksDims, imageBlocks, markedImage, markedImageDims);
    
    delete[] distances;
    delete[] kimSignatures;
    delete[] meanMatrixes;
    delete[] dctMeanMatrixes;
    
    return ret;
}