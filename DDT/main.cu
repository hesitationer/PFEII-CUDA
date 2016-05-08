#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include "utilities.h"
#include "kim.h"
#include "online_detection.h"
#include "detection.cuh"
#include "cuda_profiler_api.h"
#include "Windows.h"

using namespace cv;
using namespace std;
using namespace utilities;

#define IMG_SIZE 98304

int main(int argc, char** argv)
{
    LARGE_INTEGER frequency;
    LARGE_INTEGER tBegin, tEnd, tProcessBegin, tProcessEnd;
    double elapsedTime;
    QueryPerformanceFrequency(&frequency);

    const int blockSize = 32;
    const float eta     = 5.0;
    Cell<const Mat> aMatrixCell(8, 12);

    int ret = cutImage(aMatrixCell, "E:/jee/cours/GEN5023/code/textile_images/Free/1.TIF", blockSize, false, false);
    
    vector<float> refSignature;
    ret = extractKimSignature(aMatrixCell.get(0, 0), blockSize, blockSize, refSignature);

    // buffer holding marked image
    unsigned char markedImg[IMG_SIZE];

    QueryPerformanceCounter(&tProcessBegin);
    
    cudaProfilerStart();
    for (int i = 1; i <= 26; ++i)
    {
        QueryPerformanceCounter(&tBegin);
        ostringstream stringStream;
        stringStream << "E:/jee/cours/GEN5023/code/textile_images/Defect/" << i << ".TIF";

        Mat img = imread(stringStream.str(), IMREAD_GRAYSCALE);
        ROI imgDims;
        imgDims.height = img.rows;
        imgDims.width  = img.cols;

        ROI blockDims;
        blockDims.height = (int)(img.rows / blockSize);
        blockDims.width  = (int)(img.cols / blockSize); 

        ret = detectionWrapperP(img.data, imgDims, blockSize, blockDims, refSignature, eta, markedImg);

        QueryPerformanceCounter(&tEnd);
        elapsedTime = (tEnd.QuadPart - tBegin.QuadPart) * 1000.0 / frequency.QuadPart;
        printf("Elapsed time for image %d : %f\n", i, elapsedTime);
        /*
        ostringstream anotherStream;
        anotherStream << "E:/jee/cours/GEN5023/code/textile_images/results_cuda/img-" << i << "-defect.tiff";

        Mat markedImgMat(imgDims.height, imgDims.width, CV_8U, markedImg);
        bool success = imwrite(anotherStream.str(), markedImgMat);
        cout << anotherStream.str() << endl;
        printf("Success for image [%d] : %d \n", i, (int)success);
        */
    }
    //QueryPerformanceCounter(&tProcessEnd);
    //elapsedTime = (tProcessEnd.QuadPart - tProcessBegin.QuadPart) * 1000.0 / frequency.QuadPart;
    cudaProfilerStop();
    return 0;
}