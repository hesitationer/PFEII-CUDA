#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include "utilities.h"
#include "kim.h"
#include "online_detection.h"

#include "detection.cuh"

using namespace cv;
using namespace std;
using namespace utilities;

void printPortion(int* arr, int arrLen)
{
    for (int i = 0; i < arrLen; ++i)
    {
        printf("%d ", arr[i]);
    }
}

int main(int argc, char** argv)
{
    const int blockSize = 32;
    const float eta     = 5.0;
    Cell<const Mat> aMatrixCell(8, 12);

    int ret = cutImage(aMatrixCell, "E:/jee/cours/GEN5023/code/textile_images/Free/1.TIF", blockSize, false, false);
    
    vector<float> refSignature;
    ret = extractKimSignature(aMatrixCell.get(0, 0), blockSize, blockSize, refSignature);

    for (int i = 1; i <= 26; ++i)
    {
        ostringstream stringStream;
        stringStream << "E:/jee/cours/GEN5023/code/textile_images/Defect/" << i << ".TIF";

        Mat img = imread(stringStream.str(), IMREAD_GRAYSCALE);
        ROI imgDims;
        imgDims.height = img.rows;
        imgDims.width  = img.cols;

        ROI blockDims;
        blockDims.height = (int)(img.rows / blockSize);
        blockDims.width  = (int)(img.cols / blockSize); 

        // buffer holding image by blocks
        unsigned char* imageBlocks = new uchar[imgDims.width*imgDims.height];

        // buffer holding marked image
        unsigned char* markedImg = new uchar[imgDims.width*imgDims.height];

        ret = cutImagePWrapper(img.data, imgDims, blockSize, imageBlocks);
        
        ret = onlineDetection(imageBlocks, blockDims, blockSize, refSignature, eta, false, markedImg, imgDims);
        
        ostringstream anotherStream;
        anotherStream << "E:/jee/cours/GEN5023/code/textile_images/results_cuda/img-" << i << "-defect.tiff";

        Mat markedImgMat(imgDims.height, imgDims.width, CV_8U, markedImg);
        bool success = imwrite(anotherStream.str(), markedImgMat);
        cout << anotherStream.str() << endl;
        printf("Success for image [%d] : %d \n", i, (int)success);
        
        delete[] markedImg;
        delete[] imageBlocks;
    }
    return 0;
}