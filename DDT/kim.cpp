//-----------------------------------------------
// implementation file of kim signature functions
//-----------------------------------------------

//----------------------------
// INCLUDES
//----------------------------
#include "kim.h"
#include "utilities.h"
//----------------------------
// NAMESPACES
//----------------------------
using namespace cv;
using namespace std;
using namespace utilities;
//----------------------------
// IMPLEMENTATION
//----------------------------
int extractKimSignature(const Mat& imageBlock, int rows, int cols, vector<float>& sign)
{
    int blockRows = rows / 8;
    int blockCols = cols / 8;
    int blockSize = blockRows*blockCols;

    //Mat imageBlocksMean(8, 8, DataType<float>::type);
    Mat_<float> imageBlocksMean(8, 8);

    // cut imageBlock in blocks of size blockRows*blockCols, like we did in cut_image
    // and compute its mean value
    for (int i = 0; i < 8; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            Mat aBlock, aMeanBlock;
            imageBlock(Range(i*blockRows, (i + 1)*blockRows), Range(j*blockCols, (j + 1)*blockCols)).copyTo(aBlock);
            cv::Scalar blockMean = mean(aBlock);
            imageBlocksMean(i, j) = (float)blockMean.val[0];
        }
    }

    // run DCT on mean matrix and keep
    // only 6*6 part
    Mat dctBlocks, dctBlocksSix;
    dct(imageBlocksMean, dctBlocks);
    dctBlocks(Range(0, 6), Range(0, 6)).copyTo(dctBlocksSix);
    
    // put all values in a signature vector
    // signature contains only 35 elements cause Kim signature discard DC coef
    sign.reserve(35);
    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 6; ++j)
        {
            if (i == 0 && j == 0) // discard DC coef
                continue;
            else
                sign.push_back( abs( dctBlocksSix.at<float>(i, j) ) );
        }
    }

    // sort and reverse signature vector
    std::sort(sign.begin(), sign.end());
    std::reverse(sign.begin(), sign.end());

    return 0;
}