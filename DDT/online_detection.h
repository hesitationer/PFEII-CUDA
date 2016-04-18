//*************************************************
// online_detection functions and types declaration
//*************************************************
#ifndef H_ONLINE_DETECTION
#define H_ONLINE_DETECTION

#include <opencv2/core/core.hpp>
#include "utilities.h"
#include "cut_image.h"

//*************************************************
// online_detection parse each image block, extract
// their kim signature and compare it to kim ref
// signature. Defectious blocs are returned in a
// Cell of image blocs with same size of given
//*************************************************
// @param imageBlocks : a Cell containing image blocks (Mat
// @param blockSize   : the size of image blocks
// @param refSign     : the kim reference signature
// @param eta         : constant used to compute defect treshold
// @param overlap     : do the cell contains overlapping blocks
// @param markedImage : a Matrix containing defectious blocks at their original position
int onlineDetection(MatrixRefCell& imageBlocks, int blockSize, std::vector<float>& refSign, float eta, bool overlap, cv::Mat& markedImage);

#endif // H_ONLINE_DETECTION