#ifndef H_KIM
#define H_KIM

#include <opencv2/core/core.hpp>

//*********************************************************
// Kim ordinal signature computation definition
//*********************************************************
// @param image : a pointer to a cv::Mat holding image data
// @param rows  : number of row of image
// @param cols  : number of cols of image
// @param sign  : a pointer to a std::vector which
//                will hold kim signature
// @return      : 0 if everything went well...
//**********************************************************
int extractKimSignature(const cv::Mat& imageBlock, int rows, int cols, std::vector<float>& sign);

#endif // H_KIM