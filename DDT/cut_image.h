//**********************************************
// cut_image functions and types declaration
//**********************************************
#ifndef H_CUT_IMAGE
#define H_CUT_IMAGE
#include <opencv2/core/core.hpp>
#include "utilities.h"

//**********************************************
// typedef a Cell of pointer to Open CV matrixes
//**********************************************
typedef utilities::Cell< const cv::Mat > MatrixRefCell;

//************************************************************
// cut given images in blocs of size sBloc*sBloc
//************************************************************
// @params imageBlocs : a pointer to an indexing matrix of image blocs
// @params image      : path to the image file
// @params sBloc      : bloc size
// @params overlap    : use overlapping between image blocs (y/n)
// @params normalize  : normalize image
// @return    : 0 if everything ok, else 1, 2, 3..
// @note      : Mat holded in imageBlocs ceil are dynamically
//              allocated here, caller responsible of
//              deallocating memory
//*************************************************************
int cutImage(MatrixRefCell& imageBlocks, std::string image, int blockSize, bool overlap, bool normalize);

//************************************************************
// rebuild previously cutted image
//************************************************************
// @paramsimageBlocs : a pointer to an indexing matrix of image blocs
// @paramssBloc      : image bloc size
// @paramsoverlap    : does the image was cutted using overlaping
// @paramsnormalize  : does the image was normalized while cutted
// @paramsimage      : a pointer to a cv::Mat
// @return    : 0 if everthing worked as expected
//************************************************************
int uncutImage(const MatrixRefCell& imageBlocks, int blockSize, bool overlap, bool normalize, cv::Mat* image);
#endif // H_CUT_IMAGE