//**********************************************
// cut_image function definition and types
//**********************************************
#ifndef H_CUT_IMAGE
#define H_CUT_IMAGE
#include <opencv2/core/core.hpp>
#include "utilities.h"

//**********************************************
// typedef a Cell of pointer to Open CV matrixes
//**********************************************
typedef utilities::Cell<cv::Mat*> MatrixPTRCell;

//************************************************************
// cut given images in blocs of size sBloc*sBloc
//************************************************************
// @params
// imageBlocs : a pointer to an indexing matrix of image blocs
// image      : path to the image file
// sBloc      : bloc size
// overlap    : use overlapping between image blocs (y/n)
// normalize  : normalize image
// @return    : 0 if everything ok, else 1, 2, 3.. 
//*************************************************************
int cutImage(MatrixPTRCell* imageBlocs, std::string image, int sBloc, bool overlap, bool normalize);

//************************************************************
// rebuild previously cutted image
//************************************************************
// @params
// imageBlocs : a pointer to an indexing matrix of image blocs
// sBloc      : image bloc size
// overlap    : does the image was cutted using overlaping
// normalize  : does the image was normalized while cutted
// image      : a pointer to a cv::Mat
// @return    : 0 if everthing worked as expected
//************************************************************
int uncutImage(MatrixPTRCell* imageBlocs, int sBloc, bool overlap, bool normalize, cv::Mat* image);
#endif // H_CUT_IMAGE