
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
/*
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "utilities.h"

using namespace utilities;
using namespace cv;

int main(int argc, char** argv)
{
    Cell<Mat*> aMatrixCell(32, 32);

    Mat img = imread("E:/jee/cours/GEN5023/code/textile_images/Free/1.TIF", CV_LOAD_IMAGE_COLOR);

    if (img.data)
    {
        Mat imBloc = img(Range(0, 32), Range(0, 32));
        aMatrixCell.set(0, 0, &imBloc);
    }

	return 0;
}*/