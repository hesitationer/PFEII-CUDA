#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "utilities.h"
#include "cut_image.h"
#include <iostream>

using namespace utilities;
using namespace cv;

const int sBloc = 32;

int main(int argc, char** argv)
{
    Cell<Mat*> aMatrixCell(8, 12);

    int ret = cutImage(&aMatrixCell, "E:/jee/cours/GEN5023/code/textile_images/Free/1.TIF", sBloc, false, false);
    int imgH = aMatrixCell.rows*sBloc;
    int imgW = aMatrixCell.cols*sBloc;
    Mat image(imgH, imgW, CV_8UC3);
    
    ret = uncutImage(&aMatrixCell, 32, false, false, &image);

    namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("Display window", image);
    waitKey(0);

    for (int i = 0; i < aMatrixCell.rows; ++i)
    {
        for (int j = 0; j < aMatrixCell.cols; ++j)
        {
            delete aMatrixCell.get(i, j);
        }
    }
    return 0;
}