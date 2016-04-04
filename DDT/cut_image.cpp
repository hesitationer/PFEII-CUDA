//-------------------------------------------
// implementation file of cut image functions
//-------------------------------------------

//----------------------------
// INCLUDES
//----------------------------
#include "cut_image.h"
#include <opencv2/highgui/highgui.hpp>

//----------------------------
// NAMESPACES
//----------------------------
using namespace cv;
using namespace std;
using namespace utilities;

//----------------------------
// IMPLEMENTATION
//----------------------------

//************************************************************
// cut given images in blocs of size sBloc*sBloc
//************************************************************
int cutImage(MatrixPTRCell* cutImage, std::string image, int sBloc, bool overlap, bool normalize)
{
	int ret = 0;
	(void)normalize; // normalization not implemented yet

	Mat img = imread(image, IMREAD_COLOR);
	
	if (!img.data) // fail to read image
	{
		ret = 1;
	}
	else
	{
		int imgHeight = img.rows;
		int imgWidth = img.cols;

        double hBlocs = ceil((double)imgHeight / (double)sBloc);
        double wBlocs = ceil((double)imgWidth / (double)sBloc);

		// overlapping increase number of blocs
		if (overlap)
		{
            hBlocs = hBlocs + (hBlocs - 1);
            wBlocs = wBlocs + (wBlocs - 1);
		}

        int step = sBloc;
        if (overlap) // overlapping divide step by two
        {
            step = ceil<int>(sBloc / 2);
        }

        // fill the cell with each blocs
        for (int i = 0; i < hBlocs; ++i)
        {
            for (int j = 0; j < wBlocs; ++j)
            {
                // caller is responsible to deallocate memory
                Mat* imgBloc = new Mat(sBloc, sBloc, CV_8UC3);
                img(Range(i*sBloc, (i + 1)*sBloc), Range(j*sBloc, (j + 1)*sBloc)).copyTo(*imgBloc);
                cutImage->set(i, j, imgBloc);
            }
        }
	}
	return ret;
}

//************************************************************
// rebuild previously cutted image
//************************************************************
int uncutImage(MatrixPTRCell* imageBlocs, int sBloc, bool overlap, bool normalize, cv::Mat* image)
{
    for (int i = 0; i < imageBlocs->rows; ++i)
    {
        for (int j = 0; j < imageBlocs->cols; ++j)
        {
            Mat* aBlock = imageBlocs->get(i, j);
            aBlock->copyTo((*image)(Range(i*sBloc, (i + 1)*sBloc), Range(j*sBloc, (j + 1)*sBloc)));
        }
    }
    return 0;
}