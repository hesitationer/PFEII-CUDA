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
int cutImage(MatrixRefCell& imageBlocks, std::string image, int blockSize, bool overlap, bool normalize)
{
    int ret = 0;
	(void)normalize; // normalization not implemented yet

    Mat img = imread(image, IMREAD_GRAYSCALE);
	
	if (!img.data) // fail to read image
	{
		ret = 1;
	}
	else
	{
		int imgHeight = img.rows;
		int imgWidth = img.cols;

        double hBlocs = ceil((double)imgHeight / (double)blockSize);
        double wBlocs = ceil((double)imgWidth / (double)blockSize);

		// overlapping increase number of blocs
		if (overlap)
		{
            hBlocs = hBlocs + (hBlocs - 1);
            wBlocs = wBlocs + (wBlocs - 1);
		}

        int step = blockSize;
        if (overlap) // overlapping divide step by two
        {
            step = ceil<int>(blockSize / 2);
        }

        // fill the cell with each blocs
        for (int i = 0; i < hBlocs; ++i)
        {
            for (int j = 0; j < wBlocs; ++j)
            {
                // caller is responsible to deallocate memory
                std::auto_ptr<Mat> imgBloc(new Mat(blockSize, blockSize, CV_8UC3));
                img(Range(i*blockSize, (i + 1)*blockSize), Range(j*blockSize, (j + 1)*blockSize)).copyTo(*imgBloc);
                imageBlocks.set(i, j, *imgBloc);
            }
        }
	}
	return ret;
}

//************************************************************
// rebuild previously cutted image
//************************************************************
int uncutImage(const MatrixRefCell& imageBlocks, int blockSize, bool overlap, bool normalize, cv::Mat* image)
{
    for (int i = 0; i < imageBlocks.rows; ++i)
    {
        for (int j = 0; j < imageBlocks.cols; ++j)
        {
            const Mat& aBlock = imageBlocks.get(i, j);
            aBlock.copyTo((*image)(Range(i*blockSize, (i + 1)*blockSize), Range(j*blockSize, (j + 1)*blockSize)));
        }
    }
    return 0;
}