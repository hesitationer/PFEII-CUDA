#include "utilities.h"
#include <iostream>

namespace utilities {
    template <class T>
    Cell<T>::Cell(int rows, int cols):
        rows(rows),
        cols(cols),
        content()
    {
        size = rows*cols;
        content.reserve(size);
    }

    template <class T>
    Cell<T>::~Cell()
    {
        content.clear();
    }

    template <class T>
    const T& Cell<T>::operator()(int rowIdx, int colIdx) const
    {
        return get(rowIdx, colIdx);
    }

    template <class T>
    const T& Cell<T>::get(int rowIdx, int colIdx) const
    {
        int contentIndex = colIdx + (rowIdx*cols);
        return content.at(contentIndex);
    }

    template <class T>
    void Cell<T>::set(int rowIdx, int colIdx, const T& data)
    {
        int contentIndex = colIdx + (rowIdx*cols);
        content.insert(content.begin()+contentIndex, data);
    }

    template <class T>
    Cell<T>::Cell(cv::Mat& src):
    cols(src.cols),
    rows(src.rows),
    content()
    {
        size = rows*cols;
        content.reserve(size);

        // copy src matrix in content
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                set(i, j, src.at<T>(i,j));
            }
        }
    }

    template <typename T>
    T iqr(std::vector<T>& src)
    {
        // sort input vector
        std::sort(src.begin(), src.end());
        int len = (int)src.size();

        // compute first and third quartiles indexes
        // in case theyre float
        double firstQrtIdx = (len*0.25) - 1;
        double thirdQrtIdx = (len*0.75) - 1;

        float firstQrtVal, thirdQrtVal;

        // index not an integer
        if (std::floor(firstQrtIdx) != firstQrtIdx)
        {
            firstQrtIdx = (int)std::ceil(firstQrtIdx);
            firstQrtVal = src[firstQrtIdx];
        }
        else // index is an integer, value is (idx+(idx+1))/2
        {
            firstQrtIdx = (int)firstQrtIdx;
            firstQrtVal = (src[firstQrtIdx] + src[firstQrtIdx + 1]) / 2;
        }

        // same for third qrt
        if (std::floor(thirdQrtIdx) != thirdQrtIdx)
        {
            thirdQrtIdx = (int)std::ceil(thirdQrtIdx);
            thirdQrtVal = src[thirdQrtIdx];
        }
        else // index is an integer, value is (idx+(idx+1))/2
        {
            thirdQrtIdx = (int)thirdQrtIdx;
            thirdQrtVal = (src[thirdQrtIdx] + src[thirdQrtIdx + 1]) / 2;
        }

        // return difference between first and third qrt
        return thirdQrtVal - firstQrtVal;
    }
} // namespace utilities

//******************************
// cook Cell templates here
//******************************
template utilities::Cell < int >;
template utilities::Cell < float >;
template utilities::Cell < double >;
template utilities::Cell < const cv::Mat >;
template utilities::Cell < cv::Mat >;

//***********************************************
// Cook iqr function templates here
//***********************************************
template int utilities::iqr<int>(std::vector<int>& src);
template float utilities::iqr<float>(std::vector<float>& src);
template double utilities::iqr<double>(std::vector<double>& src);
