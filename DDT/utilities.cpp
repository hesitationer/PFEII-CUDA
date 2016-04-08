#include "utilities.h"
using namespace utilities;

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
