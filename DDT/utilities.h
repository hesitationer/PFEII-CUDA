#ifndef H_UTILITIES
#define H_UTILITIES
#include <opencv2/core/core.hpp>

//***********************************************************
// Utility classes/sctructures used to make some Matlab magic
//***********************************************************
namespace utilities
{
    //**********************************************
    // Cell class is lightweight and simple 2D 
    // container indexed by integers, like a matrix
    // data are holded by a std vector
    //**********************************************
	template <class T>
	class Cell
	{
	public:
		// constructor
		Cell<T>(int rows, int cols);

		// destructor, clear
		~Cell();

		// (i,j) cell accessor
		const T& operator()(int rowIdx, int colIdx) const;

		// common cell accessor
		const T& get(int rowIdx, int colIdx) const;

        // set value of a cell
        void set(int rowIdx, int colIdx, const T& data);

        int rows; //<! rows number
        int cols; //<! cols number

	private:
		int size; //<! number of elements in cell (cols*rows)
		std::vector<T> content; //<! cell content holder
    };

    //******************************
    // compile some basic types here
    //******************************
    template class Cell < int > ;
    template class Cell < float > ;
    template class Cell < double > ;

    //***********************************************
    // Compile Cell template code for cv::Mat pointer
    //***********************************************
    template class Cell < const cv::Mat >;
    template class Cell < cv::Mat >;

} // namespace utilities
#endif //H_UTILITIES