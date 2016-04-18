#ifndef H_UTILITIES
#define H_UTILITIES
#include <opencv2/core/core.hpp>

//***********************************************************
// Utility classes/sctructures used to make some Matlab magic
//***********************************************************
namespace utilities
{
    //********************************
    // Interquartile range of a vector
    //********************************
    template <typename T>
    T iqr(std::vector<T>& src);

    //template int iqr(std::vector<int>& src);
    //template float iqr(std::vector<float>& src);

    //**********************************************
    // Cell class is lightweight and simple 2D 
    // container indexed by integers, like a matrix.
    // data are holded by a std vector
    //**********************************************
	template <class T>
	class Cell
	{
	public:
		// constructor
		Cell<T>(int rows, int cols);

        // copy fro cv::Mat constructor
        Cell<T>(cv::Mat& src);

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
        const std::vector<T>& getContent() const {return content;} //<! a raw reference to content

	private:
		int size; //<! number of elements in cell (cols*rows)
		std::vector<T> content; //<! cell content holder
    };
} // namespace utilities
#endif //H_UTILITIES