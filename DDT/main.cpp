#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "utilities.h"
#include "cut_image.h"
#include "kim.h"
#include "online_detection.h"
#include <iostream>

using namespace std;
using namespace utilities;
using namespace cv;

const int sBloc = 32;

int main(int argc, char** argv)
{
    Cell<const Mat> aMatrixCell(8, 12);
    int ret = cutImage(aMatrixCell, "E:/jee/cours/GEN5023/code/textile_images/Free/1.TIF", sBloc, false, false);
    int imgH = aMatrixCell.rows*sBloc;
    int imgW = aMatrixCell.cols*sBloc;
    
    vector<float> refSignature;
    ret = extractKimSignature(aMatrixCell.get(0, 0), 32, 32, refSignature);
    
    for (int i = 1; i <= 26; ++i)
    {
        ostringstream stringStream; 
        stringStream << "E:/jee/cours/GEN5023/code/textile_images/Defect/" << i << ".TIF";
        
        Cell<const Mat> anotherMatrixCell(8,12);

        Mat markedImg(256, 384, aMatrixCell.get(0,0).type());
        
        ret = cutImage(anotherMatrixCell, stringStream.str(), sBloc, false, false);
    
        onlineDetection(anotherMatrixCell, sBloc, refSignature, 4, false, markedImg);
        
        //imshow(stringStream.str(), markedImg);
        //waitKey(1);
        ostringstream anotherStream;
        anotherStream << "E:/jee/cours/GEN5023/code/textile_images/results/img-" << i << "-defect.tiff";
        bool success = imwrite(anotherStream.str(), markedImg);
        cout << anotherStream.str() << endl;
        printf("Success for image [%d] : %d \n", i, (int)success);
    }
    return 0;
}