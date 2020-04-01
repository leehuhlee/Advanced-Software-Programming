// OpenCVcolorConversion.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std; 

int _tmain(int argc, _TCHAR* argv[])
{

         IplImage *img1 = cvLoadImage("lena_8.bmp", CV_LOAD_IMAGE_UNCHANGED);
		printf("width: %d, hight: %d\n", img1->width, img1->height);
		printf("widthStep: %d\n", img1->widthStep);
		printf("nChannels: %d\n", img1->nChannels);
		printf("depth: %d\n\n", img1->depth);

        cvNamedWindow("problem1_lena_8:",1);  //int cvNamedWindow( const char* name, int flags=CV_WINDOW_AUTOSIZE );
        cvShowImage("problem1_lena_8:",img1); // image disply function (for IplImage datatype)

		double alpha, beta;
		//Mat A(30, 40, DataType<unsigned char>::type);
		Mat img1_Mat = cvarrToMat(img1);
		Mat img1_Mat_half, img1_Mat_double;
		alpha = 0.5;
		beta = 0;

		/////////////////////////////////////////////////////////////////

		// USE addWeighted(...) FUNCTION TO REDUCE PIXEL VALUES BY HALF
		addWeighted(img1_Mat, alpha, img1_Mat, beta, 0, img1_Mat_half);
		
		/////////////////////////////////////////////////////////////////

		alpha = 2;

		/////////////////////////////////////////////////////////////////

		// USE addWeighted(...) FUNCTION TO INCREASE PIXEL VALUES TWICE

		addWeighted(img1_Mat, alpha, img1_Mat, beta, 0, img1_Mat_double);

		/////////////////////////////////////////////////////////////////

		//cvNamedWindow("problem1_lena_8_half:",1);
		imshow("problem1_lena_8_half", img1_Mat_half);
		//cvNamedWindow("problem1_lena_8_double:",1);
		imshow("problem1_lena_8_double", img1_Mat_double);

		imwrite("problem1_lena_8_half.bmp", img1_Mat_half);
		imwrite("problem1_lena_8_double.bmp", img1_Mat_double);
		


		IplImage *img2 = cvLoadImage("lena_24.bmp", CV_LOAD_IMAGE_UNCHANGED);
		printf("width: %d, hight: %d\n", img2->width, img2->height);
		printf("widthStep: %d\n", img2->widthStep);
		printf("nChannels: %d\n", img2->nChannels);
		printf("depth: %d\n\n", img2->depth);

        cvNamedWindow("problem2_lena_24:",1);  //int cvNamedWindow( const char* name, int flags=CV_WINDOW_AUTOSIZE );
        cvShowImage("problem2_lena_24:",img2);


		/////////////////////////////////////////////////////////////////

		// USE cvtColor(...) FUNCTION TO CONVERT 24 BIT COLOR IMAGE TO GRAYSCALE
		// USE imshow OR cvShowImage TO DISPLAY THE IMAGE
		// USE imwrite TO SAVE THE IMAGE

		Mat img2_Mat = cvarrToMat(img2);
		Mat img2_Mat_gray;
		cvtColor(img2_Mat, img2_Mat_gray, COLOR_BGR2GRAY);

		imshow("problem2_lena_24_gray.bmp", img2_Mat_gray);
		imwrite("problem2_lena_24_gray.bmp", img2_Mat_gray);

		/////////////////////////////////////////////////////////////////



		IplImage *img3 = cvLoadImage("image_16.bmp", CV_LOAD_IMAGE_UNCHANGED);
		printf("width: %d, hight: %d\n", img3->width, img3->height);
		printf("widthStep: %d\n", img3->widthStep);
		printf("nChannels: %d\n", img3->nChannels);
		printf("depth: %d\n\n", img3->depth);

        cvNamedWindow("problem3_image_16:",1);  //int cvNamedWindow( const char* name, int flags=CV_WINDOW_AUTOSIZE );
        cvShowImage("problem3_image_16:",img3);


		/////////////////////////////////////////////////////////////////

		// USE cvtColor(...) FUNCTION TO CONVERT 16 BIT COLOR IMAGE TO GRAYSCALE
		// USE imshow OR cvShowImage TO DISPLAY THE IMAGE
		// USE imwrite TO SAVE THE IMAGE

		Mat img3_Mat = cvarrToMat(img3);
		Mat img3_Mat_gray;
		cvtColor(img3_Mat, img3_Mat_gray, COLOR_BGR2GRAY);

		imshow("problem3_image_16.bmp", img3_Mat_gray);
		imwrite("problem3_image_16.bmp", img3_Mat_gray);

		/////////////////////////////////////////////////////////////////

		printf("OpenCV regards the \"depth\" as depth for each channel.\n");
		printf("The \"depth\" information in OpenCV seems little wrong for 16 bit image, but it does the color conversion correctly. This is why sometimes it is the best choice to use the pixel value accessing functions provided by the tool under use.\n");


        cvWaitKey();
        cvDestroyWindow("problem1_lena_8:");
		cvDestroyWindow("problem1_lena_half:");
		cvDestroyWindow("problem1_lena_double:");
		cvReleaseImage(&img1);

		cvDestroyWindow("problem2_lena_24:");
		//cvDestroyWindow("problem2_lena_24_gray:");
		cvReleaseImage(&img2);

		cvDestroyWindow("problem3_image_16:");
		//cvDestroyWindow("problem3_image_16_gray:");
		cvReleaseImage(&img3);


        return 0;
}

