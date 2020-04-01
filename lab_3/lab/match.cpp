/*
Detects SIFT features in two images and finds matches between them.

Copyright (C) 2006  Rob Hess <hess@eecs.oregonstate.edu>

@version 1.1.1-20070330
*/
#include "stdafx.h"

#include "sift.h"
#include "imgfeatures.h"
#include "kdtree.h"
#include "utils.h"
#include "xform.h"

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

#include <stdio.h>
#include <iostream>
#include <fstream>

#include "HelperFunctions.h"

using namespace std;

/* the maximum number of keypoint NN candidates to check during BBF search */
#define KDTREE_BBF_MAX_NN_CHKS 200

/* threshold on squared ratio of distances between NN and 2nd NN */
#define NN_SQ_DIST_RATIO_THR 0.49
//#define NN_SQ_DIST_RATIO_THR 0.79

#define PI 3.141592
#define square(a) (a)*(a)

/******************************** Globals ************************************/

//char img1_file[] = "..\\beaver.png";
//char img2_file[] = "..\\beaver_xform.png";

//char img1_file[] = "train\\basmati.bmp";
//char img2_file[] = "train\\scene.bmp";

//char img1_file[] = "F:/_sift_fp/FVC2002/db1_jpg/0101.jpg";
//char img2_file[] = "F:/_sift_fp/FVC2002/db1_jpg/0102.jpg";

//char img1_file[] = "train\\051000012319_162830_1_03_0000000000.jpg";
//char img2_file[] = "train\\051000012319_162957_1_02_0000000000.jpg";

char img1_file[] = "../../../TattooIDtemp/All_Query/0001_0000237688.jpg";
char img2_file[] = "../../../TattooIDtemp/All_Query/0001_0000237702.jpg";

//char img1_file[] = "F:/TattooIDtemp/All_Query/0002_0000237742.jpg";
//char img2_file[] = "F:/TattooIDtemp/All_Query/0002_0000237780.jpg";

//char img1_file[] = "F:/TattooIDtemp/All_Query/0031_0000638323.jpg";
//char img2_file[] = "F:/TattooIDtemp/All_Query/0031_0000638445.jpg";

char inPath[] = "F:/SIFT/_out_prip/db1_jpg/";
char outPath[] = "F:/SIFT/_out_prip/db1_ucla.txt";

int batch_matching( int argc, char** argv );
int getDirBar(float ori, float &x2, float &y2);
int matchPoints (feature *feat1, feature *feat2, int, int) ;
float matchGeometry(struct feature *feat1, struct feature *feat2, int, int);
int getCorrespondences(struct feature *feat1, struct feature *feat2, Coord &crspnd1, Coord &crspnd2, Coord &crd1, Coord &crd2, int, int);
float getMatchLSF(Coord &crspnd1, Coord &crspnd2, Coord &crd1, Coord &crd2);
void getMask (unsigned char *mask, IplImage *img);
int GetDirFiles(const char *path_name, CString *lst);
int GetDirFilesAscend(CString *lst, CString *lst2, int n);
void fillMiddle(float* pGray, float* out, int curRR, int curCC);
int edgeSobel(float *img, float *sbMag, float *sbDir, int w, float thresh, int curRR, int curCC);
int Dilate(float* pGray, float* out, unsigned char kernel, int curRR, int curCC);	// Dilate
int Erode(float* pGray, float* out, unsigned char kernel, int curRR, int curCC);	// Erode

int wIm1, hIm1, wIm2, hIm2, nMatchCoarse, nMatchTrim, nMatchBest;
float ** inputArray;
float * d;
float * e;
float ** transform;
unsigned char *siftMask1, *siftMask2, *Bin;
float *baseIm;
int *lBound, *rBound;
IplImage *stacked;

LARGE_INTEGER Start, End, Frequency;	// variables for time measure
double TimeTaken;
void tic(char* str);
void toc(char* str);

/********************************** Main *************************************/

int main( int argc, char** argv )
{
	IplImage* img1, * img2;
	struct feature* feat1, * feat2, * feat;
	struct feature** nbrs;
	struct kd_node* kd_root;
	CvPoint pt1, pt2;
	double d0, d1;
	int n1, n2, fn1, fn2, k, i, j, m, n, n_imp;
	ofstream ofile;

	CString filelist[15000], filelist2[15000], fname1, fname2, csIn, csOut;
	const char *path_name1;

	csIn = inPath;
	csOut = outPath;


	path_name1 = (LPCTSTR) csIn;
	int filenum = GetDirFiles(path_name1, filelist2);
	GetDirFilesAscend(filelist2, filelist, filenum);

	//fname1 = csIn + filelist[0];
	img1 = cvLoadImage( img1_file, 1 );
	img2 = cvLoadImage( img2_file, 1 );
	
	wIm1 = img1->width;
	hIm1 = img1->height;
	wIm2 = img2->width;
	hIm2 = img2->height;

	//fname1 = csIn + filelist[0];
	//fname2 = csIn + filelist[1];

	//CString strFilter;
	//CSimpleArray<GUID> aguidFileTypes;
	//HRESULT hResult;
	//COLORREF col;
	//CDC *pDC1=NULL, *pDC2=NULL, *pDC3=NULL;
	//int i, j, k;

	//hResult = m_Image1.GetExporterFilterString(strFilter,aguidFileTypes);
	//CFileDialog dlg1(TRUE, ".bmp", NULL, OFN_FILEMUSTEXIST, "*.*||" );
	//dlg1.m_ofn.nFilterIndex = m_nFilterLoad;
	//if (dlg1.DoModal()==IDOK) {
	//	m_nFilterLoad = dlg1.m_ofn.nFilterIndex;
	//	//m_Image1.Destroy();
	//	hResult = m_Image1.Load(dlg1.GetFileName());
	//}
	//else
	//	return;


	stacked = stack_imgs( img1, img2 );

	tic("");
	n1 = sift_features( img1, &feat1);
	toc("Feature extraction, image 1: ");
	printf("%d features\n", n1);

	tic("");
	n2 = sift_features( img2, &feat2);
	toc("Feature extraction, image 2: ");
	printf("%d features\n", n2);

	tic("");
	nMatchCoarse = matchPoints (feat1, feat2, n1, n2);
	toc("Matching: ");

	
	printf("score: %d\n", nMatchCoarse);


	//fprintf( stderr, "Found %d total matches\n", m );
	cvNamedWindow( "Matches", 1 );
	cvShowImage( "Matches", stacked );
	cvWaitKey( 0 );


	/* 
	UNCOMMENT BELOW TO SEE HOW RANSAC FUNCTION WORKS

	Note that this line above:

	feat1[i].fwd_match = nbrs[0];

	is important for the RANSAC function to work.
	*/
	/*
	{
		CvMat* H;
		H = ransac_xform( feat1, n1, FEATURE_FWD_MATCH, lsq_homog, 4, 0.01,
			homog_xfer_err, 3.0, NULL, NULL );
		if( H )
		{
			IplImage* xformed;
			xformed = cvCreateImage( cvGetSize( img2 ), IPL_DEPTH_8U, 3 );
			cvWarpPerspective( img1, xformed, H, 
				CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS,
				cvScalarAll( 0 ) );
			cvNamedWindow( "Xformed", 1 );
			cvShowImage( "Xformed", xformed );
			cvWaitKey( 0 );
			cvReleaseImage( &xformed );
			cvReleaseMat( &H );
		}
	}
	*/

	cvReleaseImage( &stacked );
	cvReleaseImage( &img1 );
	cvReleaseImage( &img2 );
	//kdtree_release( kd_root );
	free( feat1 );
	free( feat2 );
	return 0;
}

int batch_matching( int argc, char** argv )
{
	IplImage* img1, * img2;
	struct feature* feat1, * feat2, * feat;
	struct feature** nbrs;
	struct kd_node* kd_root;
	CvPoint pt1, pt2;
	double d0, d1;
	int n1, n2, fn1, fn2, k, i, j, m, n, n_imp;
	ofstream ofile;

	CString filelist[15000], filelist2[15000], fname1, fname2, csIn, csOut;
	const char *path_name1;

	csIn = inPath;
	csOut = outPath;


	path_name1 = (LPCTSTR) csIn;
	int filenum = GetDirFiles(path_name1, filelist2);
	GetDirFilesAscend(filelist2, filelist, filenum);

	fname1 = csIn + filelist[0];
	img1 = cvLoadImage( fname1, 1 );
	wIm1 = img1->width;
	hIm1 = img1->height;
	siftMask1 = (unsigned char*) malloc(sizeof(unsigned char)*wIm1*hIm1);
	siftMask2 = (unsigned char*) malloc(sizeof(unsigned char)*wIm1*hIm1);
	baseIm = (float*) malloc(sizeof(float)*wIm1*hIm1);
	Bin = new unsigned char[wIm1*hIm1];
	lBound = new int[hIm1];
	rBound = new int[hIm1];

	//fname1 = csIn + filelist[0];
	//fname2 = csIn + filelist[1];

	//CString strFilter;
	//CSimpleArray<GUID> aguidFileTypes;
	//HRESULT hResult;
	//COLORREF col;
	//CDC *pDC1=NULL, *pDC2=NULL, *pDC3=NULL;
	//int i, j, k;

	//hResult = m_Image1.GetExporterFilterString(strFilter,aguidFileTypes);
	//CFileDialog dlg1(TRUE, ".bmp", NULL, OFN_FILEMUSTEXIST, "*.*||" );
	//dlg1.m_ofn.nFilterIndex = m_nFilterLoad;
	//if (dlg1.DoModal()==IDOK) {
	//	m_nFilterLoad = dlg1.m_ofn.nFilterIndex;
	//	//m_Image1.Destroy();
	//	hResult = m_Image1.Load(dlg1.GetFileName());
	//}
	//else
	//	return;

	for (fn1=0; fn1<filenum; fn1++) {

		fname1 = csIn + filelist[fn1];
		img1 = cvLoadImage( fname1, 1 );
		if( ! img1 )
			fatal_error( "unable to load image from %s", fname1 );
		getMask (siftMask1, img1);
		n1 = sift_features( img1, &feat1);

		n = min((floor(((float)fn1)/8)+1)*8, filenum);

		for (fn2=fn1+1; fn2<n; fn2++) {
			fname2 = csIn + filelist[fn2];
			img2 = cvLoadImage( fname2, 1 );
			wIm2 = img2->width;
			hIm2 = img2->height;

			if( ! img2 )
				fatal_error( "unable to load image from %s", fname2 );
			stacked = stack_imgs( img1, img2 );

			//fprintf( stderr, "Finding features in %s...\n", img2_file );
			getMask (siftMask2, img2);
			n2 = sift_features( img2, &feat2);
			//printf("n2 = %d\n", n2);


			nMatchCoarse = matchPoints (feat1, feat2, n1, n2);
//cvNamedWindow( "Matches", 1 );
//cvShowImage( "Matches", stacked );
//cvWaitKey( 0 );

			nMatchTrim = matchGeometry(feat1, feat2, n1, n2);
//cvNamedWindow( "Matches", 1 );
//cvShowImage( "Matches", stacked );
//cvWaitKey( 0 );

			for( i = 0; i < n1; i++ ) {
				feat1[i].dist0 = -1;
				feat1[i].matched = 0;
				feat1[i].trimmed = 0;
			}

			printf( "%s %s: %d %d ", filelist[fn1].Left(4), filelist[fn2].Left(4), nMatchCoarse, nMatchTrim );

			ofile.open (csOut, ofstream::app);
			ofile << filelist[fn1].Left(4) << " " << filelist[fn2].Left(4) << " " << 
				nMatchCoarse << " " << nMatchTrim << " ";
			ofile.close();


			nMatchCoarse = matchPoints (feat2, feat1, n2, n1);

			nMatchTrim = matchGeometry(feat2, feat1, n2, n1);

			for( i = 0; i < n2; i++ ) {
				feat2[i].dist0 = -1;
				feat2[i].matched = 0;
				feat2[i].trimmed = 0;
			}

			printf( "%d %d\n", nMatchCoarse, nMatchTrim );

			ofile.open (csOut, ofstream::app);
			ofile << nMatchCoarse << " " << nMatchTrim << endl;
			ofile.close();

			free( feat2 );

		}

		if (fn1%8==0) {
			n_imp = fn1 + 8;
			while (n_imp < filenum) {

				fname2 = csIn + filelist[n_imp];
				img2 = cvLoadImage( fname2, 1 );
				wIm2 = img2->width;
				hIm2 = img2->height;

				if( ! img2 )
					fatal_error( "unable to load image from %s", fname2 );
				//stacked = stack_imgs( img1, img2 );

				//fprintf( stderr, "Finding features in %s...\n", img2_file );
				getMask (siftMask2, img2);
				n2 = sift_features( img2, &feat2);
				//printf("n2 = %d\n", n2);


				nMatchCoarse = matchPoints (feat1, feat2, n1, n2);

				nMatchTrim = matchGeometry(feat1, feat2, n1, n2);

				for( i = 0; i < n1; i++ ) {
					feat1[i].dist0 = -1;
					feat1[i].matched = 0;
					feat1[i].trimmed = 0;
				}

				printf( "%s %s: %d %d ", filelist[fn1].Left(4), filelist[n_imp].Left(4), nMatchCoarse, nMatchTrim );

				ofile.open (csOut, ofstream::app);
				ofile << filelist[fn1].Left(4) << " " << filelist[n_imp].Left(4) << " " << 
					nMatchCoarse << " " << nMatchTrim << " ";
				ofile.close();


				nMatchCoarse = matchPoints (feat2, feat1, n2, n1);

				nMatchTrim = matchGeometry(feat2, feat1, n2, n1);

				for( i = 0; i < n2; i++ ) {
					feat2[i].dist0 = -1;
					feat2[i].matched = 0;
					feat2[i].trimmed = 0;
				}

				printf( "%d %d\n", nMatchCoarse, nMatchTrim );

				ofile.open (csOut, ofstream::app);
				ofile << nMatchCoarse << " " << nMatchTrim << endl;
				ofile.close();


				//cvNamedWindow( "Matches", 1 );
				//cvShowImage( "Matches", stacked );
				//cvWaitKey( 0 );

				free( feat2 );

				n_imp += 8;
			}
		}
		
		
		free( feat1 );
	}


	//fprintf( stderr, "Found %d total matches\n", m );
	//cvNamedWindow( "Matches", 1 );
	//cvShowImage( "Matches", stacked );
	//cvWaitKey( 0 );


	/* 
	UNCOMMENT BELOW TO SEE HOW RANSAC FUNCTION WORKS

	Note that this line above:

	feat1[i].fwd_match = nbrs[0];

	is important for the RANSAC function to work.
	*/
	/*
	{
		CvMat* H;
		H = ransac_xform( feat1, n1, FEATURE_FWD_MATCH, lsq_homog, 4, 0.01,
			homog_xfer_err, 3.0, NULL, NULL );
		if( H )
		{
			IplImage* xformed;
			xformed = cvCreateImage( cvGetSize( img2 ), IPL_DEPTH_8U, 3 );
			cvWarpPerspective( img1, xformed, H, 
				CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS,
				cvScalarAll( 0 ) );
			cvNamedWindow( "Xformed", 1 );
			cvShowImage( "Xformed", xformed );
			cvWaitKey( 0 );
			cvReleaseImage( &xformed );
			cvReleaseMat( &H );
		}
	}
	*/

	cvReleaseImage( &stacked );
	cvReleaseImage( &img1 );
	cvReleaseImage( &img2 );
	kdtree_release( kd_root );
	free( feat1 );
	free( feat2 );
	return 0;
}

int matchPoints (feature *feat1, feature *feat2, int n1, int n2) 
{
	int i, k, d0, d1;
	struct feature * feat;
	struct feature** nbrs;
	struct kd_node* kd_root;
	CvPoint pt1, pt2;

	kd_root = kdtree_build( feat2, n2 );
	int m = 0;

	for( i = 0; i < n1; i++ )
	{
		feat = feat1 + i;
		k = kdtree_bbf_knn( kd_root, feat, 2, &nbrs, KDTREE_BBF_MAX_NN_CHKS );
		if( k == 2 )
		{
			d0 = descr_dist_sq( feat, nbrs[0] );
			d1 = descr_dist_sq( feat, nbrs[1] );
			if( d0 < d1 * NN_SQ_DIST_RATIO_THR )
			{
				pt1 = cvPoint( cvRound( feat->x ), cvRound( feat->y ) );
				pt2 = cvPoint( cvRound( nbrs[0]->x ), cvRound( nbrs[0]->y ) );
				////pt2.y += img1->height;
				pt2.x += wIm1;
				cvLine( stacked, pt1, pt2, CV_RGB(255,255,0), 1, 8, 0 );
				m++;
				feat1[i].fwd_match = nbrs[0];
//printf ("%d %d %d %d %d\n", i, pt2.x, pt2.y, pt1.x, pt1.y);

				feat1[i].matched = 1;
				feat1[i].fx = (double) pt2.x-wIm2;	// added due to the error in fwd_match???
				feat1[i].fy = (double) pt2.y;	// right?
			}
			feat1[i].dist0 = d0;
		}
		else {
			feat1[i].dist0 = -1;
			feat1[i].matched = 0;
		}

		feat1[i].trimmed = 0;	// initial value for triming

		free( nbrs );
	}
	return m;
}

float matchGeometry(struct feature *feat1, struct feature *feat2, int n1, int n2)
{

	int nCrspnd;
	float score;
	Coord crspnd1, crspnd2, crd1, crd2;
	CvPoint pt1, pt2;

	nCrspnd = getCorrespondences(feat1, feat2, crspnd1, crspnd2, crd1, crd2, n1, n2);

//printf("nCrspnd = %d\n", nCrspnd);


	//if (nCrspnd >= 3) {
	//	//printf("cnt coord: %d %d %d %d\n", crspnd1.count, crspnd2.count, crd1.count, crd2.count);
	//	score = getMatchLSF(crspnd1, crspnd2, crd1, crd2);
	//}
	//else
	//	score = -1;

	//free(crspnd1.dim1);
	//free(crspnd1.dim2);
	//free(crspnd1.dim3);
	//free(crspnd2.dim1);
	//free(crspnd2.dim2);
	//free(crspnd2.dim3);
	//free(crd1.dim1);
	//free(crd1.dim2);
	//free(crd1.dim3);
	//free(crd2.dim1);
	//free(crd2.dim2);
	//free(crd2.dim3);

	//printf("coarse match: %d\n", m_NmatchCoarse);		// DEBUG
	//printf("trimmed match: %d\n", m_NmatchTrim);
	//printf("best match: %d\n", m_NmatchBest);
	//printf("score, geometry: %f\n", score);

	//return score;
	return nCrspnd;
}

int getCorrespondences(struct feature *feat1, struct feature *feat2, Coord &crspnd1, Coord &crspnd2, 
					   Coord &crd1, Coord &crd2, int n1, int n2)
{
	//// trim outliers based on coherent directions and lengths
	int i, j, o;
	int nBinDir=120, nBinLen=10;
	int mchDirHist[120], mchLenHist[10]; // -90 ~ 90
	int mchIdx, mchDirMaxIdx, mchLenMaxIdx, mchcnt, x1, x2, y1, y2;
	float mchVal, mchMaxVal, Len, lenIntv, minLen, maxLen;
	CvPoint pt1, pt2;

	struct feature *feat, *featmatch;
	struct detection_data* ddata;

	//std::vector <FeatureSift>::iterator Iter1;
	//std::vector <FeatureSift>::iterator Iter2;

	for (i=0; i<nBinDir; i++)
		mchDirHist[i] = 0;
	for (i=0; i<nBinLen; i++)
		mchLenHist[i] = 0;

//int cnt = 1;
//for( i = 0; i < nFeat1; i++ ) {
//	feat = feat1+i;
//	if (feat1[i].dist0 >= 0 && feat1[i].matched==1) {
//		printf("matched %d\n", cnt++);
//	}
//}


	minLen = 9999;
	maxLen = 0;
	for( i = 0; i < n1; i++ ) {
	//for (Iter1 = feat1->Features.begin(); Iter1 != feat1->Features.end(); Iter1++) {

		feat = feat1+i;
		if (feat1[i].dist0 >= 0 && feat1[i].matched==1) {
			//featmatch = feat1[i].fwd_match;

	//printf("%f %f %f %f\n", feat->x, feat->y, featmatch->x, featmatch->y);
			//printf("%f %f %f %f\n", feat->x, feat->y, feat->fx, feat->fy);

			//ddata = feat_detection_data( feat );

			//o = pow(2.0,ddata->octv);
			x1 = feat->x;
			y1 = feat->y;
			//x2 = featmatch->x + wIm1;
			//y2 = featmatch->y;
			x2 = feat->fx + wIm1;
			y2 = feat->fy;

			mchVal = atan2((double)y2-y1, (double)x2-x1)+PI;
			mchIdx = (int) floor(((double)mchVal/PI*180-1)/(360/nBinDir));
	//printf("%d  ", mchIdx);
			mchDirHist[mchIdx]++;

			Len = sqrt((double) (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));

			if (Len > maxLen) {
				maxLen = Len;
			}
			if (Len < minLen) {
				minLen = Len;
			}
		}

	}
	//printf("----------------mchIdx  \n");
	//printf("\n\n\n");



	lenIntv = (maxLen-minLen)/(nBinLen-1);

	//printf("mxlen = %f  mnlen = %f  lenIntv=%f\n", maxLen, minLen, lenIntv);

	for( i = 0; i < n1; i++ ) {
	//for (Iter1 = Sift1->Features.begin(); Iter1 != Sift1->Features.end(); Iter1++) {
		//if (Iter1->match > 0) {
		
		if (feat1[i].dist0 >= 0 && feat1[i].matched==1) {
			feat = feat1+i;
			//featmatch = feat->fwd_match;

			//ddata = feat_detection_data( feat );

			//o = pow(2.0,ddata->octv);
			x1 = feat->x;
			y1 = feat->y;
			//x2 = featmatch->x + wIm1;
			//y2 = featmatch->y;
			x2 = feat->fx + wIm1;
			y2 = feat->fy;

			Len = sqrt((double) (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));

			mchIdx = (int) floor((Len-minLen)/lenIntv);
	//printf("%f %f %d  \n", Len, minLen, mchIdx);
			mchLenHist[mchIdx]++;
		}
	}

	mchMaxVal = 0;
	for (i=0; i<nBinDir; i++) {
		if (mchDirHist[i]>mchMaxVal) {
			mchMaxVal = mchDirHist[i];
			mchDirMaxIdx = i;
		}
	}
	//printf("dirmax: %d\n", mchDirMaxIdx);

	mchMaxVal = 0;
	for (i=0; i<nBinLen; i++) {
//printf("i=%d  mchLenHist[i]=%d\n", i, mchLenHist[i]);
		if (mchLenHist[i]>mchMaxVal) {
			mchMaxVal = mchLenHist[i];
			mchLenMaxIdx = i;
//printf("i=%d lenmax: %d\n", i, mchLenMaxIdx);
		}
	}
	//printf("lenmax: %d\n", mchLenMaxIdx);
	//printf("\n\n");

//printf("nMatchCoarse=%d\n", nMatchCoarse);

	mchcnt = nMatchCoarse;
	for( i = 0; i < n1; i++ ) {
		//if (Iter1->match > 0) {
		if (feat1[i].dist0 >= 0 && feat1[i].matched==1) {
//printf("--for this match:\n");
			feat = feat1+i;
			//featmatch = feat->fwd_match;

			//ddata = feat_detection_data( feat );

		//for (Iter1 = Sift1->Features.begin(); Iter1 != Sift1->Features.end(); Iter1++) {

			//o = pow(2.0,ddata->octv);
			x1 = feat->x;
			y1 = feat->y;
			//x2 = featmatch->x + wIm1;
			//y2 = featmatch->y;
			x2 = feat->fx + wIm1;
			y2 = feat->fy;

			Len = sqrt((double) (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
			mchVal = atan2((double)y2-y1, (double)x2-x1) + PI;
			mchIdx = (int) floor(((double)mchVal/PI*180-1)/(360/nBinDir));

			bool flag_trimmed = 0;

			if (mchIdx != mchDirMaxIdx) {
//printf("    trimmed for dir\n");
	//printf("[%d cnt--  %d]\n", mchIdx, mchDirMaxIdx);
				//Iter1->match = -1;
				//Iter1->flag = 0;
	//printf("   %f   %d", Iter1->match, Iter1->matchidx);
				//Iter2 = Sift2->Features.begin()+Iter1->matchidx;
	//printf("%d  %f  ", Iter1->matchidx, Iter2->match);
				//Iter2->match = -1;
				//Iter2->flag = 0;

				feat1->trimmed = 1;

				//mchcnt--;
				flag_trimmed = 1;
			}
		//printf("   ok1  \n");	
	
			mchIdx = (int) floor((Len-minLen)/lenIntv);
			mchVal = max(ceil(mchcnt*.05),1);

	//printf("  |       %d  %d  ", mchIdx, mchLenHist[mchIdx]);
			//if (Iter1->match>0 && mchLenHist[mchIdx]<mchVal) {
			if (feat1->trimmed==0 && mchIdx!=mchLenMaxIdx) {
//printf("    trimmed for len\n");
	//printf("[%d <= %f  cnt--] ", mchLenHist[mchIdx], mchVal);
				//Iter1->match = -1;
				//Iter2 = Sift2->Features.begin()+Iter1->matchidx;
				//Iter2->match = -1;
				feat1->trimmed = 1;

				//mchcnt--;
				flag_trimmed = 1;
			}
			if (flag_trimmed)
				mchcnt--;
			else {
				pt1 = cvPoint( cvRound( x1 ), cvRound( y1 ) );
				pt2 = cvPoint( cvRound( x2 ), cvRound( y2 ) );
				cvLine( stacked, pt1, pt2, CV_RGB(0,0,255), 1, 8, 0 );
			}
	
		//printf("   ok2  \n");
		//printf("\n");
		}
		
	}
	nMatchTrim = mchcnt;
//printf("nMatchTrim = %d\n", nMatchTrim);

	//printf("\n\n");
	//for (i=0; i<nBinDir; i++)
	//	printf("%d  ", mchDirHist[i]);
	//printf("\n\n");
	//for (i=0; i<nBinLen; i++)
	//	printf("%d  ", mchLenHist[i]);

//// Select best matches to be used in transformation

	////// shows best matches
	//int bestIdx[10], idx;
	//float val, bestMatches[10];


	//for (int i=0; i<10; i++)
	//	bestMatches[i] = -1;

//val = 99999;
//for (Iter = Sift1->Features.begin(); Iter != Sift1->Features.end(); Iter++)
//	if (Iter->match > 0 && Iter->match< val) 
//		printf("%f  ", Iter->match);
//printf("\n\n\n");
//for (Iter = Sift2->Features.begin(); Iter != Sift2->Features.end(); Iter++)
//	if (Iter->match > 0 && Iter->match< val) 
//		printf("%f  ", Iter->match);

	//int tmp = min(nMatchTrim, 10);

	//for (mchIdx=0; mchIdx<tmp; mchIdx++) {
	//	val = 99999;
	//	//idx = 0;

	//	for( i = 0; i < nFeat1; i++ ) {
	//	//for (Iter1 = Sift1->Features.begin(); Iter1 != Sift1->Features.end(); Iter1++) {
	//		//if (Iter1->flag && Iter1->match< val) {
	//		if (feat1[i].dist0 >= 0 && feat1[i].matched==1) {
	//			feat = feat1+i;

	//			if (feat->trimmed==0 && feat->dist0 < val) {

	//			//featmatch = feat->fwd_match;

	//			//ddata = feat_detection_data( feat );

	//				if (mchIdx==0) {
	//					bestMatches[mchIdx] = feat->dist0;
	//					val = bestMatches[mchIdx];
	//					bestIdx[mchIdx] = i;

	////printf("%d %f %f\n", mchIdx, Iter->match, bestMatches[mchIdx]);
	//				}
	//				else if (feat->dist0 > bestMatches[mchIdx-1]) {
	//					bestMatches[mchIdx] = feat->dist0;
	//					val = bestMatches[mchIdx];
	//					bestIdx[mchIdx] = i;
	//				}

	//			}
	//		}
	//		//idx++;
	//	}
	//}

////printf("\n\n\n");
////for (int i=0; i<10; i++) {
////	printf("%f  ", bestMatches[i]);
////}
//
//	idx = 0;
//	mchIdx = 0;
//	int nCrsp = 0, rectW;
//	while(bestMatches[mchIdx]!=-1 && nCrsp<=10) {
//		
//		//Iter1 = Sift1->Features.begin()+bestIdx[mchIdx];
//		//Iter1->matchBest = mchIdx+1;
//
////printf("%d\n", Iter1->matchBest); 
//		nCrsp++;
//		mchIdx++;
//	}
//
////printf("nCrsp = %f\n", nCrsp);
//
//	nCrsp = min(10, nCrsp);
////printf("nCrsp = %f\n", nCrsp);
//
//	nMatchBest = nCrsp;
//
//	//printf("\n");		// DEBUG
//	//for (mchIdx=0; mchIdx<nCrsp; mchIdx++)
//	//	printf("%d ", bestIdx[mchIdx]);
//	//printf("\n");
//
//
//	if (nCrsp >= 3) {
//
//		crspnd1.count = nCrsp;
//		crspnd2.count = nCrsp;
//
//		crspnd1.dim1 = (float*) malloc(sizeof(float)*nCrsp);
//		crspnd1.dim2 = (float*) malloc(sizeof(float)*nCrsp);
//		crspnd1.dim3 = (float*) malloc(sizeof(float)*nCrsp);
//		crspnd2.dim1 = (float*) malloc(sizeof(float)*nCrsp);
//		crspnd2.dim2 = (float*) malloc(sizeof(float)*nCrsp);
//		crspnd2.dim3 = (float*) malloc(sizeof(float)*nCrsp);
//		
//
//		//crd1.count = Sift1->Features.size();
//		//crd2.count = Sift2->Features.size();
//		crd1.count = nMatchTrim;
//		crd2.count = nMatchTrim;
//
//		crd1.dim1 = (float*) malloc(sizeof(float)*nMatchTrim);
//		crd1.dim2 = (float*) malloc(sizeof(float)*nMatchTrim);
//		crd1.dim3 = (float*) malloc(sizeof(float)*nMatchTrim);
//		crd2.dim1 = (float*) malloc(sizeof(float)*nMatchTrim);
//		crd2.dim2 = (float*) malloc(sizeof(float)*nMatchTrim);
//		crd2.dim3 = (float*) malloc(sizeof(float)*nMatchTrim);
//
//		// copy sift points to crd1 and crd2 for getMatchLSF()
//		idx = 0; 
//		//for (Iter1 = Sift1->Features.begin(); Iter1 != Sift1->Features.end(); Iter1++) {
//		for( i = 0; i < nFeat1; i++ ) {
//			feat = feat1+i;
//			if (feat->dist0>0 && feat1[i].matched==1 && feat->trimmed == 0) {
//				//ddata = feat_detection_data( feat );
//				//o = pow(2.0,ddata->octv);
//
//				// copy cord1
//				//if (Iter1->match > 0) {
//				crd1.dim1[idx] = feat->x; 
//				crd1.dim2[idx] = feat->y; 
//				crd1.dim3[idx] = 0;
//
//				crd2.dim1[idx] = feat->fx; 
//				crd2.dim2[idx] = feat->fy; 
//				crd2.dim3[idx] = 0;
//
//				//printf("[crd, %d, %f %f   %f %f]\n", idx,  feat->x, feat->y, feat->fx, feat->fy);
//
//				idx++;
//			}
//		}
//		//printf("\n\n");
//		//// copy crd2
//		//idx = 0;
//		////for (Iter2 = Sift2->Features.begin(); Iter2 != Sift2->Features.end(); Iter2++) {
//		//for( i = 0; i < nFeat2; i++ ) {
//		//	//if (Iter2->match > 0) {
//		//	feat = feat2+i;
//
//		//	if (feat->dist0>0 && feat1[i].matched==1 && feat->trimmed == 0) {
//		//		//ddata = feat_detection_data( feat );
//		//		//o = pow(2.0,ddata->octv);
//
//		//		crd2.dim1[idx] = feat->x; 
//		//		crd2.dim2[idx] = feat->y; 
//		//		crd2.dim3[idx] = 0;
//
//		//		printf("[feat2, %d, %f, %f] ", idx,  feat->x, feat->y);
//
//		//		idx++;
//		//	}
//		//}
//		// copy crspnd1 and crspnd2
//		for (i=0; i<nCrsp; i++) {
//			feat = feat1 + bestIdx[i]; 
//			//ddata = feat_detection_data( feat );
//			//o = pow(2.0,ddata->octv);
//
//			crspnd1.dim1[i] = feat->x; 
//			crspnd1.dim2[i] = feat->y;
//			crspnd1.dim3[i] = 0;
////printf("hello\n");
////printf("%f ", crspnd1.dim1[i]);	// DEBUG
//
//			//ddata = feat_detection_data( feat->fwd_match );
//			//o = pow(2.0,ddata->octv);
//
//			//crspnd2.dim1[i] = feat->fwd_match->x; 
//			//crspnd2.dim2[i] = feat->fwd_match->y;
//			crspnd2.dim1[i] = feat->fx;
//			crspnd2.dim2[i] = feat->fy;
//			crspnd2.dim3[i] = 0;
//
//			//printf("[crspnd, %d, %f %f   %f %f]\n", i,  feat->x, feat->y, feat->fx, feat->fy);
//		}
//	}

//printf("nCrsp = %d\n", nCrsp);

	//return nCrsp;
	return nMatchTrim;
}

void getMask (unsigned char *mask, IplImage *img) 
{
	int r, c;
	int curRR = img->height;
	int curCC = img->width;

	//float *tmp = (float*) malloc(sizeof(float)*curRR*curCC);
	float *tmp = new float[curRR*curCC];

	char *t = img->imageData;
	for (r = 0; r<curRR; r++) {
		for (c=0; c<curCC; c++) {
			baseIm[r*curCC+c] = (float) (t[(r*curCC+c)*3]+t[(r*curCC+c)*3+1]+t[(r*curCC+c)*3+2])/3;
			//baseIm[r*curCC+c] = (float) t[r*curCC+c];
		}
	}

	edgeSobel(baseIm, tmp, NULL, 3, 50, curRR, curCC);
	Erode(tmp, tmp, 3, curRR, curCC);
	Dilate(tmp, tmp, 9, curRR, curCC);
	Erode(tmp, tmp, 9, curRR, curCC);
	fillMiddle(tmp, tmp, curRR, curCC);
	Erode(tmp, tmp, 19, curRR, curCC);

	//printf("%d, %d\n", img->depth, img->nChannels);
	//IplImage *im = cvCreateImage(cvSize(curCC, curRR), IPL_DEPTH_8U, 1);
	//char* t2 = im->imageData;

	for (int r=0; r<curRR; r++) {
		for (int c=0; c<curCC; c++) {
			if (tmp[r*curCC+c])
				mask[r*curCC+c]=1;	// valid
			else
				mask[r*curCC+c]=0;	// invalid

			//t2[r*curCC+c] = mask[r*curCC+c]*255;
		}
	}

	//cvSaveImage ("C:/temp/org.jpg", img);
	//cvSaveImage ("C:/temp/1.jpg", im);

	//free (tmp);
	delete [] tmp;
	
}

int edgeSobel(float *img, float *sbMag, float *sbDir, int w, float thresh, int curRR, int curCC)
{
	w = 3; // w will be always 3 in sobel operation

	// dim1: height, dim2: width
	const int rr = curRR, cc = curCC;

	//Array2D<float> sobelX(w,w), sobelY(w,w);
	float *sobelX = (float*) malloc(sizeof(float)*curRR*curCC);
	float *sobelY = (float*) malloc(sizeof(float)*curRR*curCC);

	//Array2D<int> gray(rr, cc);

	int r, c, i, j, que;
	float maxi, mini;

	//if (sbMag.dim1() != cc || sbMag.dim2() != rr)
	//	sbMag.SetSize(rr, cc);
	//if (sbDir.dim1() != cc || sbDir.dim2() != rr)
	//	sbDir.SetSize(rr, cc);

	que = (w-1)/2;
	//s = que/(2*sqrt((float)2));

	sobelX[0*w+0] = -1; sobelX[0*w+1] = 0; sobelX[0*w+2] = 1;
	sobelX[1*w+0] = -2; sobelX[1*w+1] = 0; sobelX[1*w+2] = 2;
	sobelX[2*w+0] = -1; sobelX[2*w+1] = 0; sobelX[2*w+2] = 1;

	sobelY[0*w+0] = 1; sobelY[0*w+1] = 2; sobelY[0*w+2] = 1;
	sobelY[1*w+0] = 0; sobelY[1*w+1] = 0; sobelY[1*w+2] = 0;
	sobelY[2*w+0] = -1; sobelY[2*w+1] = -2; sobelY[2*w+2] = -1;

	//Array2D<float> ImX(rr, cc), ImY(rr, cc);
	float *ImX = (float*) malloc(sizeof(float)*curRR*curCC);
	float *ImY = (float*) malloc(sizeof(float)*curRR*curCC);

	maxi = -1 * 99999;
	mini = 99999;

	/// convolve input image with Sobel operator
	for(r=0; r<rr; r++) {
		for(c=0; c<cc; c++) {
			if(c<que || c>(cc-que-1) || r<que || r>(rr-que-1)) {
				ImX[r*curCC+c] = 0;
				ImY[r*curCC+c] = 0;
			}
			else {
				ImX[r*curCC+c] = 0;
				ImY[r*curCC+c] = 0;
				for(i=0; i<w; i++) {
					for(j=0; j<w; j++){
						ImX[r*curCC+c] = ImX[r*curCC+c] + (sobelX[i*w+j])*(img[(r+que-i)*cc+c+que-j]);
						ImY[r*curCC+c] = ImY[r*curCC+c] + (sobelY[i*w+j])*(img[(r+que-i)*cc+c+que-j]);
					}
				}
			}
		}
	}
	for(r=0; r<rr; r++) {
		for(c=0; c<cc; c++) {
			sbMag[r*cc+c] = sqrt(square(ImX[r*cc+c])+square(ImY[r*cc+c]));
			if (sbMag[r*cc+c]<thresh) 
				sbMag[r*cc+c] = 0;
		}
	}
	if (sbDir!=NULL) {
		for(r=0; r<rr; r++) {
			for(c=0; c<cc; c++) {
				sbDir[r*cc+c] = atan2(ImY[r*cc+c], ImX[r*cc+c]);
			}
		}
	}

	// Fixed leak -RPB
	free(sobelX);
	// Fixed leak -RPB
	free(sobelY);
	// Fixed leak -RPB
	free(ImX);
	// Fixed leak -RPB
	free(ImY);

	return 1;
}

void fillMiddle(float* pGray, float* out, int curRR, int curCC)
{
	int r,c;
	//unsigned char* Bin = (unsigned char*) malloc(sizeof(unsigned char)*curRR*curCC);
	//unsigned char* Bin = new unsigned char[curRR*curCC];
	for (r=0; r<curRR; r++) {
		for (c=0; c<curCC; c++) {
			Bin[r*curCC+c] = 0;
		}
	}

	//int* lBound = (int*) malloc(sizeof(int)*curRR);
	//int* rBound = (int*) malloc(sizeof(int)*curRR);

	//int* lBound = new int[curRR];
	//int* rBound = new int[curRR];

	for (r=0; r<curRR; r++) {
		lBound[r] = 0;
		rBound[r] = 0;
	}
//rBound[369] = 0;
	int stop;

	for (r=0; r<curRR; r++) {
		stop = 0;
		while (pGray[r*curCC+stop] == 0 && stop<curCC)
			stop++;
		lBound[r] = min(curCC-1,stop);

		stop = curCC-1;
		while (pGray[r*curCC+stop] == 0 && stop>=0)
			stop--;
//printf("%d %d %f\n", r, stop, pGray[r*curCC+stop]);
		rBound[r] = max(0,stop);
//if (r==368)
//	printf("");
//printf("%d %d  %d\n", r, lBound[r], rBound[r]);
	}
	for (r=0; r<curRR; r++) {
		for (c=lBound[r]; c<rBound[r]; c++)
			Bin[r*curCC+c] = 1;
	}

	for (r=0; r<curRR; r++) 
		for (c=0; c<curCC; c++) 
			out[r*curCC+c] = Bin[r*curCC+c];

	//free(Bin);
	//free(lBound);
	//free(rBound);

	//delete [] Bin;
	//delete [] lBound;
	//delete [] rBound;
}

float getMatchLSF(Coord &crspnd1, Coord &crspnd2, Coord &crd1, Coord &crd2)
{
	//float ** transform;
	Coord crd1Trans;

	//transform = deriveTransformLSF_G(crspnd1, crspnd2);		///////

//for (int i=0; i<crspnd1.count; i++) {	// DEBUG
//	printf("%f ", crspnd1.dim1[i]);
//	printf("%f \n", crspnd1.dim2[i]);
//}
//printf("\n"); printf("\n");
//for (int i=0; i<crspnd2.count; i++) {	// DEBUG
//	printf("%f ", crspnd2.dim1[i]);
//	printf("%f \n", crspnd2.dim2[i]);
//}

// t1 r11 r12
// t2 r21 r22
//  0   0   0
//printf("\n");
//for (int i=0; i<2; i++) {		// DEBUG
//	for (int j=0; j<3; j++) {
//		printf("%f ", transform[i][j]);
//	}
//	printf("\n");
//}

	//for (int i=0; i<crspnd1.count; i++)		// DEBUG
	//	printf("%f  %f  |  %f  %f\n", crspnd1.dim1[i], crspnd1.dim2[i], crspnd2.dim1[i], crspnd2.dim2[i]);
	//printf("\n");
	//crd1Trans.count = crspnd1.count;		
	//crd1Trans.dim1 = (float*) malloc(sizeof(float)*crd1Trans.count);
	//crd1Trans.dim2 = (float*) malloc(sizeof(float)*crd1Trans.count);
	//crd1Trans.dim3 = (float*) malloc(sizeof(float)*crd1Trans.count);
	//applyTransform(transform, crspnd1, crd1Trans);	
	//for (int i=0; i<crspnd1.count; i++)
	//	printf("%f  %f  |  %f  %f\n", crd1Trans.dim1[i], crd1Trans.dim2[i], crspnd2.dim1[i], crspnd2.dim2[i]);

	//crd1Trans.count = crd1.count;

	//applyTransform(transform, crd1, crd1Trans);

	//float sco = getFit(crd1Trans, crd2);

	//crd1Trans.count = crspnd1.count;
	//crd1Trans.dim1 = (float*) malloc(sizeof(float)*crd1Trans.count);
	//crd1Trans.dim2 = (float*) malloc(sizeof(float)*crd1Trans.count);
	//crd1Trans.dim3 = (float*) malloc(sizeof(float)*crd1Trans.count);
//printf("%d corresponds\n", crspnd1.count);

	// errors in the following two routines, but now it's done with the transformations matrix
	//applyTransform(transform, crspnd1, crd1Trans);

	//float sco = getFit(crd1Trans, crd2);

	//return sco;
	return 1;
}

//probePixels: 3xN
//referencePixels: 3xN
//float ** deriveTransformLSF(SelectedPixels & probePixels, SelectedPixels & referencePixels);

//extern "C" void tred2(float **a, int n, float d[], float e[]);
//extern "C" void tqli(float d[], float e[], int n, float **z);

////float ** deriveTransformLSF_G(SelectedPixels & probePixels, SelectedPixels & referencePixels)
//float ** deriveTransformLSF_G(struct Coord & probePixels, struct Coord & referencePixels)
//{
//	// up and ux are the means of the r, g, and b channels.
//	float up[3]={0,0,0};
//	float ux[3]={0,0,0};
//
//	//printf("cnt=%d cnt=%d\n", probePixels.count, referencePixels.count);
//
////	float * probeArray[3]={probePixels.b,probePixels.g, probePixels.r};
////	float * referenceArray[3]={referencePixels.b, referencePixels.g, referencePixels.r};
//
//	float * probeArray[3]={probePixels.dim1, probePixels.dim2, probePixels.dim3};
//	float * referenceArray[3]={referencePixels.dim1, referencePixels.dim2, referencePixels.dim3};
//
//
//
//
//	for(unsigned i=0;i<probePixels.count;i++)
//	{
//		for(unsigned j=0;j<3;j++)
//		{
//			up[j]+=probeArray[j][i];
//			ux[j]+=referenceArray[j][i];
//		}
//	}
//
//	for(unsigned i=0;i<3;i++)
//	{
//		up[i]/=probePixels.count;
//		ux[i]/=probePixels.count;
//	}
//
//
//#if DEBUG_OUT
//	std::cout << "up is: ";
//	for(unsigned i=0;i<3;i++)
//	{
//		std::cout <<up[i] << '\t';
//	}
//	std::cout << std::endl;
//
//	std::cout << "ux is: ";
//	for(unsigned i=0;i<3;i++)
//	{
//		std::cout << ux[i] << '\t';
//	}
//	std::cout << std::endl;
//#endif
//
//
//	float cv[3][3]={{0,0,0},
//					{0,0,0},
//					{0,0,0} };
//
//
//
//	for(unsigned i=0;i<probePixels.count;i++)
//	{
//
//		for(unsigned j=0;j<3;j++)
//		{
//			for(unsigned k=0;k<3;k++)
//			{
//				cv[k][j]+=referenceArray[j][i]*probeArray[k][i];
//			}
//		}
//	}
//
//	for(unsigned i=0;i<3;i++)
//	{
//		for(unsigned j=0;j<3;j++)
//		{
//			cv[i][j]/=probePixels.count;
//		}
//	}
//#if DEBUG_OUT
//	std::cout << "pre-sub cv is:" << std::endl;
//	//floatOut<3>(cv);
//#endif
//
//	float up_ux[3][3]={ {0,0,0},
//						{0,0,0},
//						{0,0,0} };
//
//
//	for(unsigned i=0;i<3;i++)
//	{
//		for(unsigned j=0;j<3;j++)
//		{
//			up_ux[i][j]=up[i]*ux[j];
//			cv[i][j]-=up_ux[i][j];
//		}
//	}
//#if DEBUG_OUT
//	std::cout << "cv is:" << std::endl;
//	//floatOut<3>(cv);
//#endif
//
//	float A[3][3]={ {0,0,0},
//					{0,0,0},
//					{0,0,0} };
//
//	for(unsigned i=0;i<3;i++)
//	{
//		for(unsigned j=0;j<3;j++)
//		{
//			A[i][j]=cv[i][j]-cv[j][i];
//		}
//	}
//
//	//for(unsigned i=0;i<3;i++)
//	//{
//	//	for(unsigned j=0;j<3;j++)
//	//	{
//	//		printf("%f ", A[i][j]);
//	//	}
//	//	printf("\n");
//	//}
//
//#if DEBUG_OUT
//	std::cout << "A is:" << std::endl;
//	//floatOut<3>(A);
//#endif
//
//	float delta[3]={A[1][2],A[2][0],A[0][1]};
//
//#if DEBUG_OUT
//	std::cout << "delta is: " << std::endl;
//	//floatOut(delta);
//#endif
//
//	float traceCV=0;
//
//	for(unsigned i=0;i<3;i++)
//	{
//		traceCV+=cv[i][i];
//	}
//
//#if DEBUG_OUT
//	std::cout << "trace cv is: " << traceCV << std::endl << std::endl;
//	for(unsigned i=0;i<3;i++) {
//		for(unsigned j=0;j<3;j++) {
//			printf("%f  ", cv[i][j]);
//		}
//		printf("\n");
//	}
//#endif
//
//	float Q[4][4]= {{traceCV,delta[0],delta[1],delta[2]},
//					{delta[0],		0,0,0},
//					{delta[1],		0,0,0}, 
//					{delta[2],		0,0,0}};
//
//
//
//	for(unsigned i=1;i<4;i++)
//	{
//		for(unsigned j=1;j<4;j++)
//		{
//			Q[i][j]=cv[i-1][j-1]+cv[j-1][i-1];
//			if(i==j)
//			{
//				Q[i][j]-=traceCV;
//			}
//		}
//	}
//
//	//for(unsigned i=1;i<4;i++)
//	//	for(unsigned j=1;j<4;j++)
//	//		printf("%f ", Q[i][j]);
//
//	
//
//#if DEBUG_OUT
//	std::cout << "Q is:" << std::endl;
//	//floatOut<4>(Q);
//	for(unsigned i=1;i<4;i++) {
//		for(unsigned j=1;j<4;j++) {
//			printf("%f  ", Q[i][j]);
//		}
//		printf("\n");
//	}
//#endif
//
//	//float ** inputArray = (float**) malloc(sizeof(float*)*5);
//	//for(unsigned i=0;i<5;i++)
//	//	inputArray[i] = (float*) malloc(sizeof(float)*5);
//	//float inputArray2[5][5];
//	//float **inputArray = inputArray2;
//
//	for(unsigned i=1;i<5;i++)
//		for(unsigned j=1;j<5;j++)
//			inputArray[i][j]=Q[i-1][j-1];
//
//	//float ** inputArray=new float*[5];
//	//for(unsigned i=0;i<5;i++)
//	//{
//	//	inputArray[i]=new float[5];
//	//}
//	//for(unsigned i=1;i<5;i++)
//	//	for(unsigned j=1;j<5;j++)
//	//		inputArray[i][j]=Q[i-1][j-1];
//
//	//float ** inputArray=new float*[5];
//	//for(unsigned i=1;i<5;i++)
//	//{
//	//	inputArray[i]=new float[5];
//	//	for(unsigned j=1;j<5;j++)
//	//	{
//	//		inputArray[i][j]=Q[i-1][j-1];
//	//	}
//	//}
//
//	//float * d=new float[5];
//	//float * e=new float[5];
//
//float val;
//for(unsigned i=1;i<5;i++)
//{
//	for(unsigned j=1;j<5;j++)
//	{
//		val = inputArray[i][j];
//	}
//}
//
//
//	tred2(inputArray,4, d, e);
//
//	tqli(d, e,4, inputArray);
//
//
//for(unsigned i=1;i<5;i++)
//{
//	for(unsigned j=1;j<5;j++)
//	{
//		val = inputArray[i][j];
//	}
//}
//
//	int maxIndex=-1;
//	float maxVal=0;
//
//	
//	for(unsigned i=1;i<5;i++)
//	{
//		if(d[i] > maxVal)
//		{
//			maxVal=d[i];
//			maxIndex=i;
//		}		
//	}
//	
//#if DEBUG_OUT
//	std::cout << "Eigenvalues are: ";
//	for(unsigned i=1;i<5;i++)
//	{
//		std::cout << d[i] << '\t';
//	}
//	std::cout << std::endl << std::endl;
//
//
//	std::cout << "eigen vectors are:\n";
//
//	for(unsigned i=1;i<5;i++)
//	{
//		for(unsigned j=1;j<5;j++)
//		{
//			std::cout << inputArray[i][j] << '\t';
//		}
//		std::cout << std::endl;
//	}
//	std::cout << std::endl;
//#endif
//
//	float q0 = inputArray[1][maxIndex];
//	float q1 = inputArray[2][maxIndex];
//	float q2 = inputArray[3][maxIndex];
//	float q3 = inputArray[4][maxIndex];
//
//#if DEBUG_OUT
//	std::cout << "max-valued vector is " << q0 << '\t' << q1 << '\t' << q2 << '\t' << q3 << std::endl << std::endl;
//#endif
//
//
//	float R[3][3]={ {q0*q0+q1*q1-q2*q2-q3*q3,     2*(q1*q2-q0*q3),         2*(q1*q3+q0*q2)},
//					{2*(q1*q2+q0*q3),        q0*q0+q2*q2-q1*q1-q3*q3,     2*(q2*q3-q0*q1)},
//					{2*(q1*q3-q0*q2),        2*(q2*q3+q0*q1),         q0*q0+q3*q3-q1*q1-q2*q2}};
//	
//
//#if DEBUG_OUT
//	std::cout << "R is:\n";
//	//floatOut<3>(R);
//	for(unsigned i=0;i<3;i++) {
//		for(unsigned j=0;j<3;j++) {
//			printf("%f  ", R[i][j]);
//		}
//		printf("\n");
//	}
//#endif
//
//
//	float qt[3]={0,0,0};
//
//	for(unsigned i=0;i<3;i++)
//	{
//		qt[i]=ux[i];
//		for(unsigned j=0;j<3;j++)
//		{
//			qt[i]-=R[i][j]*up[i];
//		}
//	}
//
//#if DEBUG_OUT
//	std::cout << "qt is: ";
//	//floatOut(qt);
//	for(unsigned j=0;j<3;j++) {
//		printf("%f  ", qt[j]);
//	}
//#endif
//
//	
//
//	for(unsigned i=0;i<3;i++)
//	{
//		transform[i][0]=qt[i];
//
//		for(unsigned j=0;j<3;j++)
//		{
//			transform[i][j+1]=R[i][j];
//		}
//	}
//
//
//	//float ** transform=new float*[3];
//
//	//for(unsigned i=0;i<3;i++)
//	//{
//	//	transform[i]=new float[4];
//
//	//	transform[i][0]=qt[i];
//
//	//	for(unsigned j=0;j<3;j++)
//	//	{
//	//		transform[i][j+1]=R[i][j];
//	//	}
//	//}
//
//#if DEBUG_OUT
//	std::cout << "created transform" << std::endl;
//#endif
//
//
//	//for(unsigned i=0;i<5;i++)
//	//{
//	//	delete [] inputArray[i];
//	//}
//	//delete [] inputArray;
//
//	//for(unsigned i=0;i<5;i++)
//	//	free(inputArray[i];
//	//free(inputArray);
//
//	//delete [] d;
//	//delete [] e;
//
//	return transform;
//}

int GetDirFiles(const char *path_name, CString *lst)
{
	WIN32_FIND_DATA FN;
	HANDLE hFind;
	char search_arg[MAX_PATH], new_file_path[MAX_PATH];
	CString cs;
	sprintf(search_arg, "%s\\*.*", path_name);
	int i;

	hFind = FindFirstFile( (LPCTSTR) search_arg, &FN);
	//cout << (hFind != INVALID_HANDLE_VALUE) << endl << FN.cFileName << endl;
	if (hFind != INVALID_HANDLE_VALUE) {
		i = 0;
		do {
			cs = FN.cFileName;
			if (cs != "." && cs != "..") {
				//cout << i << "  " << cs << endl;
				lst[i++] = cs;
			}
			//// Make sure that we don't process . or .. in FN.cFileName here.
			//if (strcmp(FN.cFileName, ".") != 0 && strcmp(FN.cFileName, "..") != 0) {
			//	sprintf(new_file_path, "%s\\%s", path_name, FN.cFileName);
			//// If this is a directory then recurse into the directory
			//if (FN.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
			//	GetDirFiles(new_file_path, flist);
			//else
			//// Do something here with the file
			//// new_file_path contains the filename with complete path.
			//;
			//}
		} while (FindNextFile(hFind, &FN) != 0);
		if (GetLastError() == ERROR_NO_MORE_FILES)
		FindClose(hFind);
	}
	//for (i=0; i<100; i++)
	//	cout << lst[i] << endl;
	return i;
}

int GetDirFilesAscend(CString *lst, CString *lst2, int n)
{
	int nList[15000], nList2[15000], num, i, j, mIdx, mVal;
	
	for (i=0; i<n; i++)
		nList[i] = atoi(lst[i].Left(4));
	
	for (i=0; i<n; i++) {
		mIdx = i;
		mVal = nList[i];

		for (j=0; j<n; j++) {
			if (mVal > nList[j]) {
				mIdx = j;
				mVal = nList[j];
			}
		}
		nList[mIdx] = 99999;
		nList2[i] = mIdx;
	}

	for (i=0; i<n; i++)
		lst2[i] = lst[nList2[i]];

	return 1;
}

int Dilate(float* pGray, float* out, unsigned char kernel, int curRR, int curCC)	// Dilate
{
	unsigned int h_kernel, r, c, rr, cc, r1, r2, c1, c2;
	h_kernel = (kernel/2);

	//unsigned char* Bin = (unsigned char*) malloc(sizeof(unsigned char)*curRR*curCC);
	//unsigned char* Bin = new unsigned char[curRR*curCC];

	for (r=0; r<curRR; r++) {
		for (c=0; c<curCC; c++) {
			Bin[r*curCC+c] = 0;
		}
	}

	r1 = max(h_kernel,0);
	r2 = min(curRR-h_kernel,curRR);
	c1 = max(h_kernel,0);
	c2 = min(curCC-h_kernel,curCC);

	for (r=r1; r<r2; r++) {
		for (c=c1; c<c2; c++) {
			if (pGray[r*curCC+c] > 0.00001) {
				rr = r-h_kernel;
				while (rr<r+h_kernel) {
					cc = c-h_kernel;
					while (cc<c+h_kernel) {
						Bin[rr*curCC+cc] = 1;
						cc++;
					}
					rr++;
				}
			}
		}
	}	
	for (r=0; r<curRR; r++) 
		for (c=0; c<curCC; c++) 
			out[r*curCC+c] = Bin[r*curCC+c];

	//out = pGray;

	//free(Bin);
	//delete [] Bin;

	return 1;
}

int Erode(float* pGray, float* out, unsigned char kernel, int curRR, int curCC)	// Erode
{
	unsigned int h_kernel, r, c, rr, cc, r1, r2, c1, c2;
	bool flag;
	h_kernel = (kernel/2);

	//unsigned char* Bin = (unsigned char*) malloc(sizeof(unsigned char)*curRR*curCC);
	//unsigned char* Bin = new unsigned char[curRR*curCC];
	for (r=0; r<curRR; r++) {
		for (c=0; c<curCC; c++) {
			Bin[r*curCC+c] = 0;
		}
	}

	r1 = max(h_kernel,0);
	r2 = min(curRR-h_kernel,curRR);
	c1 = max(h_kernel,0);
	c2 = min(curCC-h_kernel,curCC);

	for (r=r1; r<=r2; r++) {
		for (c=c1; c<=c2; c++) {
			flag = 1;
			rr = r-h_kernel;
			while (rr<r+h_kernel && flag) {
				cc = c-h_kernel;
				while (cc<c+h_kernel && flag) {
					if (pGray[rr*curCC+cc] < 0.00001) {
						flag=0;
					}
					cc++;
				}
				rr++;
			}
			if (flag)
				Bin[r*curCC+c] = 1;
		}
	}
	for (r=0; r<curRR; r++) 
		for (c=0; c<curCC; c++) 
			out[r*curCC+c] = Bin[r*curCC+c];

	//out = pGray;

	//free(Bin);
	//delete [] Bin;

	return 1;
}

/********************************************************************************/
//	tic
//
//	Measure the starting time
/********************************************************************************/
void tic(char* str)		// Starting point of measuring time
{
	QueryPerformanceCounter(&Start);
	cout << str << endl;
}

/********************************************************************************/
//	toc
//
//	Measure the finishing time and dispaly
/********************************************************************************/
void toc(char* str)	// Finishing point of measuring time, display result
{
	QueryPerformanceCounter(&End);
	QueryPerformanceFrequency(&Frequency);

	TimeTaken = (double) 1000*(End.QuadPart-Start.QuadPart)/Frequency.QuadPart;		// milliseconds

	if (TimeTaken < 1000)
		printf("%s: %5.2f msec\n", str, TimeTaken);
	else if (TimeTaken < 60000)
		printf("%s: %5.2f sec\n", str, TimeTaken/1000);
	else
		printf("%s: %10.2f min\n", str, TimeTaken/(1000*60));
	//else
	//	printf("%s: %f sec", str, TimeTaken/1000);
	
//cout << str << ":		" << TimeTaken << " ms" << endl;

	//CString cs;
	//CClientDC pDC(this);
	//m_Str.Format(" %3.2f     ", result);
	//CString Str(str);
	//Str += m_Str;
	//
	//pDC.TextOut(x, y, Str);
	
}