
// HW2Dlg.cpp : 實作檔
//

#include "stdafx.h"
#include "HW2.h"
#include "HW2Dlg.h"
#include "afxdialogex.h"
#include <iostream>
#include "opencv2\highgui\highgui.hpp"
#include "opencv/cv.h"
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
using namespace std;
using namespace cv;
#ifdef _DEBUG
#define new DEBUG_NEW
#endif

#pragma warning( disable : 4996 )

// 對 App About 使用 CAboutDlg 對話方塊

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 對話方塊資料
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支援

// 程式碼實作
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CHW2Dlg 對話方塊



CHW2Dlg::CHW2Dlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_HW2_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CHW2Dlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_COMBO3, COMBO1);
}

BEGIN_MESSAGE_MAP(CHW2Dlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON1, &CHW2Dlg::OnBnClickedButton1)
	ON_BN_CLICKED(IDC_BUTTON2, &CHW2Dlg::OnBnClickedButton2)
	ON_BN_CLICKED(IDC_BUTTON3, &CHW2Dlg::OnBnClickedButton3)
	ON_BN_CLICKED(IDC_BUTTON4, &CHW2Dlg::OnBnClickedButton4)
	ON_BN_CLICKED(IDC_BUTTON5, &CHW2Dlg::OnBnClickedButton5)
	ON_BN_CLICKED(IDC_BUTTON6, &CHW2Dlg::OnBnClickedButton6)
	ON_BN_CLICKED(IDC_BUTTON7, &CHW2Dlg::OnBnClickedButton7)
	ON_BN_CLICKED(IDC_BUTTON8, &CHW2Dlg::OnBnClickedButton8)
	ON_BN_CLICKED(IDC_BUTTON9, &CHW2Dlg::OnBnClickedButton9)
	ON_BN_CLICKED(IDC_BUTTON10, &CHW2Dlg::OnBnClickedButton10)
END_MESSAGE_MAP()


// CHW2Dlg 訊息處理常式

BOOL CHW2Dlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 將 [關於...] 功能表加入系統功能表。

	// IDM_ABOUTBOX 必須在系統命令範圍之中。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 設定此對話方塊的圖示。當應用程式的主視窗不是對話方塊時，
	// 框架會自動從事此作業
	SetIcon(m_hIcon, TRUE);			// 設定大圖示
	SetIcon(m_hIcon, FALSE);		// 設定小圖示

	// TODO: 在此加入額外的初始設定
	AllocConsole();
	freopen("CONOUT$", "w", stdout);


	return TRUE;  // 傳回 TRUE，除非您對控制項設定焦點
}

void CHW2Dlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 如果將最小化按鈕加入您的對話方塊，您需要下列的程式碼，
// 以便繪製圖示。對於使用文件/檢視模式的 MFC 應用程式，
// 框架會自動完成此作業。

void CHW2Dlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 繪製的裝置內容

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 將圖示置中於用戶端矩形
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 描繪圖示
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// 當使用者拖曳最小化視窗時，
// 系統呼叫這個功能取得游標顯示。
HCURSOR CHW2Dlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

//====================================1==========================

void drawHistImg(const Mat &src, Mat &dst) {
	int histSize = 256;
	float histMaxValue = 0;
	for (int i = 0; i<histSize; i++) {
		float tempValue = src.at<float>(i);
		if (histMaxValue < tempValue) {
			histMaxValue = tempValue;
		}
	}

	float scale = (0.9 * 256) / histMaxValue;
	for (int i = 0; i<histSize; i++) {
		int intensity = static_cast<int>(src.at<float>(i)*scale);
		line(dst, Point(i, 255), Point(i, 255 - intensity), Scalar(0, 0,255),1);
	}
}

void CHW2Dlg::OnBnClickedButton1()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	Mat src = imread("images/plant.jpg", 0);
	int histSize = 256;
	float range[] = { 0, 255 };
	const float* histRange = { range };
	Mat histImg;
	calcHist(&src, 1, 0, Mat(), histImg, 1, &histSize, &histRange);

	Mat showHistImg(256,260, CV_8UC3, Scalar(255,255,255)); 
	drawHistImg(histImg, showHistImg);
	imshow("1.1src", src);
	imshow("1.1his", showHistImg);
	waitKey(0);
}


void CHW2Dlg::OnBnClickedButton2()
{
	// TODO: 在此加入控制項告知處理常式程式碼

	// TODO: 在此加入控制項告知處理常式程式碼
	Mat src = imread("images/plant.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	int histSize = 256;
	float range[] = { 0, 255 };
	const float* histRange = { range };
	Mat histImg;

	equalizeHist(src, src);
	calcHist(&src, 1, 0, Mat(), histImg, 1, &histSize, &histRange);

	Mat showHistImg(256, 260, CV_8UC3, Scalar(255, 255, 255));
	drawHistImg(histImg, showHistImg);
	imshow("1.2src", src);
	imshow("1.2his", showHistImg);
	waitKey(0);

}

//=====================3=============================
/***********************3.1**********************/
void CHW2Dlg::OnBnClickedButton4()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	int size = 512;
	Mat src[15];
	Mat img[15];
	Mat show[15];
	char* filename = (char*)malloc(100),*tmp= (char*)malloc(100);
	

	for(int i =0;i<15;i++){
		sprintf(filename, "images/CameraCalibration/");
		sprintf(tmp, "%d.bmp", i+1);
		strcat(filename, tmp);
		src[i] = imread(filename, 0);
		img[i] = imread(filename, 1);
		//corner(&src[i],&img[i]);
		int imagewidth = 2048;
		int imageheight = 2048;
		std::vector<std::vector<cv::Point2f>>imagePoints;
		int key;
		cv::Size boardSize(11, 8);
		cv::Size imageSize(imagewidth, imageheight);
		std::vector<cv::Point2f> imageCorners;
		bool found = cv::findChessboardCorners(src[i], boardSize, imageCorners);

		cornerSubPix(src[i], imageCorners, cv::Size(5, 5), cv::Size(-1, -1),
			cv::TermCriteria(cv::TermCriteria::MAX_ITER +
				cv::TermCriteria::EPS,
				30,      // max number of iterations
				0.1));   // min accuracy

		if (imageCorners.size() == boardSize.area())
		{
			imagePoints.push_back(imageCorners);
		}
		//img.copyTo(img2);
		drawChessboardCorners(img[i], boardSize, imageCorners, found);
		//Mat tmp;
		//resize(img2, tmp, Size(img2.rows*0.5, img2.cols*0.5));
		//imshow("image", tmp);
		//key = cvWaitKey(400);
		//====================
		resize(img[i], show[i], Size(img[i].cols*0.4, img[i].rows*0.4));//Resize 方便看
		cvNamedWindow(tmp);
		//resizeWindow(tmp, size, size);
		imshow(tmp, show[i]);
		key = cvWaitKey(5);
	}

	cvWaitKey(0);
	
	//cvReleaseImage(&img);

	//cvReleaseImage(&src);
	//cvDestroyWindow("3.1");
}

/*******************3.2*******************************/

int imagewidth = 2048;
int imageheight = 2048;

cv::Mat cameraMatrix;
cv::Mat distCoeffs;
vector<Mat> rvecs, tvecs;

Mat rotation_side;
Mat translation_side;
Mat extrinsicMat_side;
boolean IS3_1 = false;
boolean p2_1 = false;
CvMat* intrinsic_matrix = cvCreateMat(3, 3, CV_32FC1);
CvMat* distortion_coeffs = cvCreateMat(5, 1, CV_32FC1);

//=============修改
std::vector<std::vector<cv::Point3f>> objectPoints;
std::vector<std::vector<cv::Point2f>> imagePoints;

void CHW2Dlg::OnBnClickedButton5()
{
	int imagewidth = 2048;
	int imageheight = 2048;

	std::vector<cv::Point2f> imageCorners;
	std::vector<cv::Point3f> objectCorners;

	cv::Mat map1, map2;

		cv::Mat img;
		cv::Mat img2;

		int key;
		char filename[256];

		int i;
		cv::Size boardSize(11, 8);
		cv::Size imageSize(imagewidth, imageheight);

		// The corners are at 3D location (X,Y,Z)= (i,j,0)
		for (int i = 0; i<boardSize.height; i++)
		{
			for (int j = 0; j<boardSize.width; j++)
			{
				objectCorners.push_back(cv::Point3f(i, j, 0.0f));
			}
		}

		for (int i = 1; i <= 15; i++) {
			// find file:
			char name[100];
			sprintf(filename, "./images/CameraCalibration/%d", i);
			strcat(filename, ".bmp");
			
			printf("IMG = %s\n", filename);

			img = imread(filename, 0);
			img2 = imread(filename, 1);
			//imshow("s", img);

			key = cvWaitKey(400);
				bool found = cv::findChessboardCorners(
					img, boardSize, imageCorners);

				cornerSubPix(img, imageCorners,
					cv::Size(5, 5),
					cv::Size(-1, -1),
					cv::TermCriteria(cv::TermCriteria::MAX_ITER +
						cv::TermCriteria::EPS,
						30,      // max number of iterations
						0.1));   // min accuracy
				if (imageCorners.size() == boardSize.area())
				{
					imagePoints.push_back(imageCorners);
					objectPoints.push_back(objectCorners);
				}
				//img.copyTo(img2);
				drawChessboardCorners(img2, boardSize, imageCorners, found);
				Mat tmp;
				resize(img2, tmp, Size(img2.rows*0.5, img2.cols*0.5));
				//imshow("image", tmp);
				//key = cvWaitKey(400);
				key = cvWaitKey( 5 );
			
		}
		// start calibration
		calibrateCamera(objectPoints, // the 3D points
			imagePoints, // the image points
			imageSize,   // image size
			cameraMatrix,// output camera matrix
			distCoeffs,  // output distortion matrix
			rvecs, tvecs,// Rs, Ts
			0);       // set options
		cout << endl <<"intrinsic"<<endl << cameraMatrix;
		//cout << "distcoeff" << distCoeffs;

	IS3_1 = true;
}


void CHW2Dlg::OnBnClickedButton6()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	
	IS3_1 ? cout << endl << "distcoeff" << endl << distCoeffs : cout << "please run 3.2 first to get coefficient\n";
}


void CHW2Dlg::OnBnClickedButton7()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	//============================算外部參數======================================
	int index = COMBO1.GetCurSel();
	//cout << i << endl;
	if (IS3_1) {
		
		Mat rotation_matrix;
		
		Rodrigues(rvecs[index], rotation_matrix);

		//cout << rotation_matrix.rows << rotation_matrix.cols << endl;
		//cout << tvecs.at(0) <<tvecs.at(0).at<double>(0)<<endl;
		
		printf("\n\nextrinsic_matrix\n");
		printf("[");
		for (int i = 0; i<3; i++) {
			for (int j = 0; j<3; j++) {
				printf("%f", rotation_matrix.at<double>(i, j));
				if (j == 2) {
					printf(", %f", tvecs.at(index).at<double>(i));
					if (i != 2)
						printf(";\n");
				}
				else {
					printf(", ");
				}
			}
		}
		printf("]\n");
		
	}
	else {
		printf("please run 3.2 first to get coefficient\n");
	}
	
}

//====================================4=====================

void CHW2Dlg::OnBnClickedButton8()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	if (IS3_1) {
		CvMat * intrinsic_matrix = cvCreateMat(3, 3, CV_32FC1);
		CvMat * distortion_coeffs = cvCreateMat(5, 1, CV_32FC1);

		*intrinsic_matrix = cameraMatrix;
		*distortion_coeffs = distCoeffs;
		//cout << endl << cameraMatrix << endl << distCoeffs<<endl<<endl;

		//cout << endl << cvarrToMat(intrinsic_matrix) << endl << cvarrToMat(distortion_coeffs) << endl << endl;
		cvNamedWindow("3", CV_WINDOW_AUTOSIZE);

		
		for (int index = 1; index <= 5; index++) {
			char name[100];
			sprintf(name, "./images/CameraCalibration/%d", index);
			strcat(name, ".bmp");
			IplImage *img = cvLoadImage(name);
			
			int board_w = 11;
			int board_h = 8;
			int board_n = board_w * board_h;
			/*
			CvSize board_sz = cvSize(board_w, board_h);
			IplImage* gray_img = cvCreateImage(cvGetSize(img), 8, 1);
			CvPoint2D32f* corners = new CvPoint2D32f[board_n];
			if (i == 4) { //4用cvFindChessboardCorners偵測不到 只好用CvMat去偵測
				goto label; }
			int corner_count;
			int found = cvFindChessboardCorners(img, board_sz, corners, &corner_count, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
			cvCvtColor(img, gray_img, CV_BGR2GRAY);
			cvFindCornerSubPix(gray_img, corners, corner_count, cvSize(22, 11), cvSize(-1, -1), cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
		label:
		*/
			CvMat* object_points2 = cvCreateMat(board_n, 3, CV_32FC1);
			CvMat* image_points2 = cvCreateMat(board_n, 2, CV_32FC1);
			CvMat* point_counts = cvCreateMat(1, 1, CV_32SC1);
			CV_MAT_ELEM(*point_counts, int, 0, 0) = board_n;
			for (int i = 0, j = 0; j<board_n; ++i, ++j) {
				CV_MAT_ELEM(*image_points2, float, i, 0) = imagePoints[index -1][j].x;//第i-1個人的第j個x點
				CV_MAT_ELEM(*image_points2, float, i, 1) = imagePoints[index -1][j].y;//第i-1個人的第j個y點
				CV_MAT_ELEM(*object_points2, float, i, 0) = j / board_w;
				CV_MAT_ELEM(*object_points2, float, i, 1) = j%board_w;
				CV_MAT_ELEM(*object_points2, float, i, 2) = 0.0f;
			}

			CvMat* rotation_matrix = cvCreateMat(1, 3, CV_32FC1);
			CvMat* translation_matrix = cvCreateMat(1, 3, CV_32FC1);

			//cvCalibrateCamera2(object_points2, image_points2, point_counts, cvGetSize(img), intrinsic_matrix, distortion_coeffs, rotation_matrix, translation_matrix, 0);
			cvFindExtrinsicCameraParams2(object_points2, image_points2, intrinsic_matrix, distortion_coeffs, rotation_matrix, translation_matrix);
			//CvMat* rotation_matrix2 = cvCreateMat(3, 3, CV_32FC1);
			//cvRodrigues2(rotation_matrix, rotation_matrix2, NULL);

			CvMat* rotation_matrix2 = cvCreateMat(3, 1, CV_32FC1);
			CvMat* translation_matrix2 = cvCreateMat(3,1, CV_32FC1);
			//CV_MAT_ELEM(*rotation_matrix2, float, 0, 0) = CV_MAT_ELEM(*rotation_matrix, float, 0, 0);
			//CV_MAT_ELEM(*rotation_matrix2, float, 1, 0) = CV_MAT_ELEM(*rotation_matrix, float, 0, 1);
			//CV_MAT_ELEM(*rotation_matrix2, float, 2, 0) = CV_MAT_ELEM(*rotation_matrix, float, 0, 2);

			//CV_MAT_ELEM(*translation_matrix2, float, 0, 0) = CV_MAT_ELEM(*translation_matrix, float, 0, 0);
			//CV_MAT_ELEM(*translation_matrix2, float, 1, 0) = CV_MAT_ELEM(*translation_matrix, float, 0, 1);
			//CV_MAT_ELEM(*translation_matrix2, float, 2, 0) = CV_MAT_ELEM(*translation_matrix, float, 0, 2);

			//==========================================

			CvMat* object_points3 = cvCreateMat(8, 3, CV_32FC1);
			CV_MAT_ELEM(*object_points3, float, 0, 0) = 2.0f;
			CV_MAT_ELEM(*object_points3, float, 0, 1) = 2.0f;
			CV_MAT_ELEM(*object_points3, float, 0, 2) = 0.0f;

			CV_MAT_ELEM(*object_points3, float, 1, 0) = 2.0f;
			CV_MAT_ELEM(*object_points3, float, 1, 1) = 0.0f;
			CV_MAT_ELEM(*object_points3, float, 1, 2) = 0.0f;

			CV_MAT_ELEM(*object_points3, float, 2, 0) = 0.0f;
			CV_MAT_ELEM(*object_points3, float, 2, 1) = 0.0f;
			CV_MAT_ELEM(*object_points3, float, 2, 2) = 0.0f;

			CV_MAT_ELEM(*object_points3, float, 3, 0) = 0.0f;
			CV_MAT_ELEM(*object_points3, float, 3, 1) = 2.0f;
			CV_MAT_ELEM(*object_points3, float, 3, 2) = 0.0f;

			CV_MAT_ELEM(*object_points3, float, 4, 0) = 2.0f;
			CV_MAT_ELEM(*object_points3, float, 4, 1) = 2.0f;
			CV_MAT_ELEM(*object_points3, float, 4, 2) = 2.0f;

			CV_MAT_ELEM(*object_points3, float, 5, 0) = 2.0f;
			CV_MAT_ELEM(*object_points3, float, 5, 1) = 0.0f;
			CV_MAT_ELEM(*object_points3, float, 5, 2) = 2.0f;

			CV_MAT_ELEM(*object_points3, float, 6, 0) = 0.0f;
			CV_MAT_ELEM(*object_points3, float, 6, 1) = 0.0f;
			CV_MAT_ELEM(*object_points3, float, 6, 2) = 2.0f;

			CV_MAT_ELEM(*object_points3, float, 7, 0) = 0.0f;
			CV_MAT_ELEM(*object_points3, float, 7, 1) = 2.0f;
			CV_MAT_ELEM(*object_points3, float, 7, 2) = 2.0f;

			CvMat* image_points3 = cvCreateMat(8, 2, CV_32FC1);

			cvProjectPoints2(object_points3, rotation_matrix, translation_matrix, intrinsic_matrix, distortion_coeffs, image_points3);

			CvPoint Down1 = cvPoint(image_points3->data.fl[0 * image_points3->cols + 0], image_points3->data.fl[0 * image_points3->cols + 1]);
			CvPoint Down2 = cvPoint(image_points3->data.fl[1 * image_points3->cols + 0], image_points3->data.fl[1 * image_points3->cols + 1]);
			CvPoint Down3 = cvPoint(image_points3->data.fl[2 * image_points3->cols + 0], image_points3->data.fl[2 * image_points3->cols + 1]);
			CvPoint Down4 = cvPoint(image_points3->data.fl[3 * image_points3->cols + 0], image_points3->data.fl[3 * image_points3->cols + 1]);
			CvPoint Top1 = cvPoint(image_points3->data.fl[4 * image_points3->cols + 0], image_points3->data.fl[4 * image_points3->cols + 1]);
			CvPoint Top2 = cvPoint(image_points3->data.fl[5 * image_points3->cols + 0], image_points3->data.fl[5 * image_points3->cols + 1]);
			CvPoint Top3 = cvPoint(image_points3->data.fl[6 * image_points3->cols + 0], image_points3->data.fl[6 * image_points3->cols + 1]);
			CvPoint Top4 = cvPoint(image_points3->data.fl[7 * image_points3->cols + 0], image_points3->data.fl[7 * image_points3->cols + 1]);

			cvLine(img, Down1, Down2, CV_RGB(225, 0, 0), 10, CV_AA, 0);
			cvLine(img, Down2, Down3, CV_RGB(225, 0, 0), 10, CV_AA, 0);
			cvLine(img, Down3, Down4, CV_RGB(225, 0, 0), 10, CV_AA, 0);
			cvLine(img, Down4, Down1, CV_RGB(225, 0, 0),10, CV_AA, 0);

			cvLine(img, Top1, Top2, CV_RGB(225, 0, 255), 10, CV_AA, 0);
			cvLine(img, Top2, Top3, CV_RGB(225, 0, 255), 10, CV_AA, 0);
			cvLine(img, Top3, Top4, CV_RGB(225, 0, 255), 10, CV_AA, 0);
			cvLine(img, Top4, Top1, CV_RGB(225, 0, 255), 10, CV_AA, 0);

			cvLine(img, Top1, Down1, CV_RGB(225, 255, 0), 10, CV_AA, 0);
			cvLine(img, Top2, Down2, CV_RGB(225, 255, 0), 10, CV_AA, 0);
			cvLine(img, Top3, Down3, CV_RGB(225,255, 0), 10, CV_AA, 0);
			cvLine(img, Top4, Down4, CV_RGB(225, 255, 0), 10, CV_AA, 0);

			IplImage *dst;
			CvSize dst_cvsize;
			dst_cvsize.width = img->width * 0.5;       //目標影像的寬為源影像寬的scale倍
			dst_cvsize.height = img->height * 0.5; //目標影像的高為源影像高的scale倍

			dst = cvCreateImage(dst_cvsize, img->depth, img->nChannels); //創立目標影像
			cvResize(img, dst, CV_INTER_LINEAR);    //縮放來源影像到目標影像
			cvShowImage("3", dst);
			if (index == 15) {
				printf("Display over\n");
				cvWaitKey(0);
			}
			else {
				cvWaitKey(500);
			}
			cvReleaseImage(&img);
		}
		cvDestroyWindow("3");
		
	}
	else
		printf("please run 3.2 first to get coefficient\n");
}


//==================================2========================================

/*******************2.1*******************************/
vector<Vec3f> circles;
Mat src, src_gray;
int a = 40, b = 58, c = 16, d = 14, e = 26;
void calcCircles(int, void*) {
	Mat src = imread("images/q2_train.jpg", CV_LOAD_IMAGE_COLOR);
	HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, 1, a, b, c, d, e);

	/// Draw the circles detected
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// circle center
		circle(src, center, 3, Scalar(0, 255, 0), -1, 8, 0);
		// circle outline
		circle(src, center, radius, Scalar(0, 0, 255), 3, 8, 0);
	}

	imshow("Hough Circle Transform Demo", src);
}



void CHW2Dlg::OnBnClickedButton3()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	//Mat img = imread("images/q2_train.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat result = imread("images/q2_train.jpg", CV_LOAD_IMAGE_COLOR);



	/// Read the image
	src = imread("images/q2_train.jpg", CV_LOAD_IMAGE_COLOR);
	/// Convert it to gray
	cvtColor(src, src_gray, CV_BGR2GRAY);

	/// Reduce the noise so we avoid false circle detection
	GaussianBlur(src_gray, src_gray, Size(9, 9), 2, 2);



	namedWindow("Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE);
	calcCircles(0,0);
	/// Apply the Hough Transform to find the circles
	//createTrackbar("a", "Hough Circle Transform Demo", &a, 500, calcCircles);
	//createTrackbar("b", "Hough Circle Transform Demo", &b, 500, calcCircles);
	//createTrackbar("c", "Hough Circle Transform Demo", &c, 500, calcCircles);
	//createTrackbar("d", "Hough Circle Transform Demo", &d, 500, calcCircles);
	//createTrackbar("e", "Hough Circle Transform Demo", &e, 500, calcCircles);

	/// Show your results



}
/*******************2.3*******************************/
 Mat hsv; Mat hue;
 Mat mask;
 int a2=23, b2=144, c2=27, d2=177;
 int bins = 25;

void Hist_and_Backproj(int, void*)
{
	MatND hist;
	int hbins = 30, sbins = 32;
	int histSize[] = { hbins, sbins };
	float hue_range[2];// = { 103, 121 };
	hue_range[0] = a2;
	hue_range[1] = b2;
	float Saturation_range[2];// = { 48, 190 };
	Saturation_range[0] = c2;
	Saturation_range[1] = d2;
	const float* ranges[] = { hue_range , Saturation_range };
	int channels[] = { 0, 1 };
	/// Get the Histogram and normalize it 
	//Mask
	calcHist(&hsv, 1, channels,mask, hist, 2, histSize, ranges, true, false);
	normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());

	//load test image
	Mat test = imread("images/q2_test.jpg", CV_LOAD_IMAGE_COLOR);
	Mat test_hsv;
	cvtColor(test, test_hsv, CV_BGR2HSV);

	Mat test_hue;
	test_hue.create(test_hsv.size(), test_hsv.depth());
	int ch[] = { 0, 0 };
	mixChannels(&test_hsv, 1, &test_hue, 1, ch, 1);
	/// Get Backprojection
	MatND backproj;
	calcBackProject(&test_hsv, 1, channels, hist, backproj, ranges, 1, true);

	/// Draw the backproj
	imshow("BackProj2.3", backproj);

	/// Draw the histogram
	/*
	int w = 400; int h = 400;
	int bin_w = cvRound((double)w / histSize[0]);
	Mat histImg = Mat::zeros(w, h, CV_8UC3);

	for (int i = 0; i < hbins; i++)
	{
		rectangle(histImg, Point(i*bin_w, h), Point((i + 1)*bin_w, h - cvRound(hist.at<float>(i)*h / 255.0)), Scalar(0, 0, 255), -1);
	}

	imshow("Histogram", histImg);
	*/
}
void CHW2Dlg::OnBnClickedButton9()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	// TODO: 在此加入控制項告知處理常式程式碼
	//Mat img = imread("images/q2_train.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat result = imread("images/q2_train.jpg", CV_LOAD_IMAGE_COLOR);



	/// Read the image
	src = imread("images/q2_train.jpg", CV_LOAD_IMAGE_COLOR);
	/// Convert it to gray
	cvtColor(src, src_gray, CV_BGR2GRAY);
	cvtColor( src, hsv, CV_BGR2HSV );

	/// Reduce the noise so we avoid false circle detection
	GaussianBlur(src_gray, src_gray, Size(9, 9), 2, 2);
	//namedWindow("Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE);
	/// Apply the Hough Transform to find the circles
	HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, 1, 40, 58, 16, 14, 26);

	/// Draw the circles detected
	mask = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
	Mat dst;

	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// circle center
		//circle(src, center, 3, Scalar(0, 255, 0), -1, 8, 0);
		// circle outline
		//circle(src, center, radius, Scalar(0, 0, 255), 3, 8, 0);
		circle(mask, center, radius, Scalar(255, 255, 255), -1, 8, 0); //-1 means filled
	}

	src.copyTo(dst);
	//src.copyTo(dst, mask);
	//threshold(dst, dst,3 , 255, THRESH_BINARY);
	//imshow("Hough Circle Transform Demo", dst);
	//做到上面，把除了藍圈圈以外的全部mask掉了

	cvtColor(dst, hsv, CV_BGR2HSV);
	/// Use only the Hue value
	hue.create(hsv.size(), hsv.depth());
	int ch[] = { 0, 0 };
	mixChannels(&hsv, 1, &hue, 1, ch, 1);

	/// Create Trackbar to enter the number of bins
	//char* window_image = "Source image";
	//namedWindow(window_image, CV_WINDOW_AUTOSIZE);
	//createTrackbar("* Hue  bins: ", window_image, &bins, 180, Hist_and_Backproj);
	//createTrackbar("a: ", window_image, &a2, 180, Hist_and_Backproj);
	//createTrackbar("b: ", window_image, &b2, 180, Hist_and_Backproj);
	//createTrackbar("c ", window_image, &c2, 255, Hist_and_Backproj);
	//createTrackbar("d ", window_image, &d2, 255, Hist_and_Backproj);
	Hist_and_Backproj(0, 0);

	/// Show the image
	//imshow(window_image, src);
	waitKey(5);
	cout << "Over\n";

	/// Show your results
}

/*******************2.2*******************************/
int a3=1;
int bin2_2=180;
void Hist(int, void*)
{
	MatND hist;
	int histSize = MAX(bin2_2, 2);
	float hue_range[] = { a3, 180 };//因為除了圈圈的其他部分轉化成黑色，所以值要從1開始
	const float* ranges = { hue_range };

	/// Get the Histogram and normalize it
	calcHist(&hue, 1, 0, Mat(), hist, 1, &histSize, &ranges, true, false);
	normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());

	/// Draw the histogram
	int w = 500; int h = 500;
	int bin_w = cvRound((double)w / histSize);
	Mat histImg(w, h, CV_8UC3, Scalar(255, 255, 255));
	//Mat histImg2(w, h, CV_8UC3, Scalar(255, 255, 255));
	for (int i = 0; i < bin2_2; i++)
	{
		rectangle(histImg, Point((i + 1)*bin_w, (h - cvRound(hist.at<float>(i)*h / 255.0))), Point(i*bin_w, h), Scalar(0, 0, 255), -1);
	}
	//drawHistImg(hist, histImg2);
	imshow("Histogram", histImg);
	
	
}
void CHW2Dlg::OnBnClickedButton10()
{
	/// Read the image
	src = imread("images/q2_train.jpg", CV_LOAD_IMAGE_COLOR);
	/// Convert it to gray
	cvtColor(src, src_gray, CV_BGR2GRAY);
	cvtColor(src, hsv, CV_BGR2HSV);
	/// Reduce the noise so we avoid false circle detection
	GaussianBlur(src_gray, src_gray, Size(9, 9), 2, 2);
	//namedWindow("Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE);
	/// Apply the Hough Transform to find the circles
	HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, 1, 40, 58, 16, 14, 26);

	/// Draw the circles detected
	mask = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
	Mat dst;

	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// circle center
		//circle(src, center, 3, Scalar(0, 255, 0), -1, 8, 0);
		// circle outline
		//circle(src, center, radius, Scalar(0, 0, 255), 3, 8, 0);
		circle(mask, center, radius, Scalar(255, 255, 255), -1, 8, 0); //-1 means filled
	}

	//src.copyTo(dst);
	src.copyTo(dst, mask);

	//做到上面，把除了藍圈圈以外的全部mask掉了 開始只畫圈圈的長條圖

	cvtColor(dst, hsv, CV_BGR2HSV);

	//imshow("Hough Circle Transform Demo", hsv);
	/// Use only the Hue value
	hue.create(hsv.size(), hsv.depth());
	int ch[] = { 0, 0 };
	mixChannels(&hsv, 1, &hue, 1, ch, 1);

	/// Create Trackbar to enter the number of bins
	//char* window_image = "Source image";
	//namedWindow(window_image, CV_WINDOW_AUTOSIZE);
	//createTrackbar("* Hue  bins: ", window_image, &bin2_3, 180, Hist);

	//createTrackbar("a: ", window_image, &a3, 180, Hist);
	Hist(0, 0);

	/// Show the image
	//imshow(window_image, src);
}
