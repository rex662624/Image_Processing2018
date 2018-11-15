
// TESTDlg.cpp : 實作檔
//

#include "stdafx.h"
#include "TEST.h"
#include "TESTDlg.h"
#include "afxdialogex.h"
#include <iostream>
#include "opencv2\highgui\highgui.hpp"
#include "opencv/cv.h"
#include <opencv2/opencv.hpp>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif
#pragma warning( disable : 4996 )

using namespace std;
using namespace cv;
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


// CTESTDlg 對話方塊



CTESTDlg::CTESTDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_TEST_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CTESTDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CTESTDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON1, &CTESTDlg::OnBnClickedButton1)
	ON_BN_CLICKED(IDC_BUTTON2, &CTESTDlg::OnBnClickedButton2)
	ON_BN_CLICKED(IDC_BUTTON3, &CTESTDlg::OnBnClickedButton3)
	ON_BN_CLICKED(IDC_BUTTON4, &CTESTDlg::OnBnClickedButton4)
	ON_BN_CLICKED(IDC_BUTTON5, &CTESTDlg::OnBnClickedButton5)
	ON_BN_CLICKED(IDC_BUTTON6, &CTESTDlg::OnBnClickedButton6)
	ON_BN_CLICKED(IDC_BUTTON7, &CTESTDlg::OnBnClickedButton7)
	ON_BN_CLICKED(IDC_BUTTON8, &CTESTDlg::OnBnClickedButton8)
	ON_BN_CLICKED(IDC_BUTTON9, &CTESTDlg::OnBnClickedButton9)
	ON_BN_CLICKED(IDC_BUTTON10, &CTESTDlg::OnBnClickedButton10)
END_MESSAGE_MAP()


// CTESTDlg 訊息處理常式

BOOL CTESTDlg::OnInitDialog()
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

void CTESTDlg::OnSysCommand(UINT nID, LPARAM lParam)
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

void CTESTDlg::OnPaint()
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
HCURSOR CTESTDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void CTESTDlg::OnBnClickedButton1()
{

	// TODO: 在此加入控制項告知處理常式程式碼
	IplImage* img = cvLoadImage("./images/dog.bmp");
	cvNamedWindow("Image", CV_WINDOW_AUTOSIZE);
	cvShowImage("Image", img);
	cout << "Height = "<<img->height<< "\nWidth = " << img->width<<endl;
	cvWaitKey(0);
	cvReleaseImage(&img);
	cvDestroyWindow("Image");
}


void CTESTDlg::OnBnClickedButton2()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	Mat input = imread("./images/color.png", CV_LOAD_IMAGE_COLOR);
	int rows = input.rows;
	int cols = input.cols;

	Mat output(rows, cols, CV_8UC3);

	for (int i = 0; i < input.rows; i++) 
		for (int j = 0; j <input.cols; j++) {
			output.at<Vec3b>(i, j)[0] = input.at<Vec3b>(i, j)[1];
			output.at<Vec3b>(i, j)[1] = input.at<Vec3b>(i, j)[2];
			output.at<Vec3b>(i, j)[2] = input.at<Vec3b>(i, j)[0];
		}
	

	imshow("RGB", output);
}


void CTESTDlg::OnBnClickedButton3()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	Mat img1 = imread("./images/dog.bmp",IMREAD_COLOR);
	Mat img2;
	flip(img1, img2, 1);

	imshow("flip", img2);

	waitKey(0);

	cvDestroyWindow("flip");
}
//========================1.4=======================
Mat img1,img2, output;
int alpha_g=50;

void  blend(int , void*)
{
	double alpha = (double)alpha_g / 100;
	double beta = (1.0 - alpha);

	addWeighted(img1, alpha, img2, beta, 0.0, output);

	imshow("Image", output);
}

void CTESTDlg::OnBnClickedButton4()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	img1 = imread("./images/dog.bmp", IMREAD_COLOR);
	flip(img1, img2,1);
	namedWindow("Image", 0);
	
	createTrackbar("Blend", "Image", &alpha_g, 100, blend);	
	blend(alpha_g,0);
	
	cvWaitKey(0);
	cvDestroyWindow("Image");
	
}
//===================================

void CTESTDlg::OnBnClickedButton5()//4.1
{
	// TODO: 在此加入控制項告知處理常式程式碼
	Mat src = imread("./images/QR.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat dst;
	threshold(src, dst, 80, 255, THRESH_BINARY);
	imshow("origin", src);
	imshow("threshold", dst);
}


void CTESTDlg::OnBnClickedButton6()//4.2
{
	// TODO: 在此加入控制項告知處理常式程式碼
	Mat src = imread("./images/QR.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat dst;
	adaptiveThreshold(src, dst, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 19, -1);
	imshow("origin", src);
	imshow("threshold", dst);
}

//=============================================
void CTESTDlg::OnBnClickedButton7()//5.1
{
	// TODO: 在此加入控制項告知處理常式程式碼
	CString text;
	GetDlgItem(IDC_EDIT1)->GetWindowText(text); //讀取第一個
	double angle = _tstof((LPTSTR)(LPCTSTR)text);
	GetDlgItem(IDC_EDIT2)->GetWindowText(text); //讀取第二個
	double scale = _tstof((LPTSTR)(LPCTSTR)text);
	GetDlgItem(IDC_EDIT3)->GetWindowText(text); //讀取第三個
	double Tx = _tstof((LPTSTR)(LPCTSTR)text);
	GetDlgItem(IDC_EDIT4)->GetWindowText(text); //讀取第四個
	double Ty = _tstof((LPTSTR)(LPCTSTR)text);
	//cout << angle << scale << Tx << Ty << endl;
	//======
	Mat src = imread("./images/OriginalTransform.png");
	Mat dst = Mat::zeros(src.rows, src.cols, src.type());
	//旋轉
	Point center = Point(130,125);
	Mat rot_mat = getRotationMatrix2D(center, angle, scale);
	warpAffine(src, dst, rot_mat, dst.size());
	//平移
	cv::Mat t_mat = cv::Mat::zeros(2, 3, CV_32FC1);
	t_mat.at<float>(0, 0) = 1;
	t_mat.at<float>(0, 2) = Tx; //水平平移
	t_mat.at<float>(1, 1) = 1;
	t_mat.at<float>(1, 2) = Ty; //垂直平移

	warpAffine(dst, dst, t_mat, dst.size())	;

	imshow("origin", src);
	imshow("Affine", dst);
	waitKey(0);
	
}

//======5.2
int index=0;
Point TopLeft(-1, -1);
Point TopRight(-1, -1);
Point DownRight(-1, -1);
Point DownLeft(-1, -1);

Mat src_img ;

void onMouse(int Event, int x, int y, int flags, void* param) {
	if (Event == CV_EVENT_LBUTTONDOWN&&index == 0) {
		TopLeft.x = x;
		TopLeft.y = y;
		index++;
	}
	else if (Event == CV_EVENT_LBUTTONDOWN&&index == 1) {
		TopRight.x = x;
		TopRight.y = y;
		index++;
	}

	else if (Event == CV_EVENT_LBUTTONDOWN&&index == 2) {
		DownRight.x = x;
		DownRight.y = y;
		index++;
	}

	else if (Event == CV_EVENT_LBUTTONDOWN&&index == 3) {
		DownLeft.x = x;
		DownLeft.y = y;
		index++;

		// 設定變換[之前]與[之後]的坐標 (左上,左下,右下,右上)
		cv::Point2f pts1[] = { TopLeft,DownLeft,TopRight,DownRight };
		cv::Point2f pts2[] = { cv::Point2f(20,20),cv::Point2f(20,450),cv::Point2f(450,20),cv::Point2f(450,450) };
		// 透視變換行列計算
		cv::Mat perspective_matrix = cv::getPerspectiveTransform(pts1, pts2);
		cv::Mat dst_img;
		// 變換
		cv::warpPerspective(src_img, dst_img, perspective_matrix, src_img.size(), cv::INTER_LINEAR);
		//Crop??
		cv::Rect roi;
		roi.x =0;
		roi.y = 0;
		roi.width = 450;
		roi.height =450;

		/* Crop the original image to the defined ROI */

		cv::namedWindow("result", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);

		cv::Mat crop = dst_img(roi);

		cv::imshow("result", crop);
		index = 0;
	}
}

void CTESTDlg::OnBnClickedButton8()//5.2
{
	// TODO: 在此加入控制項告知處理常式程式碼
	//mouse callback
	namedWindow("image", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
	setMouseCallback("image", onMouse, NULL);
	src_img = cv::imread("./images/OriginalPerspective.png", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
	imshow("image", src_img);
	cv::waitKey(0);
}

//==================3===================
void CTESTDlg::OnBnClickedButton9()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	Mat Gaussion_0,Gaussion_1,Gaussion_2,Lap_0, Lap_1,Inv_1,Inv_0,tmp1,tmp2	;
	Gaussion_0 = imread("./images/pyramids_Gray.jpg");
	
	pyrDown(Gaussion_0, Gaussion_1, Size(Gaussion_0.cols / 2, Gaussion_0.rows / 2));
	pyrDown(Gaussion_1, Gaussion_2, Size(Gaussion_1.cols / 2, Gaussion_1.rows / 2));

	pyrUp(Gaussion_2,tmp2, Size(Gaussion_2.cols * 2, Gaussion_2.rows * 2));
	pyrUp(Gaussion_1, tmp1, Size(Gaussion_1.cols * 2, Gaussion_1.rows * 2));

	subtract(Gaussion_1,tmp2,Lap_1);
	subtract(Gaussion_0,tmp1, Lap_0);
	add(tmp2, Lap_1,Inv_1);
	add(tmp1, Lap_0, Inv_0);

	namedWindow("Inverse_1", CV_WINDOW_AUTOSIZE);
	imshow("Inverse_1", Inv_1);
	namedWindow("Inverse_0", CV_WINDOW_AUTOSIZE);
	imshow("Inverse_0", Inv_0);
	namedWindow("Gaussion_1", CV_WINDOW_AUTOSIZE);
	imshow("Gaussion_1", Gaussion_1);
	namedWindow("Lap_1", CV_WINDOW_AUTOSIZE);
	imshow("Lap_1", Lap_1);
}

//====

Mat magnitude_1,direction_1;
int threshold_Mag=40, threshold_Dir=40;

void  Mag_threshold(int, void*) {
	Mat result;
	threshold(magnitude_1, result,threshold_Mag, 255, THRESH_BINARY);
	imshow("Magnitude", result);
}
void  Dir_threshold(int, void*) {
	Mat result(direction_1.rows, direction_1.cols, CV_32F, Scalar(0));;
	
	for (int i = 0; i <= direction_1.rows - 1; i++)
		for (int j = 0; j <= direction_1.cols - 1; j++) {
			if (direction_1.at<float>(i, j) > threshold_Dir+10 || direction_1.at<float>(i, j) < threshold_Dir-10)
				result.at<float>(i, j) = 0;
			else
				result.at<float>(i, j) = 255;
		}

	
	result.convertTo(result, CV_8UC1);

	//normalize(direction_1, direction_1, 0, 255, NORM_MINMAX); //歸一化 方便顯示
	/*
	threshold(direction_1, result, threshold_Dir-10, 255, THRESH_TOZERO);
	threshold(direction_1, result, threshold_Dir + 10, 255, THRESH_TOZERO_INV);
	*/
	imshow("Direction", result);
	
}

Mat calculateOrientations(Mat gradientX, Mat gradientY) {
	// Create container element
	Mat orientation = Mat(gradientX.rows, gradientX.cols, CV_32F);

	// Calculate orientations of gradients --> in degrees
	// Loop over all matrix values and calculate the accompagnied orientation
	for (int i = 0; i < gradientX.rows; i++) {
		for (int j = 0; j < gradientX.cols; j++) {
			// Retrieve a single value
			float valueX = gradientX.at<uchar>(i, j);
			float valueY = gradientY.at<uchar>(i, j);
			// Calculate the corresponding single direction, done by applying the arctangens function
			float result = fastAtan2(valueX, valueY);
			// Store in orientation matrix element
			orientation.at<float>(i, j) = result;
		}
	}

	return orientation;
}



void CTESTDlg::OnBnClickedButton10()//2
{
		
	// TODO: 在此加入控制項告知處理常式程式碼
	Mat src = imread("./images/M8.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat Gaussian;
	GaussianBlur(src, Gaussian, Size(3, 3), 0, 0);
	//imshow("origin", src);
	imshow("gaussianBlur_3", Gaussian);

	//vertical
	Mat vertical(Gaussian.rows, Gaussian.cols, CV_16S, Scalar(0));
	

	for (int i = 1; i < Gaussian.rows - 1; i++) {
		for (int j = 1; j < Gaussian.cols - 1; j++) {

			int S1 = Gaussian.at<uchar>(i - 1, j - 1) * 1;
			int S2 = Gaussian.at<uchar>(i, j - 1) * 2;
			int S3 = Gaussian.at<uchar>(i + 1, j - 1) * 1;

			int S4 = Gaussian.at<uchar>(i - 1, j + 1) * -1;
			int S5 = Gaussian.at<uchar>(i, j + 1) * -2;
			int S6 = Gaussian.at<uchar>(i + 1, j + 1) * -1;
			int sum = S1 + S2 + S3 + S4 + S5 + S6 ;
			//if (sum < 0) {
			//	sum *=-1;
			//}
			vertical.at<short>(i, j) = sum;
		}
	}
	//normalize(vertical, vertical, 0, 255, NORM_MINMAX);
	//Mat vertical_thr;
	
	//imshow("vertical_thr", vertical_thr);
	Mat vertical_2;
	convertScaleAbs(vertical, vertical_2);  //轉成CV_8U
	threshold(vertical_2, vertical_2, 40, 255, THRESH_TOZERO);
	imshow("vertical", vertical_2);
	//horizontal

	Mat horizontal(Gaussian.rows, Gaussian.cols, CV_16S, Scalar(0));
	//horizontal = Gaussian;

	for (int i = 1; i < Gaussian.rows - 1; i++) {
		for (int j = 1; j < Gaussian.cols - 1; j++) {
			int S1 = Gaussian.at<uchar>(i - 1, j - 1) * 1;
			int S2 = Gaussian.at<uchar>(i - 1, j) * 2;
			int S3 = Gaussian.at<uchar>(i - 1, j + 1) * 1;
			int S4 = Gaussian.at<uchar>(i + 1, j - 1) * -1;
			int S5 = Gaussian.at<uchar>(i + 1, j) * -2;
			int S6 = Gaussian.at<uchar>(i + 1, j + 1) * -1;
			int sum = S1 + S2 + S3 + S4 + S5 + S6;
			
			//if (sum < 0) {
			//	sum *= -1;
			//}
			horizontal.at<short>(i, j) = sum;
		}
	}
	//normalize(horizontal, horizontal, 0, 255, NORM_MINMAX);
	//Mat horizontal_thr;
	//threshold(horizontal, horizontal, 40, 255, THRESH_TOZERO);
	//imshow("horizontal_threshold", horizontal_thr);
	Mat horizontal_2;
	convertScaleAbs(horizontal, horizontal_2);  //轉成CV_8U
	threshold(horizontal_2, horizontal_2, 40, 255, THRESH_TOZERO);
	imshow("horizontal", horizontal_2);


	//Magnitude
	
	namedWindow("Magnitude", WINDOW_AUTOSIZE);
	Mat vertical_f, horizontal_f;
	vertical.convertTo(vertical_f,CV_32F);
	horizontal.convertTo(horizontal_f,CV_32F);

	addWeighted(vertical_2, 0.5, horizontal_2, 0.5, 0, magnitude_1);
	//magnitude(vertical_f, horizontal_f, magnitude_1);
	createTrackbar("value", "Magnitude", &threshold_Mag,255, Mag_threshold);


	//direction
	/*
	namedWindow("Direction", WINDOW_AUTOSIZE);
	direction_1 = calculateOrientations(horizontal, vertical);
	direction_1.convertTo(direction_1, CV_8U);
	imshow("Direction", direction_1);
	createTrackbar("value", "Direction", &threshold_Dir, 255, Dir_threshold);
	*/
	

	namedWindow("Direction", WINDOW_AUTOSIZE);
	//phase(horizontal_f, vertical_f, direction_1,false);
	//Mat result;
	//direction_1.convertTo(direction_1, CV_32F);
	//normalize(direction_1, direction_1, 0, 255);
	//int low = 0, high = 20;
	//createTrackbar("value", "Direction", &threshold_Dir,360, Dir_threshold);
	
	/*
	for (int i = 0; i <= direction_1.rows - 1; i++){
		for (int j = 0; j <= direction_1.cols - 1; j++) {
			if (direction_1.at<float>(i, j) > 360 || direction_1.at<float>(i, j) < 0)
				cout << direction_1.at<float>(i, j)<<endl;//direction_1.at<float>(i, j) = 0;
		}
	}
	*/
	
	/*
	for (int i = 0; i <= result.rows - 1; i++) 
		for (int j = 0; j <= result.cols - 1; j++) {
			if (result.at<uchar>(i, j) > high || result.at<uchar>(i, j) < low)
				result.at<uchar>(i, j) = 0;
		}
		*/
	phase (vertical_f, horizontal_f, direction_1, true);
	//direction_1.convertTo(direction_1, CV_8UC1);
	createTrackbar("value", "Direction", &threshold_Dir,360, Dir_threshold);
	//imshow("result", direction_1);
	


	waitKey(0);


}
