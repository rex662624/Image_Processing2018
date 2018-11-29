
// HW2.cpp : �w�q���ε{�������O�欰�C
//

#include "stdafx.h"
#include "HW2.h"
#include "HW2Dlg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CHW2App

BEGIN_MESSAGE_MAP(CHW2App, CWinApp)
	ON_COMMAND(ID_HELP, &CWinApp::OnHelp)
END_MESSAGE_MAP()


// CHW2App �غc

CHW2App::CHW2App()
{
	// �䴩���s�Ұʺ޲z��
	m_dwRestartManagerSupportFlags = AFX_RESTART_MANAGER_SUPPORT_RESTART;

	// TODO: �b���[�J�غc�{���X�A
	// �N�Ҧ����n����l�]�w�[�J InitInstance ��
}


// �Ȧ����@�� CHW2App ����

CHW2App theApp;


// CHW2App ��l�]�w

BOOL CHW2App::InitInstance()
{
	// ���p���ε{����T�M����w�ϥ� ComCtl32.dll 6 (�t) �H�᪩���A
	// �ӱҰʵ�ı�Ƽ˦��A�b Windows XP �W�A�h�ݭn InitCommonControls()�C
	// �_�h����������إ߳��N���ѡC
	INITCOMMONCONTROLSEX InitCtrls;
	InitCtrls.dwSize = sizeof(InitCtrls);
	// �]�w�n�]�t�Ҧ��z�Q�n�Ω����ε{������
	// �q�α�����O�C
	InitCtrls.dwICC = ICC_WIN95_CLASSES;
	InitCommonControlsEx(&InitCtrls);

	CWinApp::InitInstance();


	AfxEnableControlContainer();

	// �إߴ߼h�޲z���A�H����ܤ���]�t
	// ����߼h���˵��δ߼h�M���˵�����C
	CShellManager *pShellManager = new CShellManager;

	// �Ұ� [Windows ���] ��ı�ƺ޲z���i�ҥ� MFC ��������D�D
	CMFCVisualManager::SetDefaultManager(RUNTIME_CLASS(CMFCVisualManagerWindows));

	// �зǪ�l�]�w
	// �p�G�z���ϥγo�ǥ\��åB�Q���
	// �̫᧹�����i�����ɤj�p�A�z�i�H
	// �q�U�C�{���X�������ݭn����l�Ʊ`���A
	// �ܧ��x�s�]�w�Ȫ��n�����X
	// TODO: �z���ӾA�׭ק惡�r��
	// (�Ҧp�A���q�W�٩β�´�W��)
	SetRegistryKey(_T("���� AppWizard �Ҳ��ͪ����ε{��"));

	CHW2Dlg dlg;
	m_pMainWnd = &dlg;
	INT_PTR nResponse = dlg.DoModal();
	if (nResponse == IDOK)
	{
		// TODO: �b����m��ϥ� [�T�w] �Ӱ���ϥι�ܤ����
		// �B�z���{���X
	}
	else if (nResponse == IDCANCEL)
	{
		// TODO: �b����m��ϥ� [����] �Ӱ���ϥι�ܤ����
		// �B�z���{���X
	}
	else if (nResponse == -1)
	{
		TRACE(traceAppMsg, 0, "ĵ�i: ��ܤ���إߥ��ѡA�]���A���ε{���N�~�פ�C\n");
		TRACE(traceAppMsg, 0, "ĵ�i: �p�G�z�n�b��ܤ���W�ϥ� MFC ����A�h�L�k #define _AFX_NO_MFC_CONTROLS_IN_DIALOGS�C\n");
	}

	// �R���W���ҫإߪ��߼h�޲z���C
	if (pShellManager != NULL)
	{
		delete pShellManager;
	}

#ifndef _AFXDLL
	ControlBarCleanUp();
#endif

	// �]���w�g������ܤ���A�Ǧ^ FALSE�A�ҥH�ڭ̷|�������ε{���A
	// �ӫD���ܶ}�l���ε{�����T���C
	return FALSE;
}


//==========================
/*
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
void CHW2Dlg::OnBnClickedButton5()
{
	int imagewidth = 2048;
	int imageheight = 2048;

	std::vector<std::vector<cv::Point3f>> objectPoints;
	std::vector<std::vector<cv::Point2f>> imagePoints;


	cv::Mat map1, map2;

	cv::Mat img;
	cv::Mat img2;

	int key;
	char filename[256];

	int i;
	cv::Size boardSize(11, 8);
	cv::Size imageSize(imagewidth, imageheight);
	std::vector<cv::Point2f> imageCorners;
	std::vector<cv::Point3f> objectCorners;

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
		img.copyTo(img2);
		drawChessboardCorners(img2, boardSize, imageCorners, found);

		//imshow("image", img2);
		//key = cvWaitKey(400);
		key = cvWaitKey(5);

	}
	// start calibration
	calibrateCamera(objectPoints, // the 3D points
		imagePoints, // the image points
		imageSize,   // image size
		cameraMatrix,// output camera matrix
		distCoeffs,  // output distortion matrix
		rvecs, tvecs,// Rs, Ts
		0);       // set options
	cout << endl << "intrinsic" << endl << cameraMatrix;
	//cout << "distcoeff" << distCoeffs;

	IS3_1 = true;
}
*/

