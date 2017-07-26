#pragma once
#include <opencv2\opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2\core.hpp>
#include <opencv2\cvconfig.h>
#include <opencv2\imgproc.hpp>
#include <opencv2\objdetect.hpp>
#include <opencv2\features2d.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\xfeatures2d.hpp>
#include <opencv2\face.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::ml;

static String FACE_CASCADE_NAME = "../data/haarcascade_frontalface_alt.xml";
static String FACE_SIDELOOKING_CASCADE_NAME = "../data/haarcascade_profileface.xml";
static String UPPERBODY_CASCADE_NAME = "../data/haarcascade_upperbody.xml";
static String FULLBODY_CASCADE_NAME = "../data/haarcascade_fullbody.xml";
static String EYES_CASCADE_NAME = "../data/haarcascade_eye.xml";
static String HEAD_CASCADE_NAME = "../data/haarcascade_head.xml";
static String PLATE_CASCADE_NAME = "../data/haarcascade_licence_plate_us.xml";

class ImageProcess {
	
public:
	ImageProcess();
	~ImageProcess();
private:
	
	//for Invasion
	bool bFirst_Inva;
	Mat gray_Inva;
	Mat Foreground_Inva;
	Mat Background_Inva;

	Mat gray32f_Inva;
	Mat Foreground32f_Inva;
	Mat Background32f_Inva;
protected:	
	CascadeClassifier face_cascade;
	CascadeClassifier faceside_cascade;
	CascadeClassifier bodyupper_cascade;
	CascadeClassifier bodyfull_cascade;
	CascadeClassifier eyes_cascade;
	CascadeClassifier head_cascade;
	CascadeClassifier plate_cascade;

	const int train_samples = 1;
	const int classes = 36;
	const int sizex = 20;
	const int sizey = 30;
	const int ImageSize = sizex * sizey;
	char* pathToImages;

	Mat norm_0_255(InputArray _src);
	void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';');
protected:
	void PreProcessImage(Mat *inImage, Mat *outImage, int sizex, int sizey);
	void LearnFromImages(CvMat* trainData, CvMat* trainClasses);
	void RunSelfTest(Ptr<KNearest> knn2);
	String AnalyseImage(Ptr<KNearest> knearest, Mat frame);

public:
	//�ƤH��(�������󥿭�)
	int Count_People_Num(Mat frame);
	//�ƤH�Y(�쥿��)
	int Count_Face_Num(Mat frame);
	//�J�I������Invasion_Begin->Invasion_Detect
	void Invasion_Begin(Mat frame);
	//�n�s�����ϡA������J�I�ɦ^��true�A�_�hfalse
	bool Invasion_Detect(Mat frame);
	//�������P
	String Plate_Detect(Mat frame);
	//�y�����ѡA�p�G���w�H�y�s�b��t�@�i�I�����^��true�A�_�h�^��false
	bool Face_Recognition(Mat target, Mat background);
};