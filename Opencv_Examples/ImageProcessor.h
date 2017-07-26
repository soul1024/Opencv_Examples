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
	//數人數(不局限於正面)
	int Count_People_Num(Mat frame);
	//數人頭(抓正面)
	int Count_Face_Num(Mat frame);
	//入侵偵測先Invasion_Begin->Invasion_Detect
	void Invasion_Begin(Mat frame);
	//要連續餵圖，偵測到入侵時回傳true，否則false
	bool Invasion_Detect(Mat frame);
	//偵測車牌
	String Plate_Detect(Mat frame);
	//臉部辨識，如果指定人臉存在於另一張背景中回傳true，否則回傳false
	bool Face_Recognition(Mat target, Mat background);
};