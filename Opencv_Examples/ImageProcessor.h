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
	char pathToImages[] = "../data/pictures";

	Mat norm_0_255(InputArray _src);
	void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';');

public:
	int Count_People_Num(Mat frame);
	bool Invasion_Detect();
	String Plate_Detect(Mat frame);
	void Face_Recognition();
};