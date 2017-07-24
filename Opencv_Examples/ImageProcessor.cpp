#include "ImageProcessor.h"

ImageProcess::ImageProcess()
{
}

ImageProcess::~ImageProcess() {}

Mat ImageProcess::norm_0_255(InputArray _src) {
	Mat src = _src.getMat();
	// Create and return normalized image:
	Mat dst;
	switch (src.channels()) {
	case 1:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}

void ImageProcess::read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if (!path.empty() && !classlabel.empty()) {
			Mat img = imread(path, 0);
			Mat ROI;
			vector<Rect> _faces;
			vector<Rect> _eyes;
			face_cascade.detectMultiScale(img, _faces, 1.1, 2, 0, Size(30, 30));
			if (_faces.size() > 0) {
				images.push_back(img(_faces[0]));
			}
			else {
				images.push_back(img);
			}
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}


int ImageProcess::Count_People_Num(Mat frame)
{
	HOGDescriptor hog;
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
	while (true)
	{
		cap >> frame;
		int Dim = (frame.rows > frame.cols ? frame.rows : frame.cols) / 16;
		vector<Rect>faces;
		vector<Rect>faces_side;
		vector<Rect>bodies_upper;
		vector<Rect>bodies_full;
		vector<Rect>Pedestrians, Pedestrains_Filterd;
		//vector<Rect>heads;
		Mat frame_gray;
		cvtColor(frame, frame_gray, CV_BGR2GRAY);
		equalizeHist(frame_gray, frame_gray);
		//°»´ú
		face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0, Size(Dim, Dim));
		//faceside_cascade.detectMultiScale(frame_gray, faces_side, 1.1, 2, 0, Size(Dim, Dim));
		//bodyupper_cascade.detectMultiScale(frame_gray, bodies_upper, 1.1, 2, 0, Size(Dim, Dim*4));
		bodyfull_cascade.detectMultiScale(frame_gray, bodies_full, 1.1, 2, 0, Size(Dim, Dim * 8));
		//head_cascade.detectMultiScale(frame_gray, heads, 1.1, 2, 0, Size(30, 40));
		hog.detectMultiScale(frame_gray, Pedestrians, 0, Size(8, 8), Size(32, 32), 1.05, 2);
		for (int i = 0; i < faces.size(); i++)
		{
			rectangle(frame, faces[i], Scalar(0, 0, 255), 2);
		}
		for (int i = 0; i < faces_side.size(); i++)
		{
			rectangle(frame, faces_side[i], Scalar(0, 255, 255), 2);
		}
		for (int i = 0; i < bodies_upper.size(); i++)
		{
			rectangle(frame, bodies_upper[i], Scalar(255, 0, 255), 2);
		}
		for (int i = 0; i < bodies_full.size(); i++)
		{
			rectangle(frame, bodies_full[i], Scalar(255, 0, 0), 2);
		}
		//for (int i = 0; i < heads.size(); i++)
		//{
		//	rectangle(frame, heads[i], Scalar(0, 255, 0), 2);
		//}
		//¹LÂo­«Å|
		for (int i = 0; i < Pedestrians.size(); i++) {
			Rect r = Pedestrians[i];

			size_t j;
			for (j = 0; j < Pedestrians.size(); j++) {
				if (j != i && (r & Pedestrians[j]) == r) {
					break;
				}
			}
			if (j == Pedestrians.size()) {
				Pedestrains_Filterd.push_back(r);
			}
		}
		//
		for (int i = 0; i < Pedestrains_Filterd.size(); i++) {
			//resize
			Rect r = Pedestrains_Filterd[i];
			r.x += cvRound(r.width*0.1);
			r.width = cvRound(r.width*0.8);
			r.y += cvRound(r.height*0.07);
			r.height = cvRound(r.height*0.8);
			rectangle(frame, r.tl(), r.br(), cv::Scalar(0, 255, 0), 3);
		}
	}
}
bool ImageProcess::Invasion_Detect()
{

}
String ImageProcess::Plate_Detect(Mat frame)
{

}
void ImageProcess::Face_Recognition()
{

}