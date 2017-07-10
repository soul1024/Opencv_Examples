#include <iostream>
#include "main.h">
using namespace cv;
using namespace std;

void detectAndDisplay(Mat frame);
/** Global variables */
String face_cascade_name = "D:/class/sampleCode/openCV/07Face/sample21_haarcascades_Camera/test/opencv-master/data/haarcascades/haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "D:/class/sampleCode/openCV/07Face/sample21_haarcascades_Camera/test/opencv-master/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);
//下載　https://github.com/opencv/opencv/tree/master/data/haarcascades


int main() {

	if (!face_cascade.load(face_cascade_name))
	{
		printf("--(!)Error loading\n");
		return -1;
	}
	if (!eyes_cascade.load(eyes_cascade_name))
	{
		printf("--(!)Error loading\n");
		return -1;
	}
	VideoCapture cap(0);             //開啟攝影機
	if (!cap.isOpened()) return -1;   //確認攝影機打開
	Mat frame;                       //用矩陣紀錄抓取的每張frame
	for (;;) {
		if (cap.isOpened())
		{
			//frame=cvQueryFrame(cam);
			cap >> frame;   //把取得的影像放置到矩陣中
			if (!frame.empty())
			{
				detectAndDisplay(frame);
			}
			else
			{
				printf(" --(!) No captured frame -- Break!"); break;
			}

			if (waitKey(30) == 27) break;  //按鍵就離開程式
		}
	}
	return 0;
}

void detectAndDisplay(Mat frame)
{
	vector<Rect>faces;
	Mat frame_gray;
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0, Size(30, 30));
	for (int i = 0; i<faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
		ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
		Mat faceROI = frame_gray(faces[i]);
		std::vector<Rect> eyes;
		//-- In each face, detect eyes
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0, Size(30, 30));
		for (int j = 0; j < eyes.size(); j++)
		{
			Point center(faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5);
			int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
			circle(frame, center, radius, Scalar(255, 0, 0), 4, 8, 0);
		}
	}
	imshow(window_name, frame);                //建立一個視窗,顯示frame到camera名稱的視窗

}