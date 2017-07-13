#include <iostream>
#include "main.h"
using namespace cv;
using namespace std;
using namespace cv::ml;


/** Global variables */
String face_cascade_name = "../data/haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "../data/haarcascade_eye.xml";
String head_cascade_name = "../data/haarcascade_head.xml";
String plate_cascade_name = "../data/haarcascade_licence_plate_us.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
CascadeClassifier head_cascade;
CascadeClassifier plate_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);
//下載　https://github.com/opencv/opencv/tree/master/data/haarcascades

const int train_samples = 1;
const int classes = 10;
const int sizex = 20;
const int sizey = 30;
const int ImageSize = sizex * sizey;
char pathToImages[] = "../data/pictures";

//excercises
void Face_detectAndDisplay();
void Plate_dectecAndDisplay();
void Face_detectAndDisplay_thirdpart();
void AutoRegister();
void Sift();
void Bgfg_segm();
void ConnectedComponent();
void PreProcessImage(Mat *inImage, Mat *outImage, int sizex, int sizey);
void LearnFromImages(CvMat* trainData, CvMat* trainClasses);
void RunSelfTest(Ptr<KNearest> knn2);
void AnalyseImage(Ptr<KNearest> knearest, Mat frame);


int main() {

	if (!face_cascade.load(face_cascade_name))
	{
		printf("--(!)Error loading\n");
		return -1;
	}
	if (!head_cascade.load(head_cascade_name))
	{
		printf("--(!)Error loading\n");
		return -1;
	}
	if (!eyes_cascade.load(eyes_cascade_name))
	{
		printf("--(!)Error loading\n");
		return -1;
	}
	if (!plate_cascade.load(plate_cascade_name))
	{
		printf("--(!)Error loading\n");
		return -1;
	}
	namedWindow("img", WINDOW_AUTOSIZE);


	//Bgfg_segm();
	//AutoRegister();
	//Sift();
	//ConnectedComponent();
	Plate_dectecAndDisplay();
	return 0;
}

void Face_detectAndDisplay()
{
	VideoCapture cap(0);             //開啟攝影機
	if (!cap.isOpened()) return ;   //確認攝影機打開
	Mat frame;                       //用矩陣紀錄抓取的每張frame
	while (true)
	{
		cap >> frame;
		vector<Rect>faces;
		//vector<Rect>heads;
		Mat frame_gray;
		cvtColor(frame, frame_gray, CV_BGR2GRAY);
		equalizeHist(frame_gray, frame_gray);
		face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0, Size(30, 30));
		//head_cascade.detectMultiScale(frame_gray, heads, 1.1, 2, 0, Size(30, 40));
		//for (int i = 0; i < faces.size(); i++)
		//{
		//	//rectangle(frame, faces[i], Scalar(0, 0, 255), 2);
		//}
		//for (int i = 0; i < heads.size(); i++)
		//{
		//	rectangle(frame, heads[i], Scalar(0, 255, 0), 2);
		//}
		//處理重疊	
		/*vector<Rect> ans;
		for (int i = 0; i < faces.size(); i++)
		{
		for (int j = 0; j < heads.size(); j++)
		{
		Rect Intersection = faces[i] & heads[j];
		if (Intersection.area() > faces[i].area() / 2 && Intersection.area() > heads[j].area() / 2)
		{

		}
		else
		{
		}
		}
		}*/
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
			//if (eyes.size() > 0)
			{
				char _text[100] = { 0 };
				String text = _itoa(faces.size(), _text, 10);
				text = "人數:" + text;
				putText(frame, text, Point(0, 15), CV_FONT_ITALIC, 2.0, Scalar(0, 0, 255), 2);
			}
		}
		imshow(window_name, frame);                //建立一個視窗,顯示frame到camera名稱的視窗
		if (waitKey(30) == 27) break;  //按鍵就離開程式
	}
}

void Plate_dectecAndDisplay()
{
	VideoCapture cap(0);             //開啟攝影機
	if (!cap.isOpened()) return;   //確認攝影機打開
	Mat frame;                       //用矩陣紀錄抓取的每張frame
	CvMat* trainData = cvCreateMat(classes * train_samples, ImageSize, CV_32FC1);
	CvMat* trainClasses = cvCreateMat(classes * train_samples, 1, CV_32FC1);

	//namedWindow("single", CV_WINDOW_AUTOSIZE);
	//namedWindow("all", CV_WINDOW_AUTOSIZE);

	LearnFromImages(trainData, trainClasses);

	Mat matTrainFeatures = cvarrToMat(trainData);
	Mat matTrainLabels = cvarrToMat(trainClasses);


	Ptr<TrainData> trainingData;
	Ptr<KNearest> kclassifier = KNearest::create();

	trainingData = TrainData::create(matTrainFeatures,
		SampleTypes::ROW_SAMPLE, matTrainLabels);
	kclassifier->setIsClassifier(true);
	kclassifier->setAlgorithmType(KNearest::Types::BRUTE_FORCE);
	kclassifier->setDefaultK(1);
	kclassifier->train(trainingData);


	RunSelfTest(kclassifier);

	cout << "losgehts\n";

	
	while (true)
	{
		cap >> frame;
		vector<Rect>plates;
		Mat frame_gray;
		cvtColor(frame, frame_gray, CV_BGR2GRAY);
		equalizeHist(frame_gray, frame_gray);
		plate_cascade.detectMultiScale(frame_gray, plates, 1.1, 2, 0, Size(30, 30));
		for (int i = 0; i < plates.size(); i++)
		{
			rectangle(frame, plates[i], Scalar(0, 0, 255), 2);
		}
		imshow(window_name, frame);                //建立一個視窗,顯示frame到camera名稱的視窗

		//
		for (int i = 0; i < plates.size(); i++) {
			//roi
			Mat plate = frame(plates[i]);
			//
			AnalyseImage(kclassifier, plate);
		}

		if (waitKey(30) == 27) break;  //按鍵就離開程式

	}
}

//void Face_detectanddisplay_thirdpart(Mat frame)
//{
//
//	Rect ROI;
//	
//	ROI.width = -1;
//	ROI.height = 720;
//	ROI.x = 0;
//	ROI.y = 00;
//
//	//
//	Mat frame_gray;
//	cvtColor(frame, frame_gray, CV_BGR2GRAY);
//	equalizeHist(frame_gray, frame_gray);
//	
//	// rectangles found or replayes for the current frame
//	vector<Rect> current_faces;
//	vector<int> pastFaces;
//
//	//if (param_replayfile_name == 0)
//	{
//		face_cascade.detectMultiScale(frame_gray, current_faces, 1.2
//			, 2
//			, CV_HAAR_SCALE_IMAGE
//			, Size(30, 30)
//			, Size(50, 50));
//	}
//	/*else
//	{
//		static unsigned int current_index_in_replay = 0;
//
//		while (replayContent[current_index_in_replay]<numImg)
//		{
//			current_index_in_replay += 5;
//		}
//		while (replayContent[current_index_in_replay] == numImg)
//		{
//			Rect r;
//			r.x = replayContent[current_index_in_replay + 1];
//			r.y = replayContent[current_index_in_replay + 2];
//			r.width = replayContent[current_index_in_replay + 3];
//			r.height = replayContent[current_index_in_replay + 4];
//			current_faces.push_back(r);
//			if (current_index_in_replay<replayContent.size() - 5)
//			{
//				current_index_in_replay += 5;
//			}
//			else
//			{
//				quit = true;
//				break;
//			}
//		}
//	}*/
//
//	static int farthest_past_value = 0;
//	static int farthest_past_index = 0;
//
//	/*if (numImg % 10 == 0)
//	{
//		printf("Frame#%d(%.1f sec): doing %d computations (%d * %d). Cross count so far : %d\n",
//			numImg,
//			numImg * 1 / 50.0f,
//			current_faces.size() * (pastFaces.size() - farthest_past_index),
//			current_faces.size(),
//			(pastFaces.size() - farthest_past_index),
//			crosscount);
//	}*/
//
//	for (unsigned int i = 0; i < current_faces.size(); i++)
//	{
//		Rect r = current_faces[i];
//		Point center;
//
//		rectangle(frame, r, Scalar(0, 0, 255), 2, 0, 0);
//		center.x = cvRound((ROI.x + r.x + r.width*0.5));
//		center.y = cvRound((ROI.y + r.y + r.height*0.5));
//
//		pastFaces.push_back(cvRound((ROI.x + r.x + r.width*0.5)));
//		pastFaces.push_back(cvRound((ROI.y + r.y + r.height*0.5)));
//		//pastFaces.push_back(numImg);
//
//
//		/*if (param_outfile_name != 0)
//		{
//			out_text_file << numImg << " " << r.x << " " << r.y << " " << r.width << " " << r.height << std::endl;
//		}*/
//
//		// Find the closest face on the last frames and link them
//		// through the use of the same face id
//		int dist_min = 100000;
//		const int FRAME_TO_FRAME_DIST_THRESHOLD = 1000;
//		const float zDimension = 1.0f;
//		int closest_x, closest_y, closest_index = 0;
//		for (unsigned int j = farthest_past_index; j<pastFaces.size(); j += 2)
//		{
//			int dist;
//
//			// The distance between two faces is an euclidian distance with
//			// time being the 3rd dimension, corrected by the zDimension constant.
//
//			dist = sqrt(center.x - pastFaces[j]) + sqrt(center.y - pastFaces[j + 1]);
//			if ((dist < dist_min) &&
//				(numImg>pastFaces[j + 2]) &&
//				(faceLinks.find(j / 3) == faceLinks.end()))
//			{
//				dist_min = dist;
//				closest_x = pastFaces[j];
//				closest_y = pastFaces[j + 1];
//				closest_index = j;
//			}
//			while (farthest_past_value + 20<pastFaces[j + 2])
//			{
//				farthest_past_index += 3;
//				farthest_past_value = pastFaces[farthest_past_index + 2];
//			}
//		}
//
//		if (dist_min < FRAME_TO_FRAME_DIST_THRESHOLD)
//		{
//			lines.push_back(closest_x);
//			lines.push_back(closest_y);
//			lines.push_back(center.x);
//			lines.push_back(center.y);
//
//			if (faceLinks.find(closest_index / 3) == faceLinks.end())
//			{
//				faceLinks[closest_index / 3] = 1;
//			}
//			else
//			{
//				faceLinks[closest_index / 3]++;
//			}
//
//			if (faceIndices.find(closest_index / 3) == faceIndices.end())
//			{
//				faceIndices[closest_index / 3] = faceUID;
//				faceIndices[(pastFaces.size() - 1) / 3] = faceUID;
//				lines.push_back(faceUID);
//				faceCounted[faceUID] = false;
//				trailLength[faceUID] = 1;
//				faceUID++;
//
//			}
//			else
//			{
//				faceIndices[(pastFaces.size() - 1) / 3] = faceIndices[closest_index / 3];
//				lines.push_back(faceIndices[closest_index / 3]);
//				trailLength[faceIndices[closest_index / 3]]++;
//				if ((faceCounted[faceIndices[closest_index / 3]]) &&
//					(trailLength[faceIndices[closest_index / 3]] == param_trail_length_threshold))
//				{
//					crosscount++;
//				}
//			}
//
//			// Check if the current link crosses one of the lines
//			if (((closest_y<param_y_line) && (center.y >= param_y_line)) ||
//				((closest_y>param_y_line) && (center.y <= param_y_line)) ||
//				((closest_x<param_x_line) && (center.x >= param_x_line)) ||
//				((closest_x>param_x_line) && (center.x <= param_x_line)))
//			{
//				if (!faceCounted[faceIndices[(pastFaces.size() - 1) / 3]])
//				{
//					faceCounted[faceIndices[(pastFaces.size() - 1) / 3]] = true;
//					if (trailLength[faceIndices[(pastFaces.size() - 1) / 3]]>param_trail_length_threshold)
//						crosscount++;
//
//					//std::cout << "cross count = " << crosscount << std::endl;
//				}
//			}
//		}
//		//if (param_show_faces)
//		{
//			Scalar color;
//			if (faceIndices.find((pastFaces.size() - 1) / 3) != faceIndices.end())
//			{
//				if (trailLength[faceIndices[(pastFaces.size() - 1) / 3]]<5)
//					color = cvScalar(0, 0, 0);
//				else
//				{
//					if (faceCounted[faceIndices[(pastFaces.size() - 1) / 3]])
//						color = Scalar(0, 0, 255);
//					else
//						color = Scalar(0, 255, 0);
//				}
//			}
//			Point p1 = cvPoint(ROI.x + r.x,
//				ROI.y + r.y);
//			Point p2 = cvPoint(ROI.x + r.x + r.width,
//				ROI.y + r.y + r.height);
//
//			rectangle(frame, p1, p2, color, 2);
//		}
//	}
//
//	//cvRectangleR(testImg, ROI, cvScalar(255,0,0),5);
//	/*if (param_show_trails)
//	{
//		Scalar color;
//		for (unsigned int i = 0; i<lines.size(); i += 5)
//		{
//			if (faceCounted[lines[i + 4]])
//				color = Scalar(0, 0, 255);
//			else
//				color = Scalar(0, 255, 0);
//
//			line(testImg, Point(lines[i], lines[i + 1]),
//				Point(lines[i + 2], lines[i + 3]),
//				color, 1);
//		}
//	}*/
//
//	//if (param_show_lines)
//	{
//		line(frame, Point(0, param_y_line),
//			Point(testImg.cols, param_y_line),
//			Scalar(255, 255, 255), 1);
//
//		line(frame, Point(param_x_line, 0),
//			Point(param_x_line, testImg.rows),
//			Scalar(255, 255, 255), 1);
//	}
//
//	//if (param_show_count)
//	{
//		char bufcount[1000];
//		sprintf(bufcount, "People found : %d", crosscount);
//		putText(frame, bufcount, Point(20, 20),
//			FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2.0);
//	}
//
//
//	//if (param_show_video)
//	{
//		imshow("result", frame);
//	}
//
//	/*if (param_dump)
//	{
//		char buf[256];
//		sprintf(buf, "%s%08d.jpg", param_dump_prefix, numImg);
//		imwrite(buf, testImg);
//	}*/
//}

void AutoRegister()
{
	VideoCapture cap(0);
	if (!cap.isOpened()) return;
	Mat frame;
	cap >> frame;

	Mat gray(frame.size(), CV_8UC1);
	cvtColor(frame, gray, COLOR_BGR2GRAY);

	Mat Foreground(frame.size(), CV_8UC1);
	Mat Background(frame.size(), CV_8UC1);

	Mat gray32f(frame.size(), CV_32FC1);
	Mat Foreground32f(frame.size(), CV_32FC1);
	Mat Background32f(frame.size(), CV_32FC1);

	gray.convertTo(Background32f, CV_32F);

	while (true)
	{
		cap >> frame;

		cvtColor(frame, gray, COLOR_BGR2GRAY);
		imshow("foreground", gray);

		gray.convertTo(gray32f, CV_32F);

		absdiff(gray32f, Background32f, Foreground32f);
		threshold(Foreground32f, Foreground, 30, 255, THRESH_BINARY);
		accumulateWeighted(gray32f, Background32f, 0.09);
		float flmean = mean(Foreground)[0];
		if (flmean > 128)
		{
			putText(Foreground, "WARNING", Point(Foreground.cols / 2, Foreground.rows / 2), CV_FONT_ITALIC, 1, Scalar(0, 0, 255));
		}

		Background32f.convertTo(Background, CV_8U);

		imshow("background", Background);
		imshow("Binary Result", Foreground);

		if (waitKey(10) == 27)  break;
	}
}

void Sift()
{
	//Mat img_1 = imread("../data/pictures/noodle.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat img_2 = imread("../data/pictures/noodle2.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	//if (!img_1.data || !img_2.data)                                                    //如果數據為空
	//{
	//	std::cout << " --(!) Error reading images " << std::endl; return ;
	//}



	////-- Step 1: Detect the keypoints using SURF Detector     //第一步，用SIFT算子檢測關鍵點
	//int minHessian = 400;

	//Ptr<Feature2D> Surf;
	//Surf = SUR
	//SurfFeatureDetector detector(minHessian);
	//std::vector<KeyPoint> keypoints_1, keypoints_2;

	//detector.detect(img_1, keypoints_1);
	//detector.detect(img_2, keypoints_2);

	////-- Draw keypoints  //在圖像中畫出特徵點
	//Mat img_keypoints_1; Mat img_keypoints_2;

	//drawKeypoints(img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	//drawKeypoints(img_2, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

	////-- Show detected (drawn) keypoints
	//imshow("Keypoints 1", img_keypoints_1);
	//imshow("Keypoints 2", img_keypoints_2);

	////計算特徵
	//SurfDescriptorExtractor extractor;//定義對象

	//Mat descriptors_1, descriptors_2;//存放特徵向量的舉陣

	//extractor.compute(img_1, keypoints_1, descriptors_1);//計算特徵向量
	//extractor.compute(img_2, keypoints_2, descriptors_2);

	////-- Step 3: Matching descriptor vectors with a brute force matcher
	//BFMatcher matcher(NORM_L2);
	//std::vector< DMatch > matches;
	//matcher.match(descriptors_1, descriptors_2, matches);

	////-- Draw matches
	//Mat img_matches;
	//drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_matches);

	////-- Show detected matches
	//imshow("Matches", img_matches);
}

void Bgfg_segm()
{
	Mat frame;

	VideoCapture cap;
	cap.open(0);
	cap >> frame;
	Mat fg_img(frame.size(), frame.type());
	Mat fg_mask;
	bool update_bg_model = true;

	/*Ptr<BackgroundSubtractor> bg_model = createBackgroundSubtractorKNN()
		.dynamicCast<BackgroundSubtractor>();*/
	Ptr<BackgroundSubtractor> bg_model = createBackgroundSubtractorMOG2()
		.dynamicCast<BackgroundSubtractor>();

	while (true)
	{
		cap >> frame;
		bg_model->apply(frame, fg_mask, update_bg_model ? -1 : 0);

		fg_img = Scalar::all(0);
		frame.copyTo(fg_img, fg_mask);
		Mat bg_img;
		bg_model->getBackgroundImage(bg_img);

		if (!bg_img.empty())
			imshow("BG2", bg_img);

		imshow("FG2", fg_img);
		imshow("FG mask2", fg_mask);
		if (waitKey(10) == 27) break;
	}
}


void ConnectedComponent()
{
	VideoCapture cap;
	Mat frame, gray, ThresholdImage;

	cap.open(0);

	while (true)
	{
		cap >> frame;
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		threshold(gray, ThresholdImage, 30, 255, THRESH_BINARY);
		Mat labelImage(frame.size(), CV_32S);
		int nLabels = connectedComponents(ThresholdImage, labelImage, 8);
		std::vector<Vec3b> colors(nLabels);
		for (int label = 0; label < nLabels; ++label) {
			colors[label] = Vec3b((rand() & 255), (rand() & 255), (rand() & 255));
		}

		Mat ResultImage(frame.size(), CV_8UC3);
		for (int i = 0; i < ResultImage.rows; ++i) {
			for (int j = 0; j < ResultImage.cols; ++j) {
				int label = labelImage.at<int>(i, j);
				Vec3b &pixel = ResultImage.at<Vec3b>(i, j);
				pixel = colors[label];
			}
		}
		imshow("labelImage", labelImage);
		imshow("Image", frame);
		imshow("Connected Components", ResultImage);
		waitKey(1000);
	}
}

void PreProcessImage(Mat *inImage, Mat *outImage, int sizex, int sizey)
{
	Mat grayImage, blurredImage, thresholdImage, contourImage, regionOfInterest;

	vector<vector<Point> > contours;
	cvtColor(*inImage, grayImage, COLOR_BGR2GRAY);
	GaussianBlur(grayImage, blurredImage, Size(5, 5), 2, 2);
	adaptiveThreshold(blurredImage, thresholdImage, 255, 1, 1, 11, 2);
	thresholdImage.copyTo(contourImage);
	findContours(contourImage, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

	int idx = 0;
	size_t area = 0;
	for (size_t i = 0; i < contours.size(); i++)
	{
		if (area < contours[i].size())
		{
			idx = i;
			area = contours[i].size();
		}
	}

	Rect rec = boundingRect(contours[idx]);

	regionOfInterest = thresholdImage(rec);

	//erode(regionOfInterest, regionOfInterest, )
	equalizeHist(regionOfInterest, regionOfInterest);

	resize(regionOfInterest, *outImage, Size(sizex, sizey));
}

void LearnFromImages(CvMat* trainData, CvMat* trainClasses)
{
	Mat img;
	char file[255];
	for (int i = 0; i < classes; i++)
	{
		sprintf(file, "%s/%d.png", pathToImages, i);
		img.release();
		img = imread(file, 1);
		if (!img.data)
		{
			cout << "File " << file << " not found\n";
			std::exit(1);
		}
		else
		{
			imshow("img", img);
		}
		Mat outfile;
		PreProcessImage(&img, &outfile, sizex, sizey);
		for (int n = 0; n < ImageSize; n++)
		{
			trainData->data.fl[i * ImageSize + n] = outfile.data[n];
		}
		trainClasses->data.fl[i] = i;
	}

}

void RunSelfTest(Ptr<KNearest> knn2)
{
	Mat img;
	Mat sample2(Size(ImageSize, 1), CV_32FC1);
	// SelfTest
	char file[255];
	int z = 0;
	while (z++ < 10)
	{
		int iSecret = rand() % 10;
		//cout << iSecret;
		sprintf(file, "%s/%d.png", pathToImages, iSecret);
		img = imread(file, 1);
		Mat stagedImage;
		PreProcessImage(&img, &stagedImage, sizex, sizey);
		for (int n = 0; n < ImageSize; n++)
		{
			//int x = n % sizex;
			//int y = n / sizex;
			sample2.at<float>(n) = stagedImage.data[n];
		}
		Mat matResult(0, 0, CV_32FC1);
		float detectedClass = knn2->findNearest(sample2, knn2->getDefaultK(), matResult);
		detectedClass = matResult.at<float>(0);
		if (iSecret != (int)((detectedClass)))
		{
			cout << "Secret is " << iSecret << " guess answer is "
				<< (int)((detectedClass)) << endl;
			//exit(1);
		}
		else
		{
			cout << "Right Answer!! Is " << (int)((detectedClass)) << "\n";
		}
		imshow("single", img);
		//waitKey(0);
	}

}

void AnalyseImage(Ptr<KNearest> knearest, Mat image)
{

	Mat sample2(Size(ImageSize,1), CV_32FC1);

	Mat gray, blur, thresh;

	vector < vector<Point> > contours;
	//std::string tPath = pathToImages;
	//tPath.append("/buchstaben.png");
	//image = imread(tPath, 1);

	cv::cvtColor(image, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, blur, Size(5, 5), 2, 2);
	adaptiveThreshold(blur, thresh, 255, 1, 1, 11, 2);
	findContours(thresh, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

	for (size_t i = 0; i < contours.size(); i++)
	{
		vector < Point > cnt = contours[i];
		if (contourArea(cnt) > 50)
		{
			Rect rec = boundingRect(cnt);
			if (rec.height > 28)
			{
				Mat roi = image(rec);
				Mat stagedImage;
				PreProcessImage(&roi, &stagedImage, sizex, sizey);
				for (int n = 0; n < ImageSize; n++)
				{
					sample2.at<float>(n) = stagedImage.data[n];
				}
				Mat matResults(0, 0, CV_32FC1);
				float result = knearest->findNearest(sample2, knearest->getDefaultK(), matResults);
				rectangle(image, Point(rec.x, rec.y),
					Point(rec.x + rec.width, rec.y + rec.height),
					Scalar(0, 0, 255), 2);

				//imshow("all", image);
				cout << result << " ";

				//imshow("single", stagedImage);
				//waitKey(0);
			}

		}
		
	}
	cout << "\n";
	imshow("all", image);
	waitKey(0);
}








//#include "opencv2/video/tracking.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/objdetect/objdetect.hpp"
//#include "stdio.h"
//#include <vector>
//#include <map>
//#include <iostream>
//#include <fstream>
//
//
//	/*CV_IMPL CvSeq*
//	cvHaarDetectObjects2( const CvArr* _img,
//	CvHaarClassifierCascade* cascade,
//	CvMemStorage* storage, double scale_factor,
//	int min_neighbors, int flags, CvSize min_size );*/
//
//
//	using namespace std;
//using namespace cv;
//
//int inline sqr(int x) { return x*x; }
//
//// All the program is in the main function. Isn't that lovely ?
//int main(int argc, char *argv[])
//{
//
//	char * param_cascade_filename = 0;
//	char * param_infile_name = 0;
//	char * param_outfile_name = 0;
//	char * param_replayfile_name = 0;
//	char param_dump_prefix[128] = "img_";
//	bool param_grab_video = true;
//	bool param_show_video = true;
//	bool param_show_faces = true;
//	bool param_show_lines = true;
//	bool param_show_roi = true;
//	bool param_show_trails = true;
//	bool param_show_count = true;
//	bool param_dump = false;
//	int	 param_y_line = 100;
//	int  param_x_line = 80;
//	int	 param_min_face_size = 20;
//	int	 param_max_face_size = 40;
//	int	 param_neighbors = 2;
//	int  param_trail_length_threshold = 5;
//
//	char usage_string[] = "Usage: headcounter [OPTIONS] input_file \n\
//  Options:\n\
//	  --cascade filename \n\
//	  -o output_file \n\
//	  --replay filename\n\
//	  --minfacesize MINSIZE (default : 20 pixels)\n\
//	  --maxfacesize MAXSIZE (default : 40 pixels)\n\
//	  --neighbors NEIGHBORS (default : 2, higher is more discriminant)\n\
//	  --traillength min_trail_length (default : 5)\n\
//	  --novideo\n\
//	  --nodisplay\n\
//	  --noface\n\
//	  --noline\n\
//	  --noROI\n\
//	  --notrails\n\
//	  --nocount\n\
//	  --dump \n\
//	  --dumpprefix prefix\n\
//	  --linex X\n\
//	  --liney Y";
//
//	ofstream out_text_file;
//
//	// Parsing of the arguments
//	int argi = 0;
//	while (argi<argc - 1)
//	{
//		argi++;
//		if ((!strcmp(argv[argi], "--cascade")) && (argi<argc))
//		{
//			argi++;
//			param_cascade_filename = argv[argi];
//		}
//		else if ((!strcmp(argv[argi], "-o")) && (argi<argc))
//		{
//			argi++;
//			param_outfile_name = argv[argi];
//		}
//		else if ((!strcmp(argv[argi], "--replay")) && (argi<argc))
//		{
//			argi++;
//			param_replayfile_name = argv[argi];
//		}
//		else if ((!strcmp(argv[argi], "--maxfacesize")) && (argi<argc))
//		{
//			argi++;
//			param_max_face_size = atoi(argv[argi]);
//		}
//		else if ((!strcmp(argv[argi], "--minfacesize")) && (argi<argc))
//		{
//			argi++;
//			param_min_face_size = atoi(argv[argi]);
//		}
//		else if ((!strcmp(argv[argi], "--neighbors")) && (argi<argc))
//		{
//			argi++;
//			param_neighbors = atoi(argv[argi]);
//		}
//		else if ((!strcmp(argv[argi], "--traillength")) && (argi<argc))
//		{
//			argi++;
//			param_trail_length_threshold = atoi(argv[argi]);
//		}
//
//		else if (!strcmp(argv[argi], "--novideo"))
//		{
//			param_grab_video = false;
//		}
//		else if (!strcmp(argv[argi], "--nodisplay"))
//		{
//			param_show_video = false;
//		}
//		else if (!strcmp(argv[argi], "--noface"))
//		{
//			param_show_faces = false;
//		}
//		else if (!strcmp(argv[argi], "--noline"))
//		{
//			param_show_lines = false;
//		}
//		else if (!strcmp(argv[argi], "--noROI"))
//		{
//			param_show_roi = false;
//		}
//		else if (!strcmp(argv[argi], "--notrails"))
//		{
//			param_show_trails = false;
//		}
//		else if (!strcmp(argv[argi], "--nocount"))
//		{
//			param_show_trails = false;
//		}
//		else if (!strcmp(argv[argi], "--dump"))
//		{
//			param_dump = true;
//		}
//		else if ((!strcmp(argv[argi], "--dumpprefix")) && (argi<argc))
//		{
//			argi++;
//			snprintf(param_dump_prefix, 127, "%s", argv[argi]);
//		}
//		else if ((!strcmp(argv[argi], "--linex")) && (argi<argc))
//		{
//			argi++;
//			param_x_line = atoi(argv[argi]);
//		}
//		else if ((!strcmp(argv[argi], "--liney")) && (argi<argc))
//		{
//			argi++;
//			param_y_line = atoi(argv[argi]);
//		}
//		else if (param_infile_name == 0)
//		{
//			param_infile_name = argv[argi];
//		}
//		else if (!strcmp(argv[argi], "-h"))
//		{
//			std::cout << usage_string << std::endl;
//			return 0;
//		}
//		else
//		{
//			std::cerr << "Invalid argument or wrong argument numbers." << std::endl;
//			std::cout << usage_string << std::endl;
//			return 1;
//		}
//	}
//	if (param_infile_name == 0)
//	{
//		std::cerr << "Missing input filename" << std::endl;
//		std::cout << usage_string << std::endl;
//		return 1;
//	}
//
//	if (param_cascade_filename == 0)
//	{
//		param_cascade_filename = new char[1024];
//		sprintf(param_cascade_filename, "../data/haarcascade_frontalface_default.xml");
//	}
//
//	if (!param_show_video)
//	{
//		param_show_faces = false;
//		param_show_lines = false;
//		param_show_roi = false;
//		param_show_trails = false;
//		param_show_count = false;
//	}
//
//	if (!param_grab_video)
//	{
//		param_show_video = false;
//		param_show_faces = false;
//		param_show_lines = false;
//		param_show_roi = false;
//		param_show_trails = false;
//		param_show_count = false;
//	}
//	// End of parameters parsing
//
//	//This map allows to attribute a unique ID to each individual face
//	map<int, int> faceIndices;
//	int faceUID = 0;
//
//	//Tags face UIDs  that have been counted as crossing the lines
//	map<int, bool> faceCounted;
//
//	//Tags face indices that have been linked to one or two other
//	// face indices
//	map<int, int> faceLinks;
//
//	// series of triplet values of faces coordinates:
//	// center_x center_y frame#
//	vector<int> pastFaces;
//
//	// Gives the trail length of a given ID
//	map<int, int> trailLength;
//
//	// segments of trails stored as record of 5 integers:
//	// x1 y1 x2 y2 uid
//	vector<int> lines;
//
//	// stores rectangles and frame numbers
//	// #frame x y w h
//	vector<int> replayContent;
//
//	// The counter of the number of faces crossing the lines
//	int crosscount = 0;
//
//	// Current frame number
//	int numImg = 0;
//
//	//CvHaarClassifierCascade*  cascade;
//	CascadeClassifier cascade;
//
//	CvMemStorage* storage = cvCreateMemStorage(0);
//	//cascade = (CvHaarClassifierCascade*)cvLoad( param_cascade_filename, 0,0,0 );
//	cascade.load(param_cascade_filename);
//
//	VideoCapture capture;
//
//	if (param_grab_video)
//	{
//		capture.open(param_infile_name);
//
//		if (!capture.isOpened())
//		{
//			std::cerr << "Could not open video device :" << param_infile_name << std::endl;
//			return 1;
//		}
//	}
//	Mat testImg;
//	Mat gray;
//	Rect ROI;
//
//	ROI.width = -1;
//	ROI.height = 720;
//	ROI.x = 0;
//	ROI.y = 00;
//
//	if (param_show_video)
//	{
//		cvNamedWindow("result", 1);
//	}
//
//	if (param_outfile_name != 0)
//	{
//		out_text_file.open(param_outfile_name);
//	}
//
//	if (param_replayfile_name != 0)
//	{
//		FILE *fd = fopen(param_replayfile_name, "r");
//		bool endreached = false;
//		while (!endreached)
//		{
//			int num, x, y, w, h;
//			int err;
//			err = fscanf(fd, "%d %d %d %d %d\n", &num, &x, &y, &w, &h);
//			replayContent.push_back(num);
//			replayContent.push_back(x);
//			replayContent.push_back(y);
//			replayContent.push_back(w);
//			replayContent.push_back(h);
//			if (err == EOF)
//				endreached = true;
//		}
//	}
//
//	bool quit = false;
//	while (!quit)
//	{
//		if (param_grab_video)
//		{
//			capture >> testImg;
//
//			char key = cvWaitKey(10);
//			if (key == 27)
//			{
//				return 0;
//			}
//			if (ROI.width>0) testImg = testImg(ROI);
//			cvtColor(testImg, gray, CV_BGR2GRAY);
//		}
//
//
//		//cvSetImageROI(gray, ROI);
//
//
//		// rectangles found or replayes for the current frame
//		vector<Rect>	current_faces;
//
//		if (param_replayfile_name == 0)
//		{
//			vector<Rect> tmp;
//
//			cascade.detectMultiScale(gray, current_faces, 1.2
//				, param_neighbors
//				, CV_HAAR_SCALE_IMAGE
//				, Size(param_min_face_size, param_min_face_size)
//				, Size(param_max_face_size, param_max_face_size));
//		}
//		else
//		{
//			static unsigned int current_index_in_replay = 0;
//
//			while (replayContent[current_index_in_replay]<numImg)
//			{
//				current_index_in_replay += 5;
//			}
//			while (replayContent[current_index_in_replay] == numImg)
//			{
//				Rect r;
//				r.x = replayContent[current_index_in_replay + 1];
//				r.y = replayContent[current_index_in_replay + 2];
//				r.width = replayContent[current_index_in_replay + 3];
//				r.height = replayContent[current_index_in_replay + 4];
//				current_faces.push_back(r);
//				if (current_index_in_replay<replayContent.size() - 5)
//				{
//					current_index_in_replay += 5;
//				}
//				else
//				{
//					quit = true;
//					break;
//				}
//			}
//		}
//
//		static int farthest_past_value = 0;
//		static int farthest_past_index = 0;
//
//		if (numImg % 10 == 0)
//		{
//			printf("Frame#%d(%.1f sec): doing %d computations (%d * %d). Cross count so far : %d\n",
//				numImg,
//				numImg * 1 / 50.0f,
//				current_faces.size() * (pastFaces.size() - farthest_past_index),
//				current_faces.size(),
//				(pastFaces.size() - farthest_past_index),
//				crosscount);
//		}
//
//		for (unsigned int i = 0; i < current_faces.size(); i++)
//		{
//			Rect r = current_faces[i];
//			Point center;
//
//			rectangle(testImg, r, Scalar(0, 0, 255), 2, 0, 0);
//			center.x = cvRound((ROI.x + r.x + r.width*0.5));
//			center.y = cvRound((ROI.y + r.y + r.height*0.5));
//
//			pastFaces.push_back(cvRound((ROI.x + r.x + r.width*0.5)));
//			pastFaces.push_back(cvRound((ROI.y + r.y + r.height*0.5)));
//			pastFaces.push_back(numImg);
//
//
//			if (param_outfile_name != 0)
//			{
//				out_text_file << numImg << " " << r.x << " " << r.y << " " << r.width << " " << r.height << std::endl;
//			}
//
//			// Find the closest face on the last frames and link them
//			// through the use of the same face id
//			int dist_min = 100000;
//			const int FRAME_TO_FRAME_DIST_THRESHOLD = 1000;
//			const float zDimension = 1.0f;
//			int closest_x, closest_y, closest_index = 0;
//			for (unsigned int j = farthest_past_index; j<pastFaces.size(); j += 3)
//			{
//				int dist;
//
//				// The distance between two faces is an euclidian distance with
//				// time being the 3rd dimension, corrected by the zDimension constant.
//
//				dist = sqr(center.x - pastFaces[j]) + sqr(center.y - pastFaces[j + 1]) + sqr(numImg - pastFaces[j + 2])*zDimension;
//				if ((dist < dist_min) &&
//					(numImg>pastFaces[j + 2]) &&
//					(faceLinks.find(j / 3) == faceLinks.end()))
//				{
//					dist_min = dist;
//					closest_x = pastFaces[j];
//					closest_y = pastFaces[j + 1];
//					closest_index = j;
//				}
//				while (farthest_past_value + 20<pastFaces[j + 2])
//				{
//					farthest_past_index += 3;
//					farthest_past_value = pastFaces[farthest_past_index + 2];
//				}
//			}
//
//			if (dist_min < FRAME_TO_FRAME_DIST_THRESHOLD)
//			{
//				lines.push_back(closest_x);
//				lines.push_back(closest_y);
//				lines.push_back(center.x);
//				lines.push_back(center.y);
//
//				if (faceLinks.find(closest_index / 3) == faceLinks.end())
//				{
//					faceLinks[closest_index / 3] = 1;
//				}
//				else
//				{
//					faceLinks[closest_index / 3]++;
//				}
//
//				if (faceIndices.find(closest_index / 3) == faceIndices.end())
//				{
//					faceIndices[closest_index / 3] = faceUID;
//					faceIndices[(pastFaces.size() - 1) / 3] = faceUID;
//					lines.push_back(faceUID);
//					faceCounted[faceUID] = false;
//					trailLength[faceUID] = 1;
//					faceUID++;
//
//				}
//				else
//				{
//					faceIndices[(pastFaces.size() - 1) / 3] = faceIndices[closest_index / 3];
//					lines.push_back(faceIndices[closest_index / 3]);
//					trailLength[faceIndices[closest_index / 3]]++;
//					if ((faceCounted[faceIndices[closest_index / 3]]) &&
//						(trailLength[faceIndices[closest_index / 3]] == param_trail_length_threshold))
//					{
//						crosscount++;
//					}
//				}
//
//				// Check if the current link crosses one of the lines
//				if (((closest_y<param_y_line) && (center.y >= param_y_line)) ||
//					((closest_y>param_y_line) && (center.y <= param_y_line)) ||
//					((closest_x<param_x_line) && (center.x >= param_x_line)) ||
//					((closest_x>param_x_line) && (center.x <= param_x_line)))
//				{
//					if (!faceCounted[faceIndices[(pastFaces.size() - 1) / 3]])
//					{
//						faceCounted[faceIndices[(pastFaces.size() - 1) / 3]] = true;
//						if (trailLength[faceIndices[(pastFaces.size() - 1) / 3]]>param_trail_length_threshold)
//							crosscount++;
//
//						//std::cout << "cross count = " << crosscount << std::endl;
//					}
//				}
//			}
//			if (param_show_faces)
//			{
//				Scalar color;
//				if (faceIndices.find((pastFaces.size() - 1) / 3) != faceIndices.end())
//				{
//					if (trailLength[faceIndices[(pastFaces.size() - 1) / 3]]<5)
//						color = cvScalar(0, 0, 0);
//					else
//					{
//						if (faceCounted[faceIndices[(pastFaces.size() - 1) / 3]])
//							color = Scalar(0, 0, 255);
//						else
//							color = Scalar(0, 255, 0);
//					}
//				}
//				Point p1 = cvPoint(ROI.x + r.x,
//					ROI.y + r.y);
//				Point p2 = cvPoint(ROI.x + r.x + r.width,
//					ROI.y + r.y + r.height);
//
//				rectangle(testImg, p1, p2, color, 2);
//			}
//		}
//
//		//cvRectangleR(testImg, ROI, cvScalar(255,0,0),5);
//		if (param_show_trails)
//		{
//			Scalar color;
//			for (unsigned int i = 0; i<lines.size(); i += 5)
//			{
//				if (faceCounted[lines[i + 4]])
//					color = Scalar(0, 0, 255);
//				else
//					color = Scalar(0, 255, 0);
//
//				line(testImg, Point(lines[i], lines[i + 1]),
//					Point(lines[i + 2], lines[i + 3]),
//					color, 1);
//			}
//		}
//
//		if (param_show_lines)
//		{
//			line(testImg, Point(0, param_y_line),
//				Point(testImg.cols, param_y_line),
//				Scalar(255, 255, 255), 1);
//
//			line(testImg, Point(param_x_line, 0),
//				Point(param_x_line, testImg.rows),
//				Scalar(255, 255, 255), 1);
//		}
//
//		if (param_show_count)
//		{
//			char bufcount[1000];
//			sprintf(bufcount, "People found : %d", crosscount);
//			putText(testImg, bufcount, Point(20, 20),
//				FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2.0);
//		}
//
//
//		if (param_show_video)
//		{
//			imshow("result", testImg);
//		}
//
//		if (param_dump)
//		{
//			char buf[256];
//			sprintf(buf, "%s%08d.jpg", param_dump_prefix, numImg);
//			imwrite(buf, testImg);
//		}
//
//		numImg++;
//	}
//	if (param_outfile_name)
//	{
//		out_text_file.close();
//	}
//}