#include "ImageProcessor.h"

ImageProcess::ImageProcess()
{
	bFirstFrame = true;
	pathToImages = "../data/pictures";
	if (!face_cascade.load(FACE_CASCADE_NAME))
	{
		printf("--(!)Error loading\n");
	}
	if (!faceside_cascade.load(FACE_SIDELOOKING_CASCADE_NAME))
	{
		printf("--(!)Error loading\n");
	}
	if (!bodyupper_cascade.load(UPPERBODY_CASCADE_NAME))
	{
		printf("--(!)Error loading\n");
	}
	if (!bodyfull_cascade.load(FULLBODY_CASCADE_NAME))
	{
		printf("--(!)Error loading\n");
	}
	if (!head_cascade.load(HEAD_CASCADE_NAME))
	{
		printf("--(!)Error loading\n");
	}
	if (!eyes_cascade.load(EYES_CASCADE_NAME))
	{
		printf("--(!)Error loading\n");
	}
	if (!plate_cascade.load(PLATE_CASCADE_NAME))
	{
		printf("--(!)Error loading\n");
	}
}

ImageProcess::~ImageProcess() {}

void ImageProcess::PreProcessImage(Mat *inImage, Mat *outImage, int sizex, int sizey)
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

	Rect rec = contours.size() == 0 ? Rect(0, 0, thresholdImage.cols, thresholdImage.rows) : boundingRect(contours[idx]);

	regionOfInterest = thresholdImage(rec);

	//erode(regionOfInterest, regionOfInterest, )
	equalizeHist(regionOfInterest, regionOfInterest);

	resize(regionOfInterest, *outImage, Size(sizex, sizey));
}

void ImageProcess::LearnFromImages(CvMat* trainData, CvMat* trainClasses)
{
	Mat img;
	char file[255];
	for (int i = 0; i < classes; i++)
	{
		sprintf(file, "%s/_%d.png", pathToImages, i);
		img.release();
		img = imread(file, 1);
		if (!img.data)
		{
			cout << "File " << file << " not found\n";
			std::exit(1);
		}
		else
		{
			//imshow("img", img);
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

void ImageProcess::RunSelfTest(Ptr<KNearest> knn2)
{
	Mat img;
	Mat sample2(Size(ImageSize, 1), CV_32FC1);
	// SelfTest
	char file[255];
	int z = 0;
	while (z++ < 20)
	{
		int iSecret = rand() % 36;
		//cout << iSecret;
		sprintf(file, "%s/_%d.png", pathToImages, iSecret);
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
		//imshow("single", img);
		//waitKey(0);
	}

}


String ImageProcess::AnalyseImage(Ptr<KNearest> knearest, Mat image)
{
	String strRet = "";
	vector<Rect> plates_second;
	Mat plate, _Temp = image;
	//迭代幾次找到最吻合的偵測區域
	int iter = 5;
	for (int i = 0; i < iter; i++) {
		plate_cascade.detectMultiScale(_Temp, plates_second, 1.1, 2, 0, Size(30, 30));
		_Temp = plates_second.size() > 0 ? _Temp(plates_second[0]) : _Temp;
		if (i == iter - 1) {
			plate = _Temp;
		}
	}


	Mat sample2(Size(ImageSize, 1), CV_32FC1);

	Mat resized, gray, blur, thresh, thresh2;

	vector < vector<Point> > contours;

	imshow("plate", plate);
	waitKey();
	//轉成 75 X 228
	resize(plate, resized, Size(228, 75), 0, 0, InterpolationFlags::INTER_CUBIC);
	//轉灰階
	cv::cvtColor(resized, gray, COLOR_BGR2GRAY);
	//模糊化去雜訊
	GaussianBlur(gray, blur, Size(5, 5), 2, 2);
	//計算圖片整體平均亮度
	Scalar meanValue = mean(blur);
	double Bright = -((meanValue[0] - 128) / 255 * 100);
	//直方圖等化
	equalizeHist(blur, blur);


	//提高亮度對比
	Mat LookupTableData(1, 256, CV_8U);//建立查表
	double Contrast = 30;
	double Brightness = Bright;
	if (Contrast > 0) {
		double Delta = 127 * Contrast / 100;
		double a = 255 / (255 - Delta * 2);
		double b = a * (Brightness - Delta);
		for (int x = 0; x < 256; x++) {
			int y = (int)(a*x + b);
			if (y < 0) y = 0;
			if (y > 255) y = 255;

			LookupTableData.at<uchar>(0, x) = (uchar)y;
		}
	}
	else {
		double Delta = -128 * Contrast / 100;
		double a = (256 - Delta * 2) / 255;
		double b = a * Brightness + Delta;
		for (int x = 0; x < 256; x++) {
			int y = (int)(a*x + b);
			if (y < 0) y = 0;
			if (y > 255) y = 255;

			LookupTableData.at<uchar>(0, x) = (uchar)y;
		}
	}
	LUT(blur, LookupTableData, blur);


	//imshow("LUT", blur);

	adaptiveThreshold(blur, thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 7, 0);

	findContours(thresh, contours, CV_RETR_LIST, CHAIN_APPROX_SIMPLE);

	vector < Rect > bigRects;
	//濾掉太小的
	for (size_t i = 0; i < contours.size(); i++) {
		vector < Point > cnt = contours[i];
		if (contourArea(cnt) > 50)
		{
			Rect rec = boundingRect(cnt);
			if (rec.height > thresh.rows / 3)
			{
				bigRects.push_back(rec);
			}
		}
	}
	//畫出未篩選的分割
	for (int i = 0; i < bigRects.size(); i++) {
		Rect rec = bigRects[i];
		rectangle(plate, Point(rec.x, rec.y),
			Point(rec.x + rec.width, rec.y + rec.height),
			Scalar(255, 0, 0), 3);
	}
	imshow("unfilter", thresh);
	waitKey();

	//濾掉包含其他的	
	vector < Rect > smallRects;
	for (size_t i = 0; i < bigRects.size(); i++) {
		bool bHaveInclude = false;
		for (size_t j = 0; j < bigRects.size(); j++) {
			if (i != j) {
				Rect unionRect = bigRects[i] | bigRects[j];
				if (unionRect == bigRects[i]) {
					bHaveInclude = true;
					break;
				}
			}
		}
		if (!bHaveInclude) {
			smallRects.push_back(bigRects[i]);
		}
	}


	//x排序
	for (size_t i = 0; i < smallRects.size(); i++) {
		double minVal = smallRects[i].x;
		size_t swapIndex = i;
		for (size_t j = i + 1; j < smallRects.size(); j++)
		{
			if (smallRects[j].x < minVal) {
				minVal = smallRects[j].x;
				swapIndex = j;
			}
		}
		if (swapIndex >= 0 && swapIndex != i) {
			//swap
			Rect temp = smallRects[i];
			smallRects[i] = smallRects[swapIndex];
			smallRects[swapIndex] = temp;
		}
	}

	//濾掉誤判的
	vector < Rect > workRects;
	for (size_t i = 0; i < smallRects.size(); i++) {
		if (i == 0 || i == smallRects.size() - 1) {
			workRects.push_back(smallRects[i]);
		}
		else {
			Rect Left = smallRects[i] & smallRects[i - 1];
			Rect Right = smallRects[i] & smallRects[i + 1];
			int TopY = smallRects[i].y;
			int BottomY = TopY - smallRects[i].height;
			int lTopY = smallRects[i - 1].y;
			int lBottomY = lTopY - smallRects[i - 1].height;
			int rTopY = smallRects[i + 1].y;
			int rBottomY = rTopY - smallRects[i + 1].height;
			if (Left.area() + Right.area() > smallRects[i].area() / 2)
			{
				//濾掉
			}
			else if (TopY < lBottomY && TopY < rBottomY) {
				//
			}
			else if (BottomY > lTopY && lBottomY > rTopY) {

			}
			else {
				workRects.push_back(smallRects[i]);
			}
		}
	}



	bool bWordFirst = true;
	for (size_t i = 0; i < workRects.size(); i++)
	{
		Rect rec = workRects[i];
		Mat roi = resized(rec);
		Mat stagedImage;
		PreProcessImage(&roi, &stagedImage, sizex, sizey);
		//resize(roi, stagedImage, Size(sizex, sizey));
		//dilation
		Mat Element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(1, 1));
		morphologyEx(stagedImage, stagedImage, MorphTypes::MORPH_CLOSE, Element, Point(1, 1), 1);
		for (int n = 0; n < ImageSize; n++)
		{
			sample2.at<float>(n) = stagedImage.data[n];
		}
		Mat matResults(0, 0, CV_32FC1);
		float result = knearest->findNearest(sample2, knearest->getDefaultK(), matResults);
		rectangle(resized, Point(rec.x, rec.y),
			Point(rec.x + rec.width, rec.y + rec.height),
			Scalar(0, 0, 255), 2);

		//imshow("all", image);
		//語意分析
		char c;
		if (result >= 10)
		{
			if (i == 0) bWordFirst = true;
			result += 55;
		}
		else {
			if (i == 0) bWordFirst = false;
			result += 48;
		}
		//
		if (result == 68 || result == 79 || result == 85)
		{
			if (bWordFirst && i > 2) {
				result = 48; //數字0
			}
			else if (!bWordFirst && i < 4) {
				result = 48;
			}
		}
		cout << (char)result << " ";
		strRet += (char)result;
		//imshow("single", stagedImage);
		//waitKey(0);
	}
	//for (size_t i = 0; i < contours.size(); i++)
	//{
	//	vector < Point > cnt = contours[i];
	//	if (contourArea(cnt) > 50)
	//	{
	//		Rect rec = boundingRect(cnt);
	//		if (rec.height > 28)
	//		{
	//			Mat roi = image(rec);
	//			Mat stagedImage;
	//			PreProcessImage(&roi, &stagedImage, sizex, sizey);
	//			for (int n = 0; n < ImageSize; n++)
	//			{
	//				sample2.at<float>(n) = stagedImage.data[n];
	//			}
	//			Mat matResults(0, 0, CV_32FC1);
	//			float result = knearest->findNearest(sample2, knearest->getDefaultK(), matResults);
	//			rectangle(image, Point(rec.x, rec.y),
	//				Point(rec.x + rec.width, rec.y + rec.height),
	//				Scalar(0, 0, 255), 2);

	//			//imshow("all", image);
	//			cout << result << " ";

	//			//imshow("single", stagedImage);
	//			//waitKey(0);
	//		}

	//	}
	//	
	//}
	//cout << "\n";
	imshow("all", resized);
	waitKey(0);
	return strRet;
}

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

void ImageProcess::read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator) {
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
	Mat img_resized,frame_gray;
	vector<Rect>Pedestrians, Pedestrains_Filterd;
	HOGDescriptor hog;
	//preprocess
	resize(frame, img_resized, Size(640, 480), 0, 0, InterpolationFlags::INTER_CUBIC);
	cvtColor(img_resized, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//svm
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
	
	//偵測
	hog.detectMultiScale(frame_gray, Pedestrians, 0, Size(8, 8), Size(16, 16), 1.05, 2);
	
	//過濾重疊
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
		rectangle(img_resized, r.tl(), r.br(), cv::Scalar(255, 255, 0), 3);
	}	
	imshow("result", img_resized);
	waitKey();
	return (int)Pedestrains_Filterd.size();
}
int ImageProcess::Count_Face_Num(Mat frame)
{
	Mat img_resized, frame_gray;
	vector<Rect>faces;

	resize(frame, img_resized, Size(640, 480), 0, 0, InterpolationFlags::INTER_CUBIC);
	cvtColor(img_resized, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0, Size(30, 30));
	for (int i = 0; i < faces.size(); i++)
	{
		rectangle(img_resized, faces[i], Scalar(0, 0, 255), 2);
	}
	imshow("result", img_resized);
	waitKey();
	return (int)faces.size();
}
bool ImageProcess::Invasion_Detect(Mat frame)
{
	bool bRet = false;
	Mat gray(frame.size(), CV_8UC1);
	Mat Foreground(frame.size(), CV_8UC1);
	Mat Background(frame.size(), CV_8UC1);

	Mat gray32f(frame.size(), CV_32FC1);
	Mat Foreground32f(frame.size(), CV_32FC1);
	Mat Background32f(frame.size(), CV_32FC1);

	if (bFirstFrame) {
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		gray.convertTo(Background32f, CV_32F);
	}

	cvtColor(frame, gray, COLOR_BGR2GRAY);
	imshow("foreground", gray);

	gray.convertTo(gray32f, CV_32F);

	absdiff(gray32f, Background32f, Foreground32f);
	threshold(Foreground32f, Foreground, 30, 255, THRESH_BINARY);
	accumulateWeighted(gray32f, Background32f, 0.09);
	float flmean = mean(Foreground)[0];
	if (flmean > 128)
	{
		bRet = true;
		putText(Foreground, "WARNING", Point(Foreground.cols / 2, Foreground.rows / 2), CV_FONT_ITALIC, 1, Scalar(0, 0, 255));
	}

	Background32f.convertTo(Background, CV_8U);

	imshow("background", Background);
	imshow("Binary Result", Foreground);	

	return bRet;
}
String ImageProcess::Plate_Detect(Mat frame)
{
	String strRet = "";
	
	CvMat* trainData = cvCreateMat(classes * train_samples, ImageSize, CV_32FC1);
	CvMat* trainClasses = cvCreateMat(classes * train_samples, 1, CV_32FC1);


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

	Mat gray(frame.size(), CV_8UC1);
	cvtColor(frame, gray, COLOR_BGR2GRAY);
	
	equalizeHist(gray, gray);
	vector<Rect>plates;
	plate_cascade.detectMultiScale(gray, plates, 1.1, 2, 0, Size(30, 30));
	for (int i = 0; i < plates.size(); i++)
	{
		rectangle(frame, plates[i], Scalar(0, 255, 0), 2);
	}
	imshow("gray", gray);
	waitKey();
	if (plates.size() > 0) {

		//選最大範圍的
		Rect ROI_Best;
		double area = 0;
		for (int i = 0; i < plates.size(); i++) {
			if (plates[i].area() > area) {
				ROI_Best = plates[i];
				area = plates[i].area();
			}
		}
		//最後要計算的車牌
		Mat plate = frame(ROI_Best);

		//分析
		strRet = AnalyseImage(kclassifier, plate);
		//AnalyseImage_SCW(kclassifier, plate);
	}
	return strRet;
}
bool ImageProcess::Face_Recognition(Mat target, Mat background)
{
	bool bRet = true;
	Mat img_1 = target;
	Mat img_2 = background;

	if (!img_1.data || !img_2.data)                                                    //如果數據為空
	{
		std::cout << " --(!) Error reading images " << std::endl; return false;
	}

	face::EigenFaceRecognizer::create();

	//-- Step 1: Detect the keypoints using SURF Detector     //第一步，用SIFT算子檢測關鍵點
	int minHessian = 400;



	Ptr<FeatureDetector> detector = FastFeatureDetector::create(15);
	//SurfFeatureDetector detector(minHessian);
	std::vector<KeyPoint> keypoints_1, keypoints_2;

	detector->detect(img_1, keypoints_1);
	detector->detect(img_2, keypoints_2);

	//-- Draw keypoints  //在圖像中畫出特徵點
	Mat img_keypoints_1; Mat img_keypoints_2;

	drawKeypoints(img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(img_2, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

	//-- Show detected (drawn) keypoints
	imshow("Keypoints 1", img_keypoints_1);
	imshow("Keypoints 2", img_keypoints_2);

	//計算特徵
	Ptr<SURF> extractor = SURF::create(minHessian);
	//SurfDescriptorExtractor extractor;//定義對象

	Mat descriptors_1, descriptors_2;//存放特徵向量的舉陣

	extractor->compute(img_1, keypoints_1, descriptors_1);//計算特徵向量
	extractor->compute(img_2, keypoints_2, descriptors_2);

	//-- Step 3: Matching descriptor vectors with a brute force matcher
	BFMatcher matcher(NORM_L2);
	std::vector< DMatch > matches;
	matcher.match(descriptors_1, descriptors_2, matches);

	//-- Draw matches
	Mat img_matches;
	drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_matches);

	//-- Step 3: Matching descriptor vectors using FLANN matcher
	cv::FlannBasedMatcher matcher2;
	std::vector< DMatch > matches2;
	matcher2.match(descriptors_1, descriptors_2, matches2);

	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		double dist = matches2[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);

	//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist )
	//-- PS.- radiusMatch can also be used here.
	std::vector< DMatch > good_matches;

	for (int i = 0; i < descriptors_1.rows; i++)
	{
		if (matches2[i].distance <= 2 * min_dist)
		{
			good_matches.push_back(matches2[i]);
		}
	}

	//-- Localize the object
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for (int i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(keypoints_1[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints_2[good_matches[i].trainIdx].pt);
	}

	Mat H = findHomography(obj, scene, CV_RANSAC);

	//-- Get the corners from the image_1 ( the object to be "detected" )
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(img_1.cols, 0);
	obj_corners[2] = cvPoint(img_1.cols, img_1.rows); obj_corners[3] = cvPoint(0, img_1.rows);
	std::vector<Point2f> scene_corners(4);

	perspectiveTransform(obj_corners, scene_corners, H);

	//太小就當作判斷錯誤
	if (contourArea(scene_corners) < contourArea(obj_corners) / 2)
	{
		bRet = false;
	}

	//-- Draw only "good" matches
	//Mat img_matches2;
	drawMatches(img_1, keypoints_1, img_2, keypoints_2,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
	line(img_matches, scene_corners[0] + Point2f(img_1.cols, 0), scene_corners[1] + Point2f(img_1.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[1] + Point2f(img_1.cols, 0), scene_corners[2] + Point2f(img_1.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[2] + Point2f(img_1.cols, 0), scene_corners[3] + Point2f(img_1.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[3] + Point2f(img_1.cols, 0), scene_corners[0] + Point2f(img_1.cols, 0), Scalar(0, 255, 0), 4);

	//-- Show detected matches  
	imshow("Good Matches", img_matches);
	waitKey();
	return bRet;
}