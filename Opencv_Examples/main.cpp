#include <iostream>
#include <vector>
#include "main.h"
#include "ImageProcessor.h"
using namespace cv;
using namespace cv::ml;
using namespace xfeatures2d;
using namespace cv::face;
using namespace std;




/** Global variables */
enum FACE_RECOGNITION_TYPE {EIGEN = 0, FISHER, LBPH};
String face_cascade_name = "../data/haarcascade_frontalface_alt.xml";
String face_sidelooking_cascade_name = "../data/haarcascade_profileface.xml";
String upperbody_cascade_name = "../data/haarcascade_upperbody.xml";
String fullbody_cascade_name = "../data/haarcascade_fullbody.xml";
String eyes_cascade_name = "../data/haarcascade_eye.xml";
String head_cascade_name = "../data/haarcascade_head.xml";
String plate_cascade_name = "../data/haarcascade_licence_plate_us.xml";
CascadeClassifier face_cascade;
CascadeClassifier faceside_cascade;
CascadeClassifier bodyupper_cascade;
CascadeClassifier bodyfull_cascade;
CascadeClassifier eyes_cascade;
CascadeClassifier head_cascade;
CascadeClassifier plate_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);
//下載　https://github.com/opencv/opencv/tree/master/data/haarcascades

const int train_samples = 1;
const int classes = 36;
const int sizex = 20;
const int sizey = 30;
const int ImageSize = sizex * sizey;
char pathToImages[] = "../data/pictures";


static Mat norm_0_255(InputArray _src) {
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

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
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

Ptr<EigenFaceRecognizer> TrainingEigenFaceRecognizer();
Ptr<FisherFaceRecognizer> TrainingFisherFaceRecognizer();
Ptr<LBPHFaceRecognizer> TrainingLBPHFaceRecognizer();
//excercises
void Fece_Count();
void Face_EigenFaceRecoginition();
void Face_FisherFaceRecoginition();
void Face_LPBFaceRecoginition();
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
void AnalyseImage_SCW(Ptr<KNearest> knearest, Mat frame);


int main() {

	Mat ImagePeople = imread("../data/pictures/people/people.jpg", CV_LOAD_IMAGE_COLOR);
	Mat ImagePeopleFace = imread("../data/pictures/people/face.jpg", CV_LOAD_IMAGE_COLOR);
	Mat ImagePlate = imread("../data/pictures/plate/plate.jpg", CV_LOAD_IMAGE_COLOR);
	Mat ImageFaceObject = imread("../data/pictures/man_template.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat ImageFaceBackground = imread("../data/pictures/man2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	ImageProcess imgProcessor;
	
	//cout << "../data/pictures/people/people.jpg" << ", 人數: " << imgProcessor.Count_People_Num(ImagePeople) << endl;
	//cout << "../data/pictures/people/people2.jpg" << ", 人數: " << imgProcessor.Count_Face_Num(ImagePeopleFace) << endl;
	cout << "../data/pictures/people/people.jpg" << ", 車牌號碼: " << imgProcessor.Plate_Detect(ImagePlate) << endl;
	//char* faceRecognitionResult = imgProcessor.Face_Recognition(ImageFaceObject, ImageFaceBackground) ? "存在" : "不存在";
	//cout << "../data/pictures/people/people.jpg" << ", 人臉偵測: " << faceRecognitionResult << endl;

	//VideoCapture cap(0);
	//Mat frame;
	//cap >> frame;
	//imgProcessor.Invasion_Begin(frame);
	//while (true) {
	//	cap >> frame;
	//	char *ans = imgProcessor.Invasion_Detect(frame) ? "有異物入侵!!" : "未入侵";
	//	cout << "入侵偵測:" << ans << endl;
	//	if (waitKey(100) == 27) break;
	//}
	


	/*if (!face_cascade.load(face_cascade_name))
	{
		printf("--(!)Error loading\n");
		return -1;
	}
	if (!faceside_cascade.load(face_sidelooking_cascade_name))
	{
		printf("--(!)Error loading\n");
		return -1;
	}
	if (!bodyupper_cascade.load(upperbody_cascade_name))
	{
		printf("--(!)Error loading\n");
		return -1;
	}
	if (!bodyfull_cascade.load(fullbody_cascade_name))
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
	}*/
	//namedWindow("img", WINDOW_AUTOSIZE);

	//Fece_Count();
	//Bgfg_segm();
	//AutoRegister();
	//Face_detectAndDisplay();
	//Sift();
	//Face_EigenFaceRecoginition();
	//Face_FisherFaceRecoginition();
	//Face_LPBFaceRecoginition();
	//ConnectedComponent();
	//Plate_dectecAndDisplay();
	return 0;
}

void Fece_Count()
{
	VideoCapture cap(0);             //開啟攝影機
	if (!cap.isOpened()) return;   //確認攝影機打開
	Mat frame;                       //用矩陣紀錄抓取的每張frame
	//利用hog來偵測行人
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
		//偵測
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
			rectangle(frame, r.tl(), r.br(), cv::Scalar(0, 255, 0), 3);
		}


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
		//for (int i = 0; i<faces.size(); i++)
		//{
		//	Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
		//	ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
		//	Mat faceROI = frame_gray(faces[i]);
		//	std::vector<Rect> eyes;
		//	//-- In each face, detect eyes
		//	eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0, Size(30, 30));
		//	for (int j = 0; j < eyes.size(); j++)
		//	{
		//		Point center(faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5);
		//		int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
		//		circle(frame, center, radius, Scalar(255, 0, 0), 4, 8, 0);
		//	}
		//	//if (eyes.size() > 0)
		//	{
		//		char _text[100] = { 0 };
		//		String text = _itoa(faces.size(), _text, 10);
		//		text = "人數:" + text;
		//		putText(frame, text, Point(0, 15), CV_FONT_ITALIC, 2.0, Scalar(0, 0, 255), 2);
		//	}
		//}
		imshow(window_name, frame);                //建立一個視窗,顯示frame到camera名稱的視窗
		if (waitKey(30) == 27) break;  //按鍵就離開程式
	}
}
void Face_EigenFaceRecoginition()
{
	String fn_csv = "../data/at.txt";
	// These vectors hold the images and corresponding labels.
	vector<Mat> images;
	vector<int> labels;
	// Read in the data. This can fail if no valid
	// input filename is given.
	try {
		read_csv(fn_csv, images, labels);
	}
	catch (cv::Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		// nothing more we can do
		exit(1);
	}
	// Quit if there are not enough images for this demo.
	if (images.size() <= 1) {
		string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
		CV_Error(CV_StsError, error_message);
	}

	// Get the height from the first image. We'll need this
	// later in code to reshape the images to their original
	// size:
	int height = images[0].rows;
	// The following lines simply get the last images from
	// your dataset and remove it from the vector. This is
	// done, so that the training data (which we learn the
	// cv::FaceRecognizer on) and the test data we test
	// the model with, do not overlap.
	Mat testSample = images[images.size() - 1];
	int testLabel = labels[labels.size() - 1];
	images.pop_back();
	labels.pop_back();
	// The following lines create an Fisherfaces model for
	// face recognition and train it with the images and
	// labels read from the given CSV file.
	// If you just want to keep 10 Fisherfaces, then call
	// the factory method like this:
	//
	//      cv::createFisherFaceRecognizer(10);
	//
	// However it is not useful to discard Fisherfaces! Please
	// always try to use _all_ available Fisherfaces for
	// classification.
	//
	// If you want to create a FaceRecognizer with a
	// confidence threshold (e.g. 123.0) and use _all_
	// Fisherfaces, then call it with:
	//
	//      cv::createFisherFaceRecognizer(0, 123.0);
	//
	Ptr<EigenFaceRecognizer> model = face::EigenFaceRecognizer::create();
	model->train(images, labels);
	// The following line predicts the label of a given
	// test image:
	int predictedLabel = model->predict(testSample);
	//
	// To get the confidence of a prediction call the model with:
	//
	//      int predictedLabel = -1;
	//      double confidence = 0.0;
	//      model->predict(testSample, predictedLabel, confidence);
	//
	string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
	cout << result_message << endl;
	// Here is how to get the eigenvalues of this Eigenfaces model:
	Mat eigenvalues = model->getEigenValues();
	// And we can do the same to display the Eigenvectors (read Eigenfaces):
	Mat W = model->getEigenVectors();
	// Get the sample mean from the training data
	Mat mean = model->getMean();
	// Display or save:
	imshow("mean", norm_0_255(mean.reshape(1, images[0].rows)));
	//if (argc == 2) {
	//	imshow("mean", norm_0_255(mean.reshape(1, images[0].rows)));
	//}
	//else {
	//	imwrite(format("%s/mean.png", output_folder.c_str()), norm_0_255(mean.reshape(1, images[0].rows)));
	//}
	// Display or save the first, at most 16 Fisherfaces:
	for (int i = 0; i < min(10, W.cols); i++) {
		string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
		cout << msg << endl;
		// get eigenvector #i
		Mat ev = W.col(i).clone();
		// Reshape to original size & normalize to [0...255] for imshow.
		Mat grayscale = norm_0_255(ev.reshape(1, height));
		// Show the image & apply a Jet colormap for better sensing.
		Mat cgrayscale;
		applyColorMap(grayscale, cgrayscale, COLORMAP_JET);
		// Display or save:
		imshow(format("fisherface_%d", i), cgrayscale);
	/*	if (argc == 2) {
			imshow(format("fisherface_%d", i), cgrayscale);
		}
		else {
			imwrite(format("%s/fisherface_%d.png", output_folder.c_str(), i), norm_0_255(cgrayscale));
		}*/
	}
	// Display or save the image reconstruction at some predefined steps:
	for (int num_components = min(W.cols, 10); num_components < min(W.cols, 300); num_components += 15) {
		// slice the eigenvectors from the model
		Mat evs = Mat(W, Range::all(), Range(0, num_components));
		Mat projection = LDA::subspaceProject(evs, mean, images[0].reshape(1, 1));
		Mat reconstruction = LDA::subspaceReconstruct(evs, mean, projection);
		// Normalize the result:
		reconstruction = norm_0_255(reconstruction.reshape(1, images[0].rows));
		// Display or save:
		imshow(format("fisherface_reconstruction_%d", num_components), reconstruction);

		/*if (argc == 2) {
			imshow(format("fisherface_reconstruction_%d", num_component), reconstruction);
		}
		else {
			imwrite(format("%s/fisherface_reconstruction_%d.png", output_folder.c_str(), num_component), reconstruction);
		}*/
	}
	// Display if we are not writing to an output folder:
	waitKey(0);

	/*if (argc == 2) {
		waitKey(0);
	}*/
}
void Face_FisherFaceRecoginition()
{
	String fn_csv = "../data/at.txt";
	// These vectors hold the images and corresponding labels.
	vector<Mat> images;
	vector<int> labels;
	// Read in the data. This can fail if no valid
	// input filename is given.
	try {
		read_csv(fn_csv, images, labels);
	}
	catch (cv::Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		// nothing more we can do
		exit(1);
	}
	// Quit if there are not enough images for this demo.
	if (images.size() <= 1) {
		string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
		CV_Error(CV_StsError, error_message);
	}

	// Get the height from the first image. We'll need this
	// later in code to reshape the images to their original
	// size:
	int height = images[0].rows;
	// The following lines simply get the last images from
	// your dataset and remove it from the vector. This is
	// done, so that the training data (which we learn the
	// cv::FaceRecognizer on) and the test data we test
	// the model with, do not overlap.
	Mat testSample = images[images.size() - 1];
	int testLabel = labels[labels.size() - 1];
	images.pop_back();
	labels.pop_back();
	// The following lines create an Fisherfaces model for
	// face recognition and train it with the images and
	// labels read from the given CSV file.
	// If you just want to keep 10 Fisherfaces, then call
	// the factory method like this:
	//
	//      cv::createFisherFaceRecognizer(10);
	//
	// However it is not useful to discard Fisherfaces! Please
	// always try to use _all_ available Fisherfaces for
	// classification.
	//
	// If you want to create a FaceRecognizer with a
	// confidence threshold (e.g. 123.0) and use _all_
	// Fisherfaces, then call it with:
	//
	//      cv::createFisherFaceRecognizer(0, 123.0);
	//
	Ptr<FisherFaceRecognizer> model = face::FisherFaceRecognizer::create();
	model->train(images, labels);
	// The following line predicts the label of a given
	// test image:
	int predictedLabel = model->predict(testSample);
	//
	// To get the confidence of a prediction call the model with:
	//
	//      int predictedLabel = -1;
	//      double confidence = 0.0;
	//      model->predict(testSample, predictedLabel, confidence);
	//
	string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
	cout << result_message << endl;
	// Here is how to get the eigenvalues of this Eigenfaces model:
	Mat eigenvalues = model->getEigenValues();
	// And we can do the same to display the Eigenvectors (read Eigenfaces):
	Mat W = model->getEigenVectors();
	// Get the sample mean from the training data
	Mat mean = model->getMean();
	// Display or save:
	imshow("mean", norm_0_255(mean.reshape(1, images[0].rows)));

	/*if (argc == 2) {
		imshow("mean", norm_0_255(mean.reshape(1, images[0].rows)));
	}
	else {
		imwrite(format("%s/mean.png", output_folder.c_str()), norm_0_255(mean.reshape(1, images[0].rows)));
	}*/
	// Display or save the first, at most 16 Fisherfaces:
	for (int i = 0; i < min(16, W.cols); i++) {
		string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
		cout << msg << endl;
		// get eigenvector #i
		Mat ev = W.col(i).clone();
		// Reshape to original size & normalize to [0...255] for imshow.
		Mat grayscale = norm_0_255(ev.reshape(1, height));
		// Show the image & apply a Bone colormap for better sensing.
		Mat cgrayscale;
		applyColorMap(grayscale, cgrayscale, COLORMAP_BONE);
		// Display or save:
		imshow(format("fisherface_%d", i), cgrayscale);

		/*if (argc == 2) {
			imshow(format("fisherface_%d", i), cgrayscale);
		}
		else {
			imwrite(format("%s/fisherface_%d.png", output_folder.c_str(), i), norm_0_255(cgrayscale));
		}*/
	}
	// Display or save the image reconstruction at some predefined steps:
	for (int num_component = 0; num_component < min(16, W.cols); num_component++) {
		// Slice the Fisherface from the model:
		Mat ev = W.col(num_component);
		Mat projection = LDA::subspaceProject(ev, mean, images[0].reshape(1, 1));
		Mat reconstruction = LDA::subspaceReconstruct(ev, mean, projection);
		// Normalize the result:
		reconstruction = norm_0_255(reconstruction.reshape(1, images[0].rows));
		// Display or save:
		imshow(format("fisherface_reconstruction_%d", num_component), reconstruction);

		/*if (argc == 2) {
			imshow(format("fisherface_reconstruction_%d", num_component), reconstruction);
		}
		else {
			imwrite(format("%s/fisherface_reconstruction_%d.png", output_folder.c_str(), num_component), reconstruction);
		}*/
	}
	// Display if we are not writing to an output folder:
	waitKey(0);
	/*if (argc == 2) {
		waitKey(0);
	}*/
}
void Face_LPBFaceRecoginition()
{
	String fn_csv = "../data/at.txt";
	// These vectors hold the images and corresponding labels.
	vector<Mat> images;
	vector<int> labels;
	// Read in the data. This can fail if no valid
	// input filename is given.
	try {
		read_csv(fn_csv, images, labels);
	}
	catch (cv::Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		// nothing more we can do
		exit(1);
	}
	// Quit if there are not enough images for this demo.
	if (images.size() <= 1) {
		string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
		CV_Error(CV_StsError, error_message);
	}

	// Get the height from the first image. We'll need this
	// later in code to reshape the images to their original
	// size:
	int height = images[0].rows;
	// The following lines simply get the last images from
	// your dataset and remove it from the vector. This is
	// done, so that the training data (which we learn the
	// cv::FaceRecognizer on) and the test data we test
	// the model with, do not overlap.
	Mat testSample = images[images.size() - 1];
	int testLabel = labels[labels.size() - 1];
	images.pop_back();
	labels.pop_back();
	// The following lines create an LBPH model for
	// face recognition and train it with the images and
	// labels read from the given CSV file.
	//
	// The LBPHFaceRecognizer uses Extended Local Binary Patterns
	// (it's probably configurable with other operators at a later
	// point), and has the following default values
	//
	//      radius = 1
	//      neighbors = 8
	//      grid_x = 8
	//      grid_y = 8
	//
	// So if you want a LBPH FaceRecognizer using a radius of
	// 2 and 16 neighbors, call the factory method with:
	//
	//      cv::createLBPHFaceRecognizer(2, 16);
	//
	// And if you want a threshold (e.g. 123.0) call it with its default values:
	//
	//      cv::createLBPHFaceRecognizer(1,8,8,8,123.0)
	//
	Ptr<LBPHFaceRecognizer> model = face::LBPHFaceRecognizer::create();
	model->train(images, labels);
	// The following line predicts the label of a given
	// test image:
	int predictedLabel = model->predict(testSample);
	//
	// To get the confidence of a prediction call the model with:
	//
	//      int predictedLabel = -1;
	//      double confidence = 0.0;
	//      model->predict(testSample, predictedLabel, confidence);
	//
	string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
	cout << result_message << endl;
	// Sometimes you'll need to get/set internal model data,
	// which isn't exposed by the public cv::FaceRecognizer.
	// Since each cv::FaceRecognizer is derived from a
	// cv::Algorithm, you can query the data.
	//
	// First we'll use it to set the threshold of the FaceRecognizer
	// to 0.0 without retraining the model. This can be useful if
	// you are evaluating the model:
	//
	model->setThreshold(0.0);
	// Now the threshold of this model is set to 0.0. A prediction
	// now returns -1, as it's impossible to have a distance below
	// it
	predictedLabel = model->predict(testSample);
	cout << "Predicted class = " << predictedLabel << endl;
	// Show some informations about the model, as there's no cool
	// Model data to display as in Eigenfaces/Fisherfaces.
	// Due to efficiency reasons the LBP images are not stored
	// within the model:
	cout << "Model Information:" << endl;
	string model_info = format("\tLBPH(radius=%i, neighbors=%i, grid_x=%i, grid_y=%i, threshold=%.2f)",
		model->getRadius(),
		model->getNeighbors(),
		model->getGridX(),
		model->getGridY(),
		model->getThreshold());
	cout << model_info << endl;
	// We could get the histograms for example:
	vector<Mat> histograms = model->getHistograms();
	// But should I really visualize it? Probably the length is interesting:
	cout << "Size of the histograms: " << histograms[0].total() << endl;
}
void Face_detectAndDisplay()
{
	VideoCapture cap(0);             //開啟攝影機
	if (!cap.isOpened()) return;   //確認攝影機打開
	//訓練好樣板
	Mat faceTemplate, img = imread("../data/pictures/man_template.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	if (!img.data)                                                    //如果數據為空
	{
		std::cout << " --(!) Error reading images " << std::endl; return;
	}
	GaussianBlur(img, img, Size(3, 3), 0, 0, BORDER_DEFAULT);
	vector<Rect> _faces;
	vector<Rect> _eyes;
	face_cascade.detectMultiScale(img, _faces, 1.1, 2, 0, Size(30, 30));
	for (int l = 0; l < _faces.size(); l++) {
		Mat _faceROI = img(_faces[l]);
		eyes_cascade.detectMultiScale(_faceROI, _eyes, 1.1, 2, 0, Size(30, 30));
		if (_eyes.size() > 0 && _eyes.size() <= 2) {
			faceTemplate = _faceROI;
			break;
		}
	}
	Ptr<face::LBPHFaceRecognizer> model = TrainingLBPHFaceRecognizer();

	Mat frame;                       //用矩陣紀錄抓取的每張frame
	while (true)
	{
		cap >> frame;
		vector<Rect>faces;
		//vector<Rect>heads;
		//GaussianBlur(frame, frame, Size(3, 3), 0, 0, BORDER_DEFAULT);


		Mat frame_gray;
		cvtColor(frame, frame_gray, CV_BGR2GRAY);
		equalizeHist(frame_gray, frame_gray);
		/*Mat laplacianImg;
		Mat abs_dst;
		Laplacian(frame_gray, laplacianImg, CV_32F, 3, 0.5, 128, BORDER_DEFAULT);
		frame_gray.convertTo(frame_gray, CV_32F);
		abs_dst = frame_gray - laplacianImg;
		laplacianImg.convertTo(laplacianImg, CV_8U);
		abs_dst.convertTo(abs_dst, CV_8U);*/
		//convertScaleAbs(laplacianImg, laplacianImg);
		//convertScaleAbs(abs_dst, abs_dst);
		face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0, Size(30, 30));
		/*imshow("Laplacian", laplacianImg);
		imshow("abs", abs_dst);
		waitKey();*/
		for (int i = 0; i < faces.size(); i++)
		{
			//Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
			//ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
			Mat faceROI = frame_gray(faces[i]);
			std::vector<Rect> eyes;
			//-- In each face, detect eyes
			eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0, Size(30, 30));
			if (eyes.size() > 0 && eyes.size() <= 2) {

				Mat ResizedImg;
				resize(faceROI, ResizedImg, Size(32, 39));
				//Recognition
				int label = model->predict(ResizedImg);

				if (label > 0) {
					Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
					ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
					cout << "Label is " << label << endl;
					imshow(window_name, frame);
					waitKey();
				}
				else {
					cout << "Can't recognition!!" << endl;
				}
/*
				//-- Step 1: Detect the keypoints using SURF Detector     //第一步，用SIFT算子檢測關鍵點
				int minHessian = 400;
				Ptr<FeatureDetector> detector = FastFeatureDetector::create(15);
				//SurfFeatureDetector detector(minHessian);
				std::vector<KeyPoint> keypoints_1, keypoints_2;

				Mat ObjectLaplacian, TemplateLaplacian;
				//laplacian
				Laplacian(faceROI, ObjectLaplacian, CV_16S, 3, 1, 0, BORDER_DEFAULT);
				Laplacian(faceTemplate, TemplateLaplacian, CV_16S, 3, 1, 0, BORDER_DEFAULT);

				//convert to CV_8U
				convertScaleAbs(ObjectLaplacian, faceROI);
				convertScaleAbs(TemplateLaplacian, faceTemplate);


				detector->detect(faceROI, keypoints_1);
				detector->detect(faceTemplate, keypoints_2);

				//-- Draw keypoints  //在圖像中畫出特徵點
				Mat img_keypoints_1; Mat img_keypoints_2;

				drawKeypoints(faceROI, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
				drawKeypoints(faceTemplate, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

				//-- Show detected (drawn) keypoints
				imshow("gray", frame_gray);
				imshow("Keypoints 1", img_keypoints_1);
				imshow("Keypoints 2", img_keypoints_2);

				//計算特徵
				Ptr<SURF> extractor = SURF::create(minHessian);

				Mat descriptors_1, descriptors_2;//存放特徵向量的舉陣

				extractor->compute(faceROI, keypoints_1, descriptors_1);//計算特徵向量
				extractor->compute(faceTemplate, keypoints_2, descriptors_2);

				//-- Step 3: Matching descriptor vectors with a brute force matcher
				BFMatcher matcher(NORM_L2);
				std::vector< DMatch > matches;
				matcher.match(descriptors_1, descriptors_2, matches);

				//-- Draw matches
				Mat img_matches;
				drawMatches(faceROI, keypoints_1, faceTemplate, keypoints_2, matches, img_matches);

				//-- Step 3: Matching descriptor vectors using FLANN matcher
				cv::FlannBasedMatcher matcher2;
				std::vector< DMatch > matches2;
				matcher2.match(descriptors_1, descriptors_2, matches2);

				double max_dist = 0; double min_dist = 100;

				//-- Quick calculation of max and min distances between keypoints
				for (int i = 0; i < descriptors_1.rows; i++)
				{
					double dist = matches[i].distance;
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
					if (matches[i].distance < 2 * min_dist)
					{
						good_matches.push_back(matches[i]);
					}
				}
				// 找到特定人臉
				if (good_matches.size() > descriptors_2.rows / 3) {
					//-- Draw only "good" matches
					Mat img_matches2;
					drawMatches(faceROI, keypoints_1, faceTemplate, keypoints_2,
						good_matches, img_matches2, Scalar::all(-1), Scalar::all(-1),
						vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

					//
					Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
					ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
					//-- Show detected matches  
					imshow("Good Matches", img_matches2);
					imshow(window_name, frame);
					waitKey(0);

				}
*/
			}
			//for (int j = 0; j < eyes.size(); j++)
			//{
			//	Point center(faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5);
			//	int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
			//	circle(frame, center, radius, Scalar(255, 0, 0), 4, 8, 0);
			//}
		}
		imshow(window_name, frame);                //建立一個視窗,顯示frame到camera名稱的視窗
		if (waitKey(30) == 27) break;  //按鍵就離開程式
	}
}

void Plate_dectecAndDisplay()
{
	//可以參考這裡 http://www.licenseplatesrecognition.com/how-lpr-works.html
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

	//輸入一張當背景
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
		cvtColor(frame, gray, CV_BGR2GRAY);

		//前景
		Mat gray32f;
		gray.convertTo(gray32f, CV_32F);

		absdiff(gray32f, Background32f, Foreground32f);
		threshold(Foreground32f, Foreground, 30, 255, THRESH_BINARY);
		accumulateWeighted(gray32f, Background32f, 0.09);
		float flmean = mean(Foreground)[0];
		if (flmean < 10) {
			equalizeHist(gray, gray);
			vector<Rect>plates;
			plate_cascade.detectMultiScale(gray, plates, 1.1, 2, 0, Size(30, 30));
			for (int i = 0; i < plates.size(); i++)
			{
				rectangle(frame, plates[i], Scalar(0, 255, 0), 2);
			}
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
				AnalyseImage(kclassifier, plate);
				//AnalyseImage_SCW(kclassifier, plate);
			}
		}

		imshow(window_name, frame);                //建立一個視窗,顯示frame到camera名稱的視窗
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
	Mat img_1 = imread("../data/pictures/man_template.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat img_2 = imread("../data/pictures/man2.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	if (!img_1.data || !img_2.data)                                                    //如果數據為空
	{
		std::cout << " --(!) Error reading images " << std::endl; return;
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


	waitKey(0);
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

	Rect rec = contours.size() == 0 ? Rect(0, 0, thresholdImage.cols, thresholdImage.rows) : boundingRect(contours[idx]);

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

void RunSelfTest(Ptr<KNearest> knn2)
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


void AnalyseImage(Ptr<KNearest> knearest, Mat image)
{

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
	

	imshow("LUT", blur);

	adaptiveThreshold(blur, thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 7, 0);
	/*
		// calhist
		int HistogramBins = 256;
		float HistogramRange1[2] = { 0,256 };
		const float *HistogramRange = { &HistogramRange1[0] };
		Rect middle = Rect(0, blur.rows * 0.333, blur.cols, blur.rows * 0.333);
		Mat MiddleImage = blur(middle);
		Mat roi(MiddleImage.size(), CV_8UC1);
		Mat Histogram1;
		calcHist(&MiddleImage, 1, 0, roi, Histogram1, 1, &HistogramBins, &HistogramRange);
		int index = -1;
		int index_second = -1;
		int index_lowestValue = -1;
		int max = 0;
		for (int i = 0; i < HistogramBins; i++) {
			if (index_lowestValue == -1 && Histogram1.at<float>(i) > 10 ) {
				index_lowestValue = i;
			}
			if (Histogram1.at<float>(i) > max)
			{
				max = Histogram1.at<float>(i);
				index_second = index;
				index = i;
			}
		}
		//show hist
		Mat HistogramImage(Size(640, 480), CV_8U);
		double HistogramBinWidth;
		normalize(Histogram1, Histogram1, 0, 200, NORM_MINMAX);

		HistogramImage = Scalar::all(255);

		HistogramBinWidth = HistogramImage.cols / HistogramBins;

		for (int i = 0; i<HistogramBins; i++)
		{
			Point Point1 = Point(i*HistogramBinWidth, 0);
			Point Point2 = Point((i + 1)*HistogramBinWidth, (int)Histogram1.at<float>(i));
			rectangle(HistogramImage, Point1, Point2, Scalar(0, 0, 0), -1);
		}
		flip(HistogramImage, HistogramImage, 0);
		//imshow("look", MiddleImage);
		//imshow("Gray Level Histogram", HistogramImage);
		//waitKey();
		int thresh_value1 = index;
		int thresh_value2 = index_second;
		//找到最好的(連通圖至少7)
		int BestValue = -1;
		Mat BestImage;
		for (int i = index_lowestValue; i < thresh_value1; i++) {
			//取分界點 + 二直化
			threshold(blur, thresh, i * 256 / HistogramBins, 255, THRESH_BINARY);

			//opening - 斷開
			Mat Element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(1, 1));
			morphologyEx(thresh, thresh, MorphTypes::MORPH_OPEN, Element, Point(1, 1), 1);

			//找連通圖
			//Mat labelImage(thresh.size(), CV_32S);
			//int nLabels = connectedComponents(thresh, labelImage, 8);

			//找分類
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
			int cClass = bigRects.size();
			//判斷品質
			if (cClass >  BestValue && cClass < 10) {

				BestValue = cClass;
				BestImage = thresh.clone();

				if (BestValue == 7) break;
			}
			imshow("look", thresh);
			waitKey();

		}
		imshow("look", BestImage);
		waitKey();
	*/

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

		imshow("single", stagedImage);
		waitKey(0);
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
	cout << "\n";
	imshow("all", resized);
	waitKey(0);
}

void AnalyseImage_SCW(Ptr<KNearest> knearest, Mat image)
{

	Mat sample2(Size(ImageSize, 1), CV_32FC1);

	Mat resized, gray, blur, thresh, thresh2;

	vector < vector<Point> > contours;
	//std::string tPath = pathToImages;
	//tPath.append("/buchstaben.png");
	//image = imread(tPath, 1);
	//轉成 75 X 228
	resize(image, resized, Size(228, 75), 0, 0, InterpolationFlags::INTER_CUBIC);
	//轉灰階
	cv::cvtColor(resized, gray, COLOR_BGR2GRAY);
	//模糊化去雜訊
	GaussianBlur(gray, gray, Size(5, 5), 2, 2);
	//直方圖等化
	equalizeHist(gray, gray);
	//區域二值化
	adaptiveThreshold(gray, gray, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 0);

	imshow("original", gray);
	//Sliding centric window - measurement: Standard deviation
	for (double _threshold = 0; _threshold < 10; _threshold += 0.5)
	{
		thresh = Mat::zeros(gray.size(), CV_8U);
		int X1 = 1, Y1 = 4, X2 = 2, Y2 = 8;
		double T = _threshold;
		for (int i = 0; i < gray.rows; i++) {
			for (int j = 0; j < gray.cols; j++) {
				double mean_A = 0, mean_B = 0;
				double standard_deviation_A = 0, standard_deviation_B = 0;
				int count = 0;
				//WA平均值
				for (int ai = i - Y1; ai < i + Y1; ai++) {
					for (int aj = j - X1; aj < j + X1; aj++) {
						if (ai < 0 || ai >= gray.rows || aj < 0 || aj >= gray.cols) continue;
						count++;
						mean_A += gray.at<unsigned char>(ai, aj);
					}
				}
				mean_A /= count;
				count = 0;
				//WA標準差
				for (int ai = i - Y1; ai < i + Y1; ai++) {
					for (int aj = j - X1; aj < j + X1; aj++) {
						if (ai < 0 || ai >= gray.rows || aj < 0 || aj >= gray.cols) continue;
						count++;
						standard_deviation_A += pow(gray.at<unsigned char>(ai, aj) - mean_A, 2);
					}
				}
				standard_deviation_A /= count;
				count = 0;
				standard_deviation_A = sqrt(standard_deviation_A);
				//WB平均值
				for (int bi = i - Y2; bi < i + Y2; bi++) {
					for (int bj = j - X2; bj < j + X2; bj++) {
						if (bi < 0 || bi >= gray.rows || bj < 0 || bj >= gray.cols) continue;
						count++;
						mean_B += gray.at<unsigned char>(bi, bj);
					}
				}
				mean_B /= count;
				count = 0;
				//WB標準差
				for (int bi = i - Y2; bi < i + Y2; bi++) {
					for (int bj = j - X2; bj < j + X2; bj++) {
						if (bi < 0 || bi >= gray.rows || bj < 0 || bj >= gray.cols) continue;
						count++;
						standard_deviation_B += pow(gray.at<unsigned char>(bi, bj) - mean_B, 2);
					}
				}
				standard_deviation_B /= count;
				count = 0;
				standard_deviation_B = sqrt(standard_deviation_B);
				//measureB / measureA > T
				if (standard_deviation_B / standard_deviation_A > T) {
					thresh.at<unsigned char>(i, j) = 255;
				}
			}
		}
		//rectangle(gray, Rect(0, 0, X2, Y2), Scalar(255, 0, 0), 2);
		//inverse
		//thresh = ~thresh;		
		imshow("thresh", thresh);
		waitKey();
	}

	//切割文字
	findContours(thresh, contours, CV_RETR_LIST, CHAIN_APPROX_SIMPLE);

	vector < Rect > Rects;
	for (size_t i = 0; i < contours.size(); i++) {
		vector < Point > cnt = contours[i];
		if (cnt.size() < 5) continue;
		Rect rec = boundingRect(cnt);
		double angle = fitEllipse(cnt).angle;
		if (rec.height > 32 && angle > 75)
		{
			Rects.push_back(rec);
		}
	}
	//x排序
	for (size_t i = 0; i < Rects.size(); i++) {
		double minVal = Rects[i].x;
		size_t swapIndex = i;
		for (size_t j = i + 1; j < Rects.size(); j++)
		{
			if (Rects[j].x < minVal) {
				minVal = Rects[j].x;
				swapIndex = j;
			}
		}
		if (swapIndex >= 0 && swapIndex != i) {
			//swap
			Rect temp = Rects[i];
			Rects[i] = Rects[swapIndex];
			Rects[swapIndex] = temp;
		}
	}

	bool bWordFirst = true;
	for (size_t i = 0; i < Rects.size(); i++)
	{
		Rect rec = Rects[i];
		Mat roi = /*image*/thresh(rec);
		Mat stagedImage;
		//PreProcessImage(&temp, &stagedImage, sizex, sizey);
		resize(roi, stagedImage, Size(sizex, sizey), 0, 0, INTER_CUBIC);
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

		imshow("single", stagedImage);
		waitKey(0);
	}
	cout << "\n";
	imshow("all", image);
	waitKey(0);
}


Ptr<FaceRecognizer> TrainingFaceRecognizer(FACE_RECOGNITION_TYPE Type)
{
	Ptr<FaceRecognizer> model;
	String fn_csv = "../data/at.txt";
	// These vectors hold the images and corresponding labels.
	vector<Mat> images;
	vector<int> labels;
	// Read in the data. This can fail if no valid
	// input filename is given.
	try {
		read_csv(fn_csv, images, labels);
	}
	catch (cv::Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		// nothing more we can do
		exit(1);
	}
	// Quit if there are not enough images for this demo.
	if (images.size() <= 1) {
		string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
		CV_Error(CV_StsError, error_message);
	}

	// Get the height from the first image. We'll need this
	// later in code to reshape the images to their original
	// size:
	int height = images[0].rows;
	// The following lines simply get the last images from
	// your dataset and remove it from the vector. This is
	// done, so that the training data (which we learn the
	// cv::FaceRecognizer on) and the test data we test
	// the model with, do not overlap.
	switch ( Type )
	{
	case EIGEN:		
		// The following lines create an Fisherfaces model for
		// face recognition and train it with the images and
		// labels read from the given CSV file.
		// If you just want to keep 10 Fisherfaces, then call
		// the factory method like this:
		//
		//      cv::createFisherFaceRecognizer(10);
		//
		// However it is not useful to discard Fisherfaces! Please
		// always try to use _all_ available Fisherfaces for
		// classification.
		//
		// If you want to create a FaceRecognizer with a
		// confidence threshold (e.g. 123.0) and use _all_
		// Fisherfaces, then call it with:
		//
		//      cv::createFisherFaceRecognizer(0, 123.0);
		//
		model = face::EigenFaceRecognizer::create();
		model->train(images, labels);
		break;
	case FISHER:
		model = face::FisherFaceRecognizer::create();
		model->train(images, labels);
		break;
	case LBPH:
		model = face::LBPHFaceRecognizer::create();
		model->train(images, labels);
		break;
	default:
		break;
	}
	return model;
}

Ptr<EigenFaceRecognizer> TrainingEigenFaceRecognizer()
{
	Ptr<EigenFaceRecognizer> model;
	String fn_csv = "../data/at.txt";
	// These vectors hold the images and corresponding labels.
	vector<Mat> images;
	vector<int> labels;
	// Read in the data. This can fail if no valid
	// input filename is given.
	try {
		read_csv(fn_csv, images, labels);
	}
	catch (cv::Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		// nothing more we can do
		exit(1);
	}
	// Quit if there are not enough images for this demo.
	if (images.size() <= 1) {
		string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
		CV_Error(CV_StsError, error_message);
	}

	// Get the height from the first image. We'll need this
	// later in code to reshape the images to their original
	// size:
	int height = images[0].rows;
	// The following lines simply get the last images from
	// your dataset and remove it from the vector. This is
	// done, so that the training data (which we learn the
	// cv::FaceRecognizer on) and the test data we test
	// the model with, do not overlap.
	model = face::EigenFaceRecognizer::create();
	model->train(images, labels);
	return model;
}
Ptr<FisherFaceRecognizer> TrainingFisherFaceRecognizer()
{
	Ptr<FisherFaceRecognizer> model;
	String fn_csv = "../data/at.txt";
	// These vectors hold the images and corresponding labels.
	vector<Mat> images;
	vector<int> labels;
	// Read in the data. This can fail if no valid
	// input filename is given.
	try {
		read_csv(fn_csv, images, labels);
	}
	catch (cv::Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		// nothing more we can do
		exit(1);
	}
	// Quit if there are not enough images for this demo.
	if (images.size() <= 1) {
		string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
		CV_Error(CV_StsError, error_message);
	}

	// Get the height from the first image. We'll need this
	// later in code to reshape the images to their original
	// size:
	int height = images[0].rows;
	// The following lines simply get the last images from
	// your dataset and remove it from the vector. This is
	// done, so that the training data (which we learn the
	// cv::FaceRecognizer on) and the test data we test
	// the model with, do not overlap.
	model = face::FisherFaceRecognizer::create();
	model->train(images, labels);
	return model;
}
Ptr<LBPHFaceRecognizer> TrainingLBPHFaceRecognizer()
{
	Ptr<LBPHFaceRecognizer> model;
	String fn_csv = "../data/at.txt";
	// These vectors hold the images and corresponding labels.
	vector<Mat> images;
	vector<int> labels;
	// Read in the data. This can fail if no valid
	// input filename is given.
	try {
		read_csv(fn_csv, images, labels);
	}
	catch (cv::Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		// nothing more we can do
		exit(1);
	}
	// Quit if there are not enough images for this demo.
	if (images.size() <= 1) {
		string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
		CV_Error(CV_StsError, error_message);
	}

	// Get the height from the first image. We'll need this
	// later in code to reshape the images to their original
	// size:
	int height = images[0].rows;
	// The following lines simply get the last images from
	// your dataset and remove it from the vector. This is
	// done, so that the training data (which we learn the
	// cv::FaceRecognizer on) and the test data we test
	// the model with, do not overlap.
	model = face::LBPHFaceRecognizer::create();
	model->train(images, labels);
	return model;
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