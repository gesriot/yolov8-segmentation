#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>
#include "yolov8_seg.h"

using namespace std;
using namespace cv;
using namespace dnn;

template<typename _Tp>
int yolov8(_Tp& cls,Mat& img,string& model_path)
{
	Net net;
	if (cls.ReadModel(net, model_path, false)) {
		cout << "read net ok!" << endl;
	}
	else {
		return -1;
	}
	// generate random colors
	vector<Scalar> color;
	srand(time(0));
	for (int i = 0; i < 80; i++) {
		int b = rand() % 256;
		int g = rand() % 256;
		int r = rand() % 256;
		color.push_back(Scalar(b, g, r));
	}
	vector<OutputSeg> result;


	if (cls.Detect(img, net, result)) {
		DrawPred(img, result, cls._className, color);
	}
	else {
		cout << "Detect Failed!" << endl;
	}
	system("pause");
	return 0;
}


int main() {

	string img_path = "./images/01.jpg";
	string seg_model_path = "./models/best-seg.onnx";

	Mat img = imread(img_path);	

	Yolov8Seg task_segment;

	yolov8(task_segment, img,seg_model_path);

}