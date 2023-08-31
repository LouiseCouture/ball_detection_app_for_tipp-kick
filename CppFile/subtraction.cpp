#include "subtraction.h"

Mat subtraction(Mat frame_0_YUV, Mat frame_1_YUV, cv::Size frame_size, Mat frame_HSV_0, Mat frame_HSV_1) {

	int blur = 15;
	int threshold = 250;
	int threshold_type = 0;
	
	cv::Mat diff0 = cv::Mat::zeros(frame_size, CV_16UC1);
	cv::Mat diff1 = cv::Mat::zeros(frame_size, CV_16UC1);

	vector<Mat> channels_1(frame_0_YUV.channels());
	vector<Mat> channels_0(frame_0_YUV.channels());

	split(frame_1_YUV, channels_1);
	split(frame_0_YUV, channels_0);

	Mat blur_frame_1 = cv::Mat(frame_size, CV_16UC1);
	Mat blur_frame_0 = cv::Mat(frame_size, CV_16UC1);

	Mat diff_1_channel = cv::Mat(frame_size, CV_8UC1);
	Mat diff_0_channel = cv::Mat(frame_size, CV_8UC1);

	Mat diff_0_channel_convert(frame_size, diff0.type());
	Mat diff_1_channel_convert(frame_size, diff0.type());
	
	for (int i = 0; i < frame_0_YUV.channels(); i++) {
		
		cv::GaussianBlur(channels_1[i], blur_frame_1, Size(blur, blur), 0);
		cv::GaussianBlur(channels_0[i], blur_frame_0, Size(blur, blur), 0);
		
		// The subtraction
		cv::absdiff(blur_frame_1, blur_frame_0, diff_1_channel);
		cv::absdiff(blur_frame_0, blur_frame_1, diff_0_channel);

		// convert to 16 because we can be > 255
		diff_0_channel.convertTo(diff_0_channel_convert, diff0.type());
		diff_1_channel.convertTo(diff_1_channel_convert, diff0.type());

		//cout << type2str(diff0.type()) << " " << type2str(diff_0_channel_convert.type())<<endl;

		diff0 = diff0 + diff_0_channel_convert;
		diff1 = diff1 + diff_0_channel_convert;
		
	}
	
	//------------------------------------------------------------ Normalization -----------------------------------------------------------------
	double min_mat, max_0, max_1;

	cv::minMaxLoc(diff0, &min_mat, &max_0);
	cv::minMaxLoc(diff1, &min_mat, &max_1);

	Mat diff_1_convert = cv::Mat(frame_size, CV_8UC1);
	Mat diff_0_convert = cv::Mat(frame_size, CV_8UC1);

	diff1.convertTo(diff_1_convert, CV_8UC1, 255 / max_1);
	diff0.convertTo(diff_0_convert, CV_8UC1, 255 / max_0);


	//------------------------------------------------------------ Threshold -----------------------------------------------------------------

	Mat threshed_0;
	Mat threshed_1;
	
	cv::threshold(diff_0_convert, threshed_0, threshold, 255, threshold_type);
	cv::threshold(diff_1_convert, threshed_1, threshold, 255, threshold_type);
	
	//------------------------------------------------------------ Add the 2 sub -----------------------------------------------------------------

	Mat diff;
	cv::add(diff_0_convert, diff_1_convert, diff);
	cv::threshold(diff, diff, 100, 255, threshold_type);

	//------------------------------------------------------------ Remove hand -----------------------------------------------------------------
	
	Mat mask_negatif_0;
	Mat mask_negatif_1;
	Mat mask_positif_0 = selectSkinColor(frame_HSV_0);
	Mat mask_positif_1 = selectSkinColor(frame_HSV_1);
	
	cv::bitwise_not(mask_positif_0, mask_negatif_0);
	cv::bitwise_not(mask_positif_1, mask_negatif_1);

	cv::bitwise_and(diff, mask_negatif_0, diff);
	cv::bitwise_and(diff, mask_negatif_1, diff);


	//------------------------------------------------------------ Erode and dilate -----------------------------------------------------------------
	/*
	int erosion_size = 2;
	Mat element_er = cv::getStructuringElement(MORPH_ELLIPSE,Size(2 * erosion_size + 1, 2 * erosion_size + 1),Point(erosion_size, erosion_size));
	cv::erode(diff, diff, element_er);
	
	erosion_size = 3;
	Mat element_dil = cv::getStructuringElement(MORPH_ELLIPSE, Size(2 * erosion_size + 1, 2 * erosion_size + 1), Point(erosion_size, erosion_size));
	cv::dilate(diff, diff, element_dil);
	*/

	return diff;

}

Mat selectSkinColor(Mat frame_HSV) { // this has been tested on white people only sorry
	Mat frame_blur;

	Mat mask1;
	Mat mask2;
	Mat mask;

	cv::GaussianBlur(frame_HSV, frame_blur, Size(11, 11), 0);
	inRange(frame_HSV, Scalar(0, 20, 100), Scalar(50, 255, 255), mask1);
	inRange(frame_HSV, Scalar(100, 20, 100), Scalar(255, 255, 255), mask2);

	cv::bitwise_or(mask1, mask2, mask );

	int erosion_size = 3;
	Mat element_dil = cv::getStructuringElement(MORPH_ELLIPSE, Size(2 * erosion_size + 1, 2 * erosion_size + 1), Point(erosion_size, erosion_size));
	cv::dilate(mask, mask, element_dil);

	return mask;

}

Rect detectMovingBall(Mat frame_mask, Mat frame_HSV, int max_size,int size_power2, vector<vector<vector<double>>> template_wave) {
	double best_diff=10000.0;
	Rect best_obj(-1,-1,-1,-1);

	dilate(frame_mask, frame_mask, Mat(), Point(-1, -1),1);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(frame_mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

	Rect ball;

	for (int i = 0; i< contours.size();i++) {

		Rect rect = boundingRect(contours[i]);

		if (rect.height <= max_size && rect.width <= max_size && rect.height >= size_power2 && rect.width >= size_power2) {

			Mat object_HSV = frame_HSV(rect);
			vector<vector<vector<double>>> rect_WavTrans = waveletTransform(object_HSV, size_power2);
			double diff = diffWaveTrans(template_wave, rect_WavTrans);

			if(diff<best_diff){
				best_diff = diff;
				best_obj = rect;
			}
		}
	}

	return best_obj;

}

vector<Rect> detectMovingObjects(Mat frame_mask, int max_size) {

	dilate(frame_mask, frame_mask, Mat(), Point(-1, -1), 1);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(frame_mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

	vector<Rect>objects;

	for (int i = 0; i < contours.size(); i++) {

		Rect rect = boundingRect(contours[i]);

		if (rect.height <= max_size && rect.width <= max_size && rect.height >= max_size/4 && rect.width >= max_size / 4) {

			objects.push_back(rect);
		}
	}

	return objects ;

}



string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
};