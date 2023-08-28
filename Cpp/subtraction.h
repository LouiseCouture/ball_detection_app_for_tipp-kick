#ifndef SUBTRACTION_H
#define SUBTRACTION_H
#pragma once


#include <vector>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "wavelet.h"

using namespace std;
using namespace cv;
 
/**
* 
* @brief Frame subtraction and skin color removal
*
* @param frame_0_YUV
* @param frame_1_YUV
* @param frame_size
* @param frame_HSV_0
* @param frame_HSV_1
* 
* @return Mat
* 
*/
Mat subtraction(Mat frame_0_YUV, Mat frame_1_YUV, cv::Size frame_size, Mat frame_HSV_0, Mat frame_HSV_1);

/**
*
* @brief Convert type Mat to string for print
*
* @param type
*
* @return string
*
*/
string type2str(int type);

/**
*
* @brief Creat mask for the skin color
*
* @param frame_HSV
*
* @return Mat
*
*/
Mat selectSkinColor(Mat frame_HSV);

/**
*
* @brief Take frame subtraction, detect all moving object and return best match with template, if no detection return (-1,-1,-1,-1)
*
* @param frame_mask
* @param frame_HSV
* @param max_size
* @param size_power2
* @param template_wave
*
* @return Rect
*
*/
Rect detectMovingBall(Mat frame_mask, Mat frame_HSV, int max_size, int size_power2, vector<vector<vector<double>>> template_wave);


#endif
