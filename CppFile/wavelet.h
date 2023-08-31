#ifndef WAVELET_H
#define WAVELET_H
#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <math.h> 
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


/**
*
* @brief Take 2 wave transformation and compute difference of their 2*2*3 right top square
*
* @param template_wave
* @param object
*
* @return double
*
*/
double diffWaveTrans(vector<vector<vector<double>>> template_wave, vector<vector<vector<double>>> object);

/**
*
* @brief Wavelet transformation of a 3D matrix 
*
* @param object_HSV
* @param size_power2
*
* @return vector<vector<vector<double>>>
*
*/
vector<vector<vector<double>>> waveletTransform(Mat object_HSV, const int size_power2);

/**
*
* @brief Wavelet transformation of a 2D matrix
*
* @param Wave_transf
* @param Mat_chan
* @param size
* @param chan
*
*/
void transf_2D(vector<vector<vector<double>>>* Wave_transf, Mat Mat_chan, int size, int chan);

/**
*
* @brief Initialize the Wavelet transformation
*
* @param Wave_transf
* @param Mat_chan
* @param size
* @param chan
* @param Matrow_chan
*
*/
void init_vecMat(vector<vector<vector<double>>>* Wave_transf, Mat Mat_chan, int size, int chan, int row);

/**
*
* @brief  Wavelet transformation of a row
*
* @param Wave_transf
* @param row
* @param size
* @param chan
*
*/
void transf_row(vector<vector<vector<double>>>* Wave_transf, int row, int size, int chan);


/**
*
* @brief  Wavelet transformation of a column
*
* @param Wave_transf
* @param column
* @param size
* @param chan
*
*/
void transf_column(vector<vector<vector<double>>>* Wave_transf, int column, int size, int chan);


#endif