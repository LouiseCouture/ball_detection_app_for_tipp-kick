#ifndef VIDEO_H
#define VIDEO_H
#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "objectdetection.h"
#include "subtraction.h"
#include "wavelet.h"
#include "particle.h"

using namespace std;
using namespace cv;

class Video {
private:
    int coeff_resize; // fixed height so the frame is always the same size
    cv::Size frame_size;

    Mat frame_resized;
    // YUV is use for subtraction
    Mat frame_0_YUV;
    Mat frame_1_YUV;
    Mat frame_subtraction;
    // HSV is use for color selection
    Mat frame_HSV_0;
    Mat frame_HSV_1;

    int time;
    int timeJump;
    int size_box;
    int size_power2;

    int variance;
    int var_thrsh;

    Point* ball_past_ObjDetec_pts;
    Point* ball_ObjDetec_pts;
    double dist_thrsh;
    int check_samePlace;  // number of time the object detection detected the ball at the same place
    int thresh_samePlace; // if check_samePlace > thresh_samePlace => veified static

    Point target_PartFilt;

    Rect moving_ball_rect;
    Rect ball_past_ObjDetec_rect;

    vector<vector<vector<double>>> template_wave;
    double wave_thrsh;

    ParticleFilter* part_filter;

    string pathModel;


public:

    Video(void) {
        coeff_resize = 500;

        timeJump = 1;
        time = timeJump;

        size_box = 32;
        size_power2 = 32;   // need a square with the size= a power of 2 for wavelet transformation

        variance = 0;
        var_thrsh = 30;

        check_samePlace = 0;
        thresh_samePlace = 3;

        wave_thrsh = 50.0;
        dist_thrsh = 100.0; // high because I don't sqrt when I compute the distance between 2 pts: sqrt(100)=10

        moving_ball_rect = Rect(-1, -1, -1, -1);

        part_filter = NULL;
        target_PartFilt = Point(-1,-1);

        pathModel = "C:\\Users\\lc100\Documents\\GitHub\\ball_detection_app_for_tipp-kick\\03_Ball_Detection\\models/ball_weights_V2.pt";
    }

    /**
    *
    * @brief Video method, capture video and algorithms to detect the ball
    * 
    * @return int
    */
    int capture(void);

    /**
    *
    * @brief Video method, new frame:change frame 1 (most recent) and frame 0 ( more ancient )
    *
    */
    void changeFrame10(void);

    /**
    *
    * @brief Video method, compute frame subtraction and variance
    *
    * @return int
    */
    int frameSubtraction(void);


};

#endif
