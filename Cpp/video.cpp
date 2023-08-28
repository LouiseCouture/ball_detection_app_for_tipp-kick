#include "video.h"

int Video::capture(void) {

    Mat direct_capture;          
    VideoCapture cap(0);   

    if (!cap.isOpened()) {        
        cout << "No video stream detected" << endl;
        system("pause");
        return -1;
    }

    cap >> direct_capture;
 
    //resizing frame so it is always the same size
    frame_size = direct_capture.size();

    int scale_percent = coeff_resize / frame_size.height;
    frame_size.height = scale_percent * frame_size.height;
    frame_size.width  = scale_percent * frame_size.width;

    frame_0_YUV = cv::Mat(frame_size.height, frame_size.width, CV_16UC3);
    frame_1_YUV = cv::Mat(frame_size.height, frame_size.width, CV_16UC3);
    frame_resized = cv::Mat(frame_size.height, frame_size.width, CV_16UC3);

    resize(direct_capture, frame_resized, Size(frame_size.width, frame_size.height), INTER_LINEAR);

    changeFrame10();

    // particle filter
    part_filter = new  ParticleFilter(100, frame_size.height, frame_size.width, 10, 50);

    //------------------------------------------------------Object detection Initial---------------------------------------------------------------------------
    // Not implemented yet
    //int res=loadModel(pathModel);

    // get a random pts for now
    ball_ObjDetec_pts=new Point (100, 100);
    ball_past_ObjDetec_pts = ball_ObjDetec_pts;
    ball_past_ObjDetec_rect = Rect(ball_ObjDetec_pts->x - size_box / 2, ball_ObjDetec_pts->y - size_box / 2, size_box, size_box);

    Mat mat_ObjDetec = frame_HSV_1( Rect(ball_ObjDetec_pts->x- size_box, ball_ObjDetec_pts->y - size_box, size_box, size_box)  ) ;
    template_wave = waveletTransform(mat_ObjDetec, size_power2);

    //------------------------------------------------------------ Loop ---------------------------------------------------------------------------
    while (true) {     
        cap >> direct_capture;
        if (direct_capture.empty()) {   
            break;
        }

        resize(direct_capture, frame_resized, Size(frame_size.width, frame_size.height), INTER_LINEAR);

        // get new frame 1 and 0 for subtraction and HSV
        changeFrame10();


        //------------------------------------------------------Object detection---------------------------------------------------------------------------
        ball_past_ObjDetec_pts = new Point(*ball_ObjDetec_pts);
        ball_ObjDetec_pts->x=-1;

        if (check_samePlace < thresh_samePlace) {
            // implement model here

            ball_ObjDetec_pts = new Point(100, 100);
        }
        if (ball_ObjDetec_pts->x > 0) {//  detect static succesful

            ball_past_ObjDetec_rect = Rect(ball_ObjDetec_pts->x- size_box/2, ball_ObjDetec_pts->y- size_box/2, size_box, size_box);
        }

        rectangle(frame_resized, Point(ball_past_ObjDetec_rect.x, ball_past_ObjDetec_rect.y ), Point(ball_past_ObjDetec_rect.x + size_box, ball_past_ObjDetec_rect.y + size_box), Scalar(255, 255, 255), 2, LINE_8);


        //------------------------------------------------------Frame subtraction---------------------------------------------------------------------------

        variance = frameSubtraction();

        if (variance < var_thrsh) {
            cout << "---------------- DETECT MOVEMENT ----------------" << endl;

            // check last static position
            Mat mat_lastStatic_HSV = frame_HSV_1(ball_past_ObjDetec_rect);
            double diff=diffWaveTrans( waveletTransform(mat_lastStatic_HSV, size_power2),template_wave);
            cout <<"wavelet difference: "<< diff << endl;

            if (diff > wave_thrsh) {
                cout << "-------- WAVELET DON'T SEE STATIC BALL --------" << endl;
                cout << "               CHECK MOVING OBJECT         " << endl;

                check_samePlace = 0;

                moving_ball_rect = detectMovingBall(frame_subtraction, frame_HSV_1, size_box * 2, size_power2, template_wave);

                // if no detection: moving_ball_rect=(-1,-1,-1,-1)
                if (moving_ball_rect.x > 0) {
                    rectangle(frame_resized, Point(moving_ball_rect.x, moving_ball_rect.y), Point(moving_ball_rect.width, moving_ball_rect.height), Scalar(100, 100, 100), 2, LINE_8);
                }
            }

        }

        //------------------------------------------------------Verify static---------------------------------------------------------------------------
        if (ball_ObjDetec_pts->x > 0 && ball_past_ObjDetec_pts->x > 0) {

            double distance = pow(ball_past_ObjDetec_pts->x - ball_ObjDetec_pts->x, 2) + pow(ball_past_ObjDetec_pts->y - ball_ObjDetec_pts->y, 2);

            if (distance <= dist_thrsh) {
                cout << "       OBJECT DETECTION FOUND THE SAME PLACE      " << endl;
                check_samePlace += 1;
            }
            else {
                cout << "    OBJECT DETECTION DID NOT FOUND THE SAME PLACE    " << endl; 
                check_samePlace = 0;
            }

            if (check_samePlace >= thresh_samePlace) {
                cout << "---------------- VERIFIED STATIC ----------------" << endl;
                cout << "             STOP USING OBJ DET             " << endl;

                //template update
                mat_ObjDetec = frame_HSV_1(ball_past_ObjDetec_rect);
                template_wave = waveletTransform(mat_ObjDetec, size_power2);

            }


        }

        //------------------------------------------------------Particle filter---------------------------------------------------------------------------
        target_PartFilt.x = -1;

        /*objet detection
        if(ball_ObjDetec_pts->x > 0){
            target_PartFilt.x = ball_ObjDetec_pts->x;
            target_PartFilt.y = ball_ObjDetec_pts->y;
        }
        
        // if verified static, take last static detection 
        if (check_samePlace >= thresh_samePlace) {

            target_PartFilt.x = ball_past_ObjDetec_rect.x + ball_past_ObjDetec_rect.width / 2;
            target_PartFilt.y = ball_past_ObjDetec_rect.y + ball_past_ObjDetec_rect.height / 2;

        }

        */
        // movement detection overide static detection
        if (moving_ball_rect.x > 0) {
            target_PartFilt.x = moving_ball_rect.x + moving_ball_rect.width / 2;
            target_PartFilt.y = moving_ball_rect.y + moving_ball_rect.height / 2;
        }

        if (target_PartFilt.x > 0) {
            part_filter->update(&target_PartFilt);
            part_filter->plot(frame_resized, &target_PartFilt);
        }

        //------------------------------------------------------Next loop---------------------------------------------------------------------------
        char c = (char)waitKey(1);        
        if (c == 27) { //If 'Esc' is entered break the loop//
            break;
        }

        imshow("Video Player", frame_resized);

        time += timeJump;
    }



    cap.release(); //Releasing the buffer memory//
    return 0;
};



void Video::changeFrame10(void) {

    if (time == timeJump) { // first loop -> initialize frame_1_YUV
        frame_resized.copyTo(frame_1_YUV);
        cvtColor(frame_1_YUV, frame_1_YUV, COLOR_BGR2YUV);
        cvtColor(frame_resized, frame_HSV_1,COLOR_BGR2HSV);
    }

    frame_1_YUV.copyTo(frame_0_YUV);
    frame_resized.copyTo(frame_1_YUV);
    cvtColor(frame_1_YUV, frame_1_YUV, COLOR_BGR2YUV);

    frame_HSV_1.copyTo(frame_HSV_0);
    cvtColor(frame_resized, frame_HSV_1, COLOR_BGR2HSV);

};

int Video::frameSubtraction(void) {
    frame_subtraction = subtraction(frame_0_YUV, frame_1_YUV, frame_size, frame_HSV_0, frame_HSV_1);

    // Variance 
    Scalar m, stdv;
    meanStdDev(frame_subtraction, m, stdv);

    //cout <<"variance frame subtraction: " << stdv[0] << endl;
    if (stdv[0] <= var_thrsh) {
        rectangle(frame_subtraction, Point(0, 0), Point(frame_size.width, frame_size.height), Scalar(255), 10, LINE_8);
    }
    imshow("frame subtraction", frame_subtraction);

    return stdv[0];

}
