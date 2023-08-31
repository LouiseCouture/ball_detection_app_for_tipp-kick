// app.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;

#include "video.h"

int main(int argc, char** argv) {
	Video film = Video();
	film.capture();
}


// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file

/*
* 
C/C++ -> General -> Additionals Include Directories ==> C:\opencv\build\include
Linker -> General -> Additionals Library Directories ==> C:\opencv\build\x64\vc16\lib
Configuration Properties -> VC++ Directories -> Library Directories ==> C:\opencv\build\x64\vc16\lib
Linker -> Input -> Additional Dependencies ==> opencv_world470d.lib
*/
