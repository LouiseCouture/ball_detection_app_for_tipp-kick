
#include "wavelet.h"


double diffWaveTrans(vector<vector<vector<double>>> template_wave, vector<vector<vector<double>>> object) {

	double diff = 0.0;

	for (int chan = 0; chan < template_wave[0][0].size(); chan++) {
		for (int column = 0; column < 2; column++) {
			for (int row = 0; row < 2; row++) {

				diff += abs(template_wave[column][row][chan] - object[column][row][chan]);

			}
		}
	}
	return diff;
}

vector<vector<vector<double>>> waveletTransform(Mat object_HSV,const int size_power2) {

	cv::Mat object = cv::Mat::zeros(object_HSV.size(), CV_32SC3);
	object_HSV.convertTo(object, object.type());

	vector<vector<vector<double>>> Wave_transf (size_power2, vector<vector<double> >(size_power2, vector <double>(object.channels())));

	// wavelet on the 3 channel
	vector<Mat> channels(object.channels());
	split(object, channels);

	for (int i = 0; i < object.channels(); i++) {
		transf_2D(&Wave_transf, channels[i], size_power2, i);

	}

	return Wave_transf;
};

void transf_2D(vector<vector<vector<double>>>* Wave_transf,Mat Mat_chan, int size,int chan) {
	
	for (int row = 0; row < size; row++) {

		init_vecMat(Wave_transf, Mat_chan, size, chan, row);
		transf_row(Wave_transf, row, size, chan);

	}

	for (int column = 0; column < size; column++) {

		transf_row(Wave_transf, column, size, chan);

	}
}

void init_vecMat(vector<vector<vector<double>>>* Wave_transf, Mat Mat_chan, int size, int chan, int row) {

	for (int column=0 ; column < size; column++) {
		double val= double(Mat_chan.at<int>(column, row)) / sqrt(2);
		Wave_transf->at(column).at(row).at(chan) = val;
	}
}

void transf_row(vector<vector<vector<double>>>* Wave_transf,int row, int size, int chan) {

	while (size > 1) {
		size = size / 2;
		for (int i = 0; i < size; i++) {

			double A = Wave_transf->operator[](2 * i)[row][chan];
			double B = Wave_transf->operator[](2 * i+1)[row][chan];

			Wave_transf->at(i).at(row).at(chan) = (A + B) / sqrt(2);
			Wave_transf->at(size + i).at(row).at(chan) = (A - B) / (2);
				

		}
	}
}

void transf_column(vector<vector<vector<double>>>* Wave_transf, int column, int size, int chan) {

	for (int row = 0; row < size; row++) {
		Wave_transf->at(column).at(row).at(chan) = Wave_transf->operator[](column)[row][chan] / sqrt(2);
	}

	while (size > 1) {
		size = size / 2;
		for (int i = 0; i < size; i++) {

			float A = Wave_transf->operator[](column)[2 * i][chan];
			float B = Wave_transf->operator[](column)[2 * i + 1][chan];

			Wave_transf->at(column).at(i).at(chan) = (A + B) / sqrt(2);
			Wave_transf->at(column).at(size + i).at(chan) = (A - B) / (2);


		}
	}

}
