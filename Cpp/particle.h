#ifndef PARTICLE_H
#define PARTICLE_H
#pragma once


#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <random>

using namespace std;
using namespace cv;


class OneParticle {
private:
	Point coord;
	int vel_x;
	int vel_y;
	double error;
	double weight;
	double max_err;

public:
	friend class ParticleFilter;
	inline OneParticle(Point C, int Vx, int Vy) {
		coord = C;
		vel_x = Vx;
		vel_y = Vy;
		max_err = 100000.0;
		error = max_err;
		weight = max_err;
	}

	inline OneParticle(OneParticle* part) {
		coord = part->coord;
		vel_x = part->vel_x;
		vel_y = part->vel_y;
		max_err = 100000.0;
		error = max_err;
		weight = max_err;
	}

	inline Point get_Pts(void) { return coord; }
};



class ParticleFilter {
private:

	vector<OneParticle*> particles;
	Point mean_location;

	int N;
	int max_height;
	int max_width;
	int velocity;
	double max_part_err;
	double max_part_weight;
	double somme_weight;

	int sum_x;
	int sum_y;

	std::normal_distribution<float> norm_distribution;

public:
	ParticleFilter(int N_part, int height, int width, int vel, int stddev);

	/**
	*
	* @brief ParticleFilter method, plot particles and target on a frame
	*
    * @param frame
    * @param target
	* 
	* @return Mat
	*/
	Mat plot(Mat frame, Point* target);


	/**
	*
	* @brief ParticleFilter method, update particles cooordinates
	*
	* @param target
	*
	*/
	void update(Point* target);


	/**
	*
	* @brief ParticleFilter method, apply velocity to a particle
	*
	* @param part
	*
	*/
	void applyVel(OneParticle* part);
	/**
	*
	* @brief ParticleFilter method, apply noise to a particle
	*
	* @param part
	*
	*/
	void applyNoise(OneParticle* part);
	/**
	*
	* @brief ParticleFilter method, make sure the particle is not outside the frame
	*
	* @param part
	*
	*/
	void enforceEdge(OneParticle* part);
	/**
	*
	* @brief ParticleFilter method, compute how far the particle is of the target
	*
	* @param part
	* @param target
	*
	*/
	void computeError(OneParticle* part,Point* target);
	/**
	*
	* @brief ParticleFilter method, compute the weight of a particle based on the max error
	* @param target
	*
	*/
	void computeWeight(OneParticle* part);
	/**
	*
	* @brief ParticleFilter method, create new sample of particles from the previous ParticleFilter with repetion and depending of the weights of the particles
	*
	*/
	void resample(void);
};

#endif