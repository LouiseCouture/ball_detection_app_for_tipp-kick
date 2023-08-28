#include "particle.h"


ParticleFilter::ParticleFilter(int N_part, int height, int width, int vel,int stddev) {
	N = N_part;
	max_height = height;
	max_width = width;
	velocity = vel;

	max_part_err = 1.0;
	somme_weight = 0.0;
	max_part_weight = 0.0;

	sum_x = 0;
	sum_y = 0;

	norm_distribution =  std::normal_distribution<float>(0, stddev);

	OneParticle* part;
	for (int i = 0; i < N; i++) {
		part = new OneParticle(Point(rand() % max_width, rand() % max_height), -velocity / 2, -velocity / 2);

		sum_x += part->coord.x;
		sum_y += part->coord.y;

		particles.push_back(part);
	}

	mean_location = Point(sum_x /N, sum_y /N);
}


void ParticleFilter::update(Point* target) {

	max_part_err = 1.0;
	somme_weight = 0.0;
	max_part_weight = 0.0;

	sum_x = 0;
	sum_y = 0;

	OneParticle* part ;

	for (int i = 0; i < N; i++) {

		part = particles[i];

		applyVel(part);
		enforceEdge(part);
		computeError(part, target);
	}

	for (int i = 0; i < N; i++) {

		part = particles[i];

		sum_x += part->coord.x;
		sum_y += part->coord.y;

		computeWeight(part);
		applyNoise(part);
	}

	resample();

	mean_location.x = sum_x / N;
	mean_location.y = sum_y / N;
}

Mat ParticleFilter::plot(Mat frame, Point* target) {

	for (int i = 0; i < N; i++) {
		circle(frame, particles[i]->coord,2, Scalar(255,25,25), 2);
	}

	circle(frame, mean_location, 3, Scalar(50, 50, 255), 3);
	
	if (target != NULL) {
		circle(frame, Point(target->x,target->y), 3, Scalar(255, 255, 255), 3);
	}

	return frame;
}


void ParticleFilter::applyVel(OneParticle* part) {
	part->coord.x += part->vel_x;
	part->coord.y += part->vel_y;
}


void ParticleFilter::applyNoise(OneParticle* part) {

	std::random_device rd{};
	std::mt19937 gen{ rd() };

	float coeff = 0.2;

	part->coord.x += int(norm_distribution(gen)* coeff);
	part->coord.y += int(norm_distribution(gen) * coeff);
	part->vel_x += int(norm_distribution(gen) * coeff);
	part->vel_y += int(norm_distribution(gen) * coeff);


}


void ParticleFilter::enforceEdge(OneParticle* part) {


	if (part->coord.x > max_width) {
		part->coord.x = max_width;
		part->vel_x = -abs(velocity);
	}
	if (part->coord.y > max_height) {
		part->coord.y = max_height;
		part->vel_y = -abs(velocity);
	}
	if (part->coord.x < 0) {
		part->coord.x = 0;
		part->vel_x = abs(velocity);
	}
	if (part->coord.y < 0) {
		part->coord.y = 0;
		part->vel_y = abs(velocity);
	}


}


void  ParticleFilter::computeError(OneParticle* part, Point* target) {
	if (target == NULL) {
		part->error = part->max_err;
		max_part_err = part->max_err;
	}
	else {
		part->error =  sqrt(  pow( (part->coord.x - target->x),2) + pow((part->coord.y - target->y), 2)  )  ;

		if (part->error > max_part_err) {
			max_part_err = part->error;
		}
	}
}


void ParticleFilter::computeWeight(OneParticle* part) {

	part->weight = max( max_part_err - part->error,0.0);
	if (part->coord.x == 0 || part->coord.y == 0 || part->coord.x == max_width || part->coord.y == max_height) {
		part->weight = 0;
	}
	part->weight = pow(part->weight, 4)/ max_part_err;


	somme_weight += part->weight;

	if (part->weight > max_part_weight) {
		max_part_weight= part->weight;
	}

}


void ParticleFilter::resample(void) {
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine gen_resample(seed);
	std::uniform_real_distribution<> ranN(0, 1);
	double beta = 0.0;

	int index = int(ranN(gen_resample) * N)%N;

	vector<OneParticle*> particles_resample;

	for (int i = 0; i < N; i++){

		double coeff = ranN(gen_resample);
		beta += coeff * 2.0 * double(max_part_weight);


		while ( beta > particles[index]->weight) {

			beta -= particles[index]->weight;
			index = (index + 1) % N;
		}
		particles_resample.push_back(new OneParticle(particles[index]));
	}

	// -------------------------delet old particle------------------------
	OneParticle* part;
	for (int i = 0; i < N; i++) {
		part = particles[i];
		delete part;
	}

	particles = particles_resample;

}

