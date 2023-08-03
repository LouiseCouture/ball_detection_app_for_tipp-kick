import matplotlib.pyplot as plt
import cv2
import numpy as np
import math 
from numpy.random import uniform,randn
import numpy as np
import scipy
import os

# Changing the current working directory
os.chdir(r"C:\Users\lc100\Documents\GitHub\\ball_detection_app_for_tipp-kick")


def create_uniform_particles(x_range, y_range, t_range, hdg_range, N):
    particles = np.empty((N, 4))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = uniform(t_range[0], t_range[1], size=N)
    particles[:, 3] = uniform(hdg_range[0], hdg_range[1], size=N)
    particles[:, 3] %= 2 * np.pi
    return particles

def create_gaussian_particles(mean, std, N):
    particles = np.empty((N, 3))
    particles[:, 0] = mean[0] + (randn(N) * std[0])
    particles[:, 1] = mean[1] + (randn(N) * std[1])
    particles[:, 2] = mean[2] + (randn(N) * std[2])
    particles[:, 3] = mean[3] + (randn(N) * std[3])
    particles[:, 3] %= 2 * np.pi
    return particles


def update(particles, weights, z, R, landmarks):
    for i, landmark in enumerate(landmarks):
        distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
        weights *= scipy.stats.norm(distance, R).pdf(z[i])

    weights += 1.e-300      # avoid round-off to zero
    weights /= sum(weights) # normalize
    
def predict(particles, u, std, dt=1.):
    """ move according to control input u (heading change, velocity)
    with noise Q (std heading change, std velocity)`"""

    N = len(particles)
    # update heading
    particles[:, 2] += u[0] + (randn(N) * std[0])
    particles[:, 2] %= 2 * np.pi

    # move in the (noisy) commanded direction
    dist = (u[1] * dt) + (randn(N) * std[1])
    particles[:, 0] += np.cos(particles[:, 2]) * dist
    particles[:, 1] += np.sin(particles[:, 2]) * dist
    particles[:, 2] += np.sin(particles[:, 2]) * dist
    
def estimate(particles, weights):
    """returns mean and variance of the weighted particles"""

    pos = particles[:, 0:2]
    mean = np.average(pos, weights=weights, axis=0)
    var  = np.average((pos - mean)**2, weights=weights, axis=0)
    return mean, var

def simple_resample(particles, weights):
    N = len(particles)
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1. # avoid round-off error
    indexes = np.searchsorted(cumulative_sum, random(N))

    # resample according to indexes
    particles[:] = particles[indexes]
    weights.fill(1.0 / N)

def neff(weights):
    return 1. / np.sum(np.square(weights))

###############################################################################

position=np.load("aaa_position.npy")
position=position[:100]
print(position.shape)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(111, projection='3d')
ax.grid()

ax.scatter(position[:,1],position[:,0],position[:,2],marker="+")
plt.show() 
###############################################################################
"""
N=200
range=[0,1000]

particles = create_uniform_particles(range, range,(0,50), (0, 6.28), N)
weights = np.ones(N) / N


NB_PTS = len(position)

for pts in position:
    predict(particles, u=(0.00, 1.414), std=(.2, .05))
    
    print(pts[:3])
    
    # distance from robot to each pts
    zs = (norm(pts - pts, axis=1) + (randn(NB_PTS) * sensor_std_err))
    update(particles, weights, z=zs, R=sensor_std_err, landmarks=pts[:3])

    if neff(weights) < N/2:
        indexes = systematic_resample(weights)
        resample_from_index(particles, weights, indexes)
"""
