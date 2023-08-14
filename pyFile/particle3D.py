import matplotlib.pyplot as plt
import cv2
import numpy as np
import math 
from numpy.random import uniform,randn
import os


# Changing the current working directory
os.chdir(r"C:\Users\lc100\Documents\GitHub\\ball_detection_app_for_tipp-kick")


NUM_PARTICLES=500
VEL_RANGE=50.0
TIME_VEL=100.0
WIDTH=2000.0
HEIGHT=2000.0
TIME=5000.0
POS_SIGMA = 15.0

"""
    0=y
    1=x
    2=t
    
"""

######################################################################################################################################

def initialize_particles(N=NUM_PARTICLES,velocity=VEL_RANGE,width=WIDTH,height=HEIGHT,time=TIME):
    particles=np.random.rand(N,6)
    particles=particles*np.array((width,height,time,velocity,velocity,velocity))
    particles[:,3:5] -= velocity/2.0
    return particles

######################################################################################################################################

def RE_initialize_particles(pts,N=NUM_PARTICLES,velocity=VEL_RANGE,size=34,time=TIME):
    particles=np.random.rand(N,6)
    
    UP0=pts[0]+size
    UP1=pts[1]+size
    DO0=pts[0]-size
    DO1=pts[1]-size
    
    particles=particles*np.array((UP0-DO0,UP1-DO1,time,velocity,velocity,velocity))
    particles[:,0:2] += [DO0,DO1]
    particles[:,3:5] -= velocity/2.0
    
    return particles

######################################################################################################################################

def apply_velocity(particles):
    particles[:,0] += particles[:,3]
    particles[:,1] += particles[:,4]
    particles[:,2] += particles[:,5]
    return particles

######################################################################################################################################

def enforce_edges(particles,N=NUM_PARTICLES,width=WIDTH,height=HEIGHT):
    for i in range(N):
        
        #print("     before: ",particles[i,0],particles[i,1],particles[i,2],particles[i,3])
        vitesse=VEL_RANGE
        
        if particles[i,0]>width-1:
            particles[i,0]=width-1-vitesse
            particles[i,3]=-vitesse
        if particles[i,1]> height-1:
            particles[i,1]=height-1-vitesse
            particles[i,4]=-vitesse
            
        if particles[i,0]< 0:
            particles[i,0]=vitesse
            particles[i,3]=vitesse
        if particles[i,1]< 0:
            particles[i,1]=vitesse
            particles[i,4]=vitesse
            
        #print("     after: ",particles[i,0],particles[i,1])
            
    return particles

######################################################################################################################################

def compute_errors(particles,target,N=NUM_PARTICLES):
    errors=np.zeros(N)
    
    if target is None:
        errors+=10000.0
    else:
        errors=particles[:,:2]-target[:2]
        errors=errors*errors
        errors=np.sum(errors,axis=1)
        errors=np.sqrt(errors)
        
    return errors

######################################################################################################################################

def compute_weights(errors,particles,width=WIDTH,height=HEIGHT,time=TIME):
    weights=np.max(errors)-errors
    weights[
        (particles[:,0]==0) |
        (particles[:,0] == width-1) |
        (particles[:,1]==0) |
        (particles[:,1] == height-1) 
    ]=0.0
    weights=weights**4
    return weights

######################################################################################################################################

def resample(particles, weights,idx,N=NUM_PARTICLES):
    weights+=1.e-100
    somme=np.sum(weights)
    
    probabilities = weights/somme
    index_numbers=np.random.choice(
        N,
        size=N,
        p=probabilities
    )
    
    particles=particles[index_numbers,:]
    y=np.mean(particles[:,0])
    x=np.mean(particles[:,1])
    return particles, ( int(y), int(x), idx)

######################################################################################################################################

def apply_noise(particles,N=NUM_PARTICLES,sigma=POS_SIGMA):
    noise=np.concatenate(
        (
        np.random.normal(0.0, sigma, (N,1)),
        np.random.normal(0.0, sigma, (N,1)),
        np.random.normal(0.0, sigma, (N,1)),
        np.random.normal(0.0, sigma, (N,1)),
        np.random.normal(0.0, sigma, (N,1)),
        np.random.normal(0.0, sigma, (N,1)),
    ),
    axis=1)
    particles+=noise
    return particles

######################################################################################################################################

def particlesDetect(particles,target,idx,N=NUM_PARTICLES,width=WIDTH,height=HEIGHT,sigma=POS_SIGMA):
    
    particles = apply_velocity(particles)
    particles = enforce_edges(particles,N=N,width=width,height=height)
    errors = compute_errors(particles,target, N=N)
    weights = compute_weights(errors,particles,width=width,height=height)
    particles, location = resample(particles, weights,idx,N=N)
    particles = apply_noise(particles,N=N,sigma=sigma)
    
    return particles,location

###############################################################################
"""
Len=100
position=np.load("aaa_position.npy")
position=position[:Len]
print(position.shape)

particleLoc=np.array([[0,0,0]])

fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(111, projection='3d')
ax.grid()

particles=initialize_particles()
idx=0
for pts3D in position:
    ax.scatter(pts3D[1],pts3D[0],pts3D[2],marker="+",color="b")
    
    #for part in particles:
    #    ax.scatter(part[1],part[0],part[2],marker="o",color="g")
    
    particles,location=particlesDetect(particles,pts3D,pts3D[3])
    
    diff=math.sqrt((location[0]-pts3D[0])**2+(location[1]-pts3D[1])**2)
    if diff<=30:
        ax.scatter(location[1],location[0],location[2],marker="o",color="r")
        particleLoc=np.concatenate((particleLoc,[[location[0],location[1],location[2]]]))
        
    #print("location:",location)
    
    fig.show() 
    cv2.waitKey(250)
    
fig.show() 
"""

