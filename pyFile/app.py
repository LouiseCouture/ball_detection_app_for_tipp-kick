import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import math 
import matplotlib

# Changing the current working directory
os.chdir(r"C:\Users\lc100\Documents\GitHub\\ball_detection_app_for_tipp-kick")

from subtraction import *
from checkB import *
from wavelet import *
from ransac import *
from white_select import *
from particle3D import *

def detect_ball(model, frame: np.ndarray):
    WIDTH = 360
    HEIGHT = 360
    resized_frame = cv2.resize(frame, (WIDTH, HEIGHT))
    detection = model(resized_frame)
    bounding_box = detection.xyxy[0].numpy()

    for box in bounding_box:
        if box[5] == 0:
            x = box[0] + (box[2] - box[0]) / 2
            y = box[1] + (box[3] - box[1]) / 2
            return (int(x), int(y)), (box[0], box[1], box[2], box[3])
            # array([     124.15,         179,      130.78,      185.98,      0.8651,           0], dtype=float32)
            # box[0]:   Left
            # box[1]:   Top
            # box[2]:   Right
            # box[3]:   Bottom
            # box[4]:   Probability
            # box[5]:   Klasse 0 = Ball

    return False, False

def get_yolo():
    b = plt.get_backend()
    ball_model = torch.hub.load('ultralytics/yolov5', 'custom', path='03_Ball_Detection/models/ball_weights_V2.pt', force_reload=False)
    matplotlib.use(b)
    return ball_model

ball_model=get_yolo()

def detect_ball_static(image,model=ball_model):

    height, width, _ = image.shape

    # start object detection
    # ball detection returns center coordinates from the ball
    ball_center, ball_bb = detect_ball(model, image)
    size=360
    if ball_center:  
        ball_center = [ball_center[0] * (width / size), ball_center[1] * (height / size)] 
        ball_bb=[int(ball_bb[0]* (width / size)),
                int(ball_bb[1]* (height / size)),
                int(ball_bb[2]* (width / size)),
                int(ball_bb[3]* (height / size))]  
        
        return [ball_bb[0],ball_bb[1],ball_bb[2]-ball_bb[0]+20,ball_bb[3]-ball_bb[1]+20]
    return None

##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################

# 2 frames for the substraction
frame1=None #newest frame
frame0=None #previous frame

background=None
boxSub=None

# 1st frame will be use to detect the white borders of the field
first_frame=None
white_border=None

# time jump between 2 frame
jump=10 #60
# index is the time of the frame
idx=jump*2

#3D points (x,y,t,type)
wavePoint=np.array([[0,0,0]])#wavePoint=np.array([[0,0,0,0]])
#3D Blob points (x,y,t,type)
blobPoint=np.array([[0,0,0,0]])

movementPoint=np.array([[0,0,0]])

#3D points wavePoint+particlePoint (x,y,t,type), 0=move, 1=static, 2=particle
allPoints=np.array([[0,0,0,0]])

sizeBox=38

# RANSAC ##################################################################################################################################################
size_slice=30 #40  #20
#dist min for RANSAC
distMinRANSAC=30 #35 #30     #25 #20 #70
#min pts for Ransac being correct
minPtsRansac=float(size_slice)*1/3
#min pts on each branche
Wbranche=5
#pts after ransac
corrected=np.array([[0,0,0,0]])
#pts after correction of the trajectory
corrected2=np.array([[0,0,0,0]])
# index of the last pts that had is trajectory corrected in corrected[]
lastcorrected=0 
lastLen=0
firstLoop=True

# all the lines found by RANSAC
all_lines=[ [ [[0,0,0],[0,0,0]],[[0,0,0],[0,0,0]] ] ]
#all_lines=[[[0,0,0],[0,0,0]]]

model=None
lastwavePoint=None
detector=initBlobDetect()

#static detection ##################################################################################################################################################
static=True   
static_box=None
lastStaticPts=None
boxStat=None
verifyBox=None 

# count nb green pixels
lastGreenScore=-1
#visual green score evolution
GSplot=[0.0]
variance=np.array([[0,0]])
sumPix=np.array([0,0])
waveCoeff=np.array([[0,0]])

# if samePlace>minDetec then its VERIFIED static
samePlace=0
minDetec=5
movement=0

#video ##################################################################################################################################################
#cap = cv2.VideoCapture('video_record/output.mp4')
cap = cv2.VideoCapture(0)
# Check if camera opened successfully
if (cap.isOpened()== False): 
    print("Error opening video stream or file")
# Capture frame-by-frame
ret, frame_og= cap.read()

#size
scale_percent = 1500//np.shape(frame_og)[0] # percent of original size
W=np.shape(frame_og)[1]*scale_percent
H=np.shape(frame_og)[0]*scale_percent

#templates ##################################################################################################################################################
"""
#template = cv2.imread('template/image.png')
#template = cv2.imread('template/templateRGB_both.png')
template = cv2.imread('template/templateRoom.png')
template_W= Wavelet(template)

#templateB = cv2.imread('template/templateB.png')
templateB = cv2.imread('template/templateRoom.png')
template_B = Wavelet(templateB)

"""
#get templates ##################################################################################################################################################
"""
template=getTemplate(W,H,cap)
template_W= Wavelet(template)
"""

boxStat=None

while boxStat is None:
    print("------------------------LOOKING FOR TEMPLATE------------------------ ")
    ret, frame_og= cap.read()
    boxStat = detect_ball_static(frame_og,ball_model) 
    
box=[min(boxStat[0]-sizeBox//2,0),min(boxStat[0]-sizeBox//2,0),sizeBox,sizeBox]
template=frame_og[box[1]:(box[1]+box[3]),box[0]:(box[0]+box[2]),:]
template_W=Wavelet(template)

#particle ##################################################################################################################################################
NUM_PARTICLES=200
VEL_RANGE=50.0
TIME_VEL=jump
POS_SIGMA =50.0
distParLoca=60 # distance min so average of particle is correct
location=None
particleLoc=np.array([[0,0,0]])

particles=initialize_particles(N=NUM_PARTICLES,velocity=VEL_RANGE,width=W,height=H,time=jump)

##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
print("------------------------START------------------------ ")

#Read until video is completed
while(cap.isOpened()):
    
    # Capture frame-by-frame
    ret, frame_og= cap.read()
    
    if ret == True:
        target=None
        model=None
        
        # resize image
        frame_og_crop = resizeFrame(W,H,frame_og.copy())
        frame_display = frame_og_crop.copy()
        #last_frame = frame_og_crop.copy()

        #select background ######################################################################################################################################
        if first_frame is None:
            frame0 = frame_og_crop.copy()
            frame1 = frame_og_crop.copy()
            
            background = frame_og_crop.copy()
            background = cv2.medianBlur(background, 21)
            background = cv2.cvtColor(background, cv2.COLOR_BGR2YCR_CB)
            background = background.astype('float32')
            
            first_frame = frame_og_crop.copy()

        frame0 = frame1.copy()
        frame1 = frame_og_crop.copy()
           
        
        #object detection ######################################################################################################################################
        """
        boxStat=None
        if samePlace<minDetec:
            boxStat = detect_ball_static(frame_og_crop.copy(),ball_model) 
            
        if (samePlace>=minDetec or boxStat is None) :
            
            verifyBox=None
            
            if lastStaticPts is not None: # has to verified last static box is still here
                print("------------------------WAVELET CHECK STATIC BOX------------------------")
                checkBoxStat=[[lastStaticPts[0]-34,lastStaticPts[1]-34,68,68]]
                box_cp = frame_og_crop[checkBoxStat[0][1]:(checkBoxStat[0][1]+checkBoxStat[0][3]),checkBoxStat[0][0]:(checkBoxStat[0][0]+checkBoxStat[0][2]),:]
                display(box_cp,name="box for wave")
                
                verifyBox,waveBoxStatic = checkBoxes(frame_og_crop.copy(),checkBoxStat,coeff=10.0,wave_temp=template_W,wave_temp2=None,verify=True) #coeff=1.0
            
            # frame substraction ######################################################################################################################################
            frameHLS0=cv2.cvtColor(frame0,cv2.COLOR_BGR2YUV)
            frameHLS1=cv2.cvtColor(frame1,cv2.COLOR_BGR2YUV)
            
            frame_substraction=substraction(frameHLS0,frameHLS1,blur_type=0,blur=21,threshold_type=0,threshold=150,show=True,erode=4,dilate=10) #threshold=120
                
            if verifyBox is None: #staticbox not here, ball moved
                print("------------------------WAVELET LOST STATIC BOX------------------------")
                boundRect=None
                dil,contours=drawContour(frame_substraction,kernelsize=30)
                #put boxes around objects
                maskRec,boundRect=boundingBoxes(contours,show=True,width=W,height=H)
                #detect the ball in one of the boxes (if any)
                boxSub,_ = checkBoxes(frame_og_crop.copy(),boundRect,coeff=20.0,wave_temp=template_W,wave_temp2=None,show=False) #coeff=15qqq
                
                if boxSub is None: # don't see movement -> go back to static
                    print("------------------------WAVELET DON'T SEE MOVEMENT------------------------")
                    samePlace=0
            else:
                print("------------------------WAVELET FOUND STATIC BOX: TEMPLATE UPDATE------------------------")
                template=frame_og_crop[verifyBox[1]:(verifyBox[1]+verifyBox[3]),verifyBox[0]:(verifyBox[0]+verifyBox[2]),:]
                display(template,name='template right now')
                template_W= waveBoxStatic
                
        """
        
        boxStat=None
        if samePlace<minDetec:
            print("------------------------OBJECT DETECTION------------------------")
            boxStat = detect_ball_static(frame_og_crop.copy(),ball_model) 
            if boxStat is not None:
                boxGreen=[boxStat[0]-sizeBox//4,boxStat[1]-sizeBox//4,sizeBox,sizeBox]
                imgBoxGreen=frame_og_crop[boxGreen[1]:(boxGreen[1]+boxGreen[3]),boxGreen[0]:(boxGreen[0]+boxGreen[2]),:]
                cv2.rectangle(frame_display, (int(boxGreen[0]), int(boxGreen[1])),(int(boxGreen[0]+boxGreen[2]), int(boxGreen[1]+boxGreen[3])), [255, 255, 255], 5) 
                
        
        # frame substraction ######################################################################################################################################
        frameHLS0=cv2.cvtColor(frame0,cv2.COLOR_BGR2YUV)
        frameHLS1=cv2.cvtColor(frame1,cv2.COLOR_BGR2YUV)
        
        frame_substraction,var,sumTot=substraction(frameHLS0,frameHLS1,blur_type=0,blur=15,threshold_type=0,threshold=30,show=False,erode=4,dilate=10) #threshold=60
        
        maskHand1=cv2.bitwise_not(selectRedHSV(frame1,show=False))
        maskHand0=cv2.bitwise_not(selectRedHSV(frame0,show=False))
        maskHand=cv2.bitwise_or(maskHand0,maskHand0, mask=maskHand1)
        
        frame_substraction = cv2.bitwise_and(frame_substraction,frame_substraction, mask=maskHand)
        
        display(frame_substraction,name='remove hand')
                                     
        if(sumTot>0):
            variance=np.append(variance,[[var,idx]],axis=0) 
            #sumPix=np.append(sumPix,sumTot) q
        
        if ( var<=500) and  lastStaticPts is not None : #samePlace>=minDetec or (boxStat is None and samePlace<minDetec and lastStaticPts is not None):
            """
            boxGreen=[lastStaticPts[0]-sizeBox//2,lastStaticPts[1]-sizeBox//2,sizeBox,sizeBox]
            imgBoxGreen=frame_og_crop[boxGreen[1]:(boxGreen[1]+boxGreen[3]),boxGreen[0]:(boxGreen[0]+boxGreen[2]),:]
            cv2.rectangle(frame_display, (int(boxGreen[0]), int(boxGreen[1])),(int(boxGreen[0]+boxGreen[2]), int(boxGreen[1]+boxGreen[3])), [20, 255, 20], 2) 
            
            greenScore=selectGreenHSV(imgBoxGreen)
            
            GSplot=np.append(GSplot,[abs(greenScore-lastGreenScore)]) 
            
            print("Green: ",lastGreenScore,greenScore,(3/100)*sizeBox*sizeBox,abs(greenScore-lastGreenScore))
            
            if  (boxStat is None and samePlace<minDetec) or abs(greenScore-lastGreenScore)>(3/100)*sizeBox*sizeBox or greenScore>(99/100)*sizeBox*sizeBox:
                print("------------------------GREEN: MOVEMENT------------------------")
                samePlace=0
                
                for box in boundRect:
                    pts=middle(box)
                    P=[pts[0],pts[1],idx]
                    movementPoint=np.append(movementPoint,[P],axis=0) 
            """
            
            dil,contours=drawContour(frame_substraction,kernelsize=10)
            maskRec,boundRect=boundingBoxes(contours,show=True,width=W,height=H)
            movementPoint=np.array([[0,0,0]])
        
            print("------------------------WAVELET CHECK STATIC BOX------------------------")
            checkBoxStat=[[lastStaticPts[0]-sizeBox//2,lastStaticPts[1]-sizeBox//2,sizeBox,sizeBox]]
            box_cp = frame_og_crop[checkBoxStat[0][1]:(checkBoxStat[0][1]+checkBoxStat[0][3]),checkBoxStat[0][0]:(checkBoxStat[0][0]+checkBoxStat[0][2]),:]
            display(box_cp,name="box for wave")
            verifyBox,waveCoeffBox = checkBoxes(frame_og_crop.copy(),checkBoxStat,coeff=25.0,wave_temp=template_W,wave_temp2=None,verify=True) #coeff=1.0
            waveCoeff=np.append(waveCoeff,[[waveCoeffBox,idx]],axis=0) 
            
            if (boxStat is None and samePlace<minDetec) or verifyBox is None: 
                print("------------------------WAVE: MOVEMENT------------------------")
                samePlace=0
                
                for box in boundRect:
                    pts=middle(box)
                    P=[pts[0],pts[1],idx]
                    movementPoint=np.append(movementPoint,[P],axis=0) 
            else:
                print("------------------------WAVE: STATIC------------------------")
                #lastGreenScore=greenScore
        
        if(samePlace>=minDetec): 
            print("------------------------TEMPLATE UPDATE------------------------")
            template_W= Wavelet(template)
            display(template,name='template right now')
                
                
            
        # Type of movement ######################################################################################################################################
        if boxStat is not None and lastStaticPts is not None:
            pts=middle(boxStat)
            P=[pts[0],pts[1],idx,1.5]
            D=((pts[0]-lastStaticPts[0])**2+(pts[1]-lastStaticPts[1])**2)
            
            if D>100:
                print("------------------------MOVING------------------------")
                samePlace=0
            elif (samePlace<minDetec):
                print("------------------------STATIC------------------------")
                samePlace+=1
                
            if samePlace>=minDetec:
                print("------------------------VERIFIED STATIC: TEMPLATE UPDATE------------------------")
                particles=RE_initialize_particles(pts,N=NUM_PARTICLES,velocity=VEL_RANGE,size=34,time=TIME)
                
                template=frame_og_crop[pts[1]-sizeBox//2:pts[1]+sizeBox//2,pts[0]-sizeBox//2:pts[0]+sizeBox//2,:]
                template_W= Wavelet(template)
                lastGreenScore=selectGreenHSV(template)
                
                display(template,name='template right now')
            
            allPoints=np.append(allPoints,[P],axis=0) 
            
        #particles############################################################################################ 
        target=None
        
        if boxStat is not None:
            print("------------------------PARTICLE STATIC------------------------")
            pts=middle(boxStat)
            lastStaticPts=pts
            target=[[pts[0],pts[1]]]
            
        elif len(movementPoint)>1:
            print("------------------------PARTICLE SUBTRACTION------------------------")
            target=movementPoint[1:]
            wavePoint=np.concatenate((wavePoint,target))
            
            
        if target is not None:
            print("------------------------PARTICLE------------------------")
            particles,location=particlesDetect(particles,target,idx,N=NUM_PARTICLES,width=W,height=H,sigma=POS_SIGMA)  
            
            pts=target[0]
            minD=((pts[0]-location[0])**2+(pts[1]-location[1])**2)
            
            for item in target:
                D=((item[0]-location[0])**2+(item[1]-location[1])**2)
                if D<minD:
                    minD=D
                    pts=item
            
            allPoints=np.append(allPoints,[[location[0],location[1],idx,type]],axis=0) 
            allPoints=np.append(allPoints,[[pts[0],pts[1],idx,type]],axis=0) 
            

        #plot######################################################################################################################################
        
        #particles   
        for index, item in enumerate(particles): 
            cv2.circle(frame_display, (int(item[0]),int(item[1])), 2, [255, 20, 20], 5)  
        if location is not None:
            cv2.circle(frame_display, (int(location[0]),int(location[1])), 2, [20, 20, 255], 15) 
        # all green 
        for index, item in enumerate(allPoints): 
            cv2.circle(frame_display, (int(item[0]),int(item[1])), 2, [20, 255, 20], 5)
        # moving object
        for index, item in enumerate(movementPoint): 
            cv2.circle(frame_display, (int(item[0]),int(item[1])), 2, [160, 50, 220], 15)
            
            
        """
        # wave purple  
        for index, item in enumerate(wavePoint): 
            cv2.circle(frame_display, (int(item[0]),int(item[1])), 2, [140, 20, 200], 5)
        # static green box last static
        if verifyBox is not None:
            checkBoxStat=[[lastStaticPts[0]-sizeBox//2,lastStaticPts[1]-sizeBox//2,sizeBox,sizeBox]]
            cv2.rectangle(frame_display, (int(verifyBox[0]), int(verifyBox[1])),(int(verifyBox[0]+verifyBox[2]), int(verifyBox[1]+verifyBox[3])), [20, 255, 20], 2) 
            
        # Blob  
        for index, item in enumerate(blobPoint): 
            cv2.circle(frame_og_crop, (int(item[0]),int(item[1])), 2, [0, 0, 0], 5)
            
        drawLines(all_lines,frame_og_crop,thick=distMinRANSAC)
        
        for index, item in enumerate(corrected): 
            cv2.circle(frame_og_crop, (int(item[0]),int(item[1])), 2, [20, 20, 255], 5)
        
        for index, item in enumerate(particleLoc): 
            cv2.circle(frame_og_crop, (int(item[0]),int(item[1])), 2, [20, 20, 255], 15)  
        """
        
        display(frame_display,name='detection')
        
        idx+=jump
        #if idx==58600:
            
        #next loop ######################################################################################################################################

        # press ESC to escape, press for long  
        if cv2.waitKey(1)==27:
            if not cv2.waitKey(1)==27:
                break
            
            
  # Break the loop
    else: 
        break
    


##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################



fig = plt.figure(figsize = (8,8)) ######################################################################################################################################
ax = fig.add_subplot(111, projection='3d')
ax.grid()

ax.scatter(allPoints[:,1],allPoints[:,0],allPoints[:,2],marker="o",color='g')
#ax.scatter(blobPoint[:,1],blobPoint[:,0],blobPoint[:,2],marker="+",color='b')
#ax.scatter(wavePoint[:,1],wavePoint[:,0],wavePoint[:,2],marker="+",color='r')

#ax.scatter(corrected[:,1],corrected[:,0],corrected[:,2],marker="o",color='r')
#ax.scatter(particleLoc[:,1],particleLoc[:,0],particleLoc[:,2],marker="o",color='r')

#for index, item in enumerate(wavePoint): 
#     if (item[2]//jump)%15==0:
#          ax.scatter(item[1],item[0],item[2],marker="o",color='r')


ax.set_ylabel('x', labelpad=20)
ax.set_xlabel('y', labelpad=20)
ax.set_zlabel('t', labelpad=20)


fig = plt.figure(figsize=(10, 7))

fig.add_subplot(1,2,1)
variance=variance[1:]
plt.plot(variance[:,1],variance[:,0])
limitey=[500,500]
limitex=[0,idx]
plt.plot(limitex,limitey)

fig.add_subplot(1,2,2)
waveCoeff=waveCoeff[1:]
plt.plot(waveCoeff[:,1],waveCoeff[:,0])
print(waveCoeff)
limitey=[25.0,25.0]
limitex=[0,idx]
plt.plot(limitex,limitey)

"""

fig2 = plt.figure(figsize = (8,8)) ######################################################################################################################################
ax2 = fig2.add_subplot(111, projection='3d')
ax2.grid()
ax2.scatter(corrected[:,1],corrected[:,0],corrected[:,2],marker='+',color='g')

for item in all_lines:
     start_pts=item[0][0]
     middle_pts=item[0][1]
     end_pts=item[1][1]
     
     ax2.scatter(middle_pts[1],middle_pts[0],middle_pts[2],color='g')
     ax2.scatter(end_pts[1],end_pts[0],end_pts[2],color='b')
     ax2.scatter(start_pts[1],start_pts[0],start_pts[2],color='r')
     
     ax2.plot([start_pts[1], middle_pts[1]], [start_pts[0], middle_pts[0]], zs=[start_pts[2], middle_pts[2]],color='k')
     ax2.plot([end_pts[1], middle_pts[1]], [end_pts[0], middle_pts[0]], zs=[end_pts[2], middle_pts[2]],color='k')

ax2.set_ylabel('x', labelpad=20)
ax2.set_ylim(0, 1700)
ax2.set_xlabel('y', labelpad=20)
ax2.set_xlim(0, 1100)
ax2.set_zlabel('t', labelpad=20)


fig = plt.figure(figsize = (8,8)) ######################################################################################################################################

plt.plot(corrected[:,2],corrected[:,1],'g')
plt.plot(corrected[:,2],corrected[:,0],'r')
plt.plot(corrected[:,2],corrected[:,1],'g+')
plt.plot(corrected[:,2],corrected[:,0],'r+')

fig = plt.figure(figsize = (8,8)) ######################################################################################################################################

plt.plot(corrected2[:,2],corrected2[:,1],'k')
plt.plot(corrected2[:,2],corrected2[:,0],'b')
"""

plt.show()

key = cv2.waitKey(0)
#When everything done, release the video capture object
cap.release() 
# Closes all the frames
cv2.destroyAllWindows()
