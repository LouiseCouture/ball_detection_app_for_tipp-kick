import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

# Changing the current working directory
os.chdir(r"C:\Users\lc100\Documents\GitHub\\ball_detection_app_for_tipp-kick")

from subtraction import *
from checkB import *
from wavelet import *
from ransac import *
from white_select import *

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

ball_model = torch.hub.load('ultralytics/yolov5', 'custom', path='03_Ball_Detection/models/ball_weights_V2.pt', force_reload=False)

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

# 1st frame will be use to detect the white borders of the field
first_frame=None
white_border=None

# time jump between 2 frame
jump=60 #60
# index is the time of the frame
idx=jump*2

#3D points (x,y,t,type)
wavePoint=[[0,0,0,0]]
#3D Blob points (x,y,t,type)
blobPoint=[[0,0,0,0]]
#3D points particle(x,y,t)
particlePoint=[[0,0,0]]

#3D points wavePoint+particlePoint (x,y,t,type), 0=move, 1=static, 2=particle
allPoints=[[0,0,0,0]]


#number of pts used for RANSAC
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

static=False
static_box=None

detector=initBlobDetect()


#templates used for the wavelet function templateOutput
#template = cv2.imread('template/image.png')
template = cv2.imread('template/templateRGB_both.png')
template_W= Wavelet(template)

templateB = cv2.imread('template/templateB.png')
template_B = Wavelet(templateB)

#video
cap = cv2.VideoCapture('video_record/1310.mp4')
# Check if camera opened successfully
if (cap.isOpened()== False): 
    print("Error opening video stream or file")


# Capture frame-by-frame
ret, frame_og= cap.read()

#size
W=1700#np.shape(frame_og)[1]
H=np.shape(frame_og)[0]


##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################


#Read until video is completed
while(cap.isOpened()):
    
    # Capture frame-by-frame
    ret, frame_og= cap.read()
    
    if ret == True:
        model=None
        
        frame_og_crop=frame_og[:,:1700,:].copy()
        last_frame = frame_og_crop.copy()

        #detect the white border so to not confuse them with the ball ######################################################################################################################################
        if first_frame is None:
            frame0 = frame_og_crop.copy()
            frame1 = frame_og_crop.copy()
            
            background = frame_og_crop.copy()
            background = cv2.medianBlur(background, 21)
            background = cv2.cvtColor(background, cv2.COLOR_BGR2YCR_CB)
            #background = background[:,:,0]
            background = background.astype('float32')
            
            first_frame = frame_og_crop.copy()
            white_border=selectWhite(first_frame,dilate=9,erode=2)

            white_border=cv2.bitwise_not(white_border)

            #display(white_border,name='not border')

    
        frame0 = frame1.copy()
        frame1 = frame_og_crop.copy()
        
        
        #substraction ######################################################################################################################################
        frameHLS0=cv2.cvtColor(frame0,cv2.COLOR_BGR2YUV)
        frameHLS1=cv2.cvtColor(frame1,cv2.COLOR_BGR2YUV)
        
        frame_substraction=substraction(frameHLS0[:,:,0],frameHLS1[:,:,0],blur_type=1,blur=21,threshold_type=0,threshold=15,show=False,erode=4,dilate=10)
        
        dil,contours=drawContour(frame_substraction,kernelsize=25)
        #display(dil,name='dilatation')
        #put boxes around objects
        maskRec,boundRect=boundingBoxes(contours,show=False,width=W,height=H)
        #detect the ball in one of the boxes (if any)
        box = checkBoxes(frame_og_crop.copy(),boundRect,coeff=15.0,wave_temp=template_W,wave_temp2=template_B,show=False) #coeff=15
        
        
        #Subtraction background and blob ######################################################################################################################################
        
        background,frame_subBackground =substractionSAD(background,frame0.copy(),frame1.copy(),idx//jump,show=True,blur=15,threshold=10.0)
        #frame_subBackground = cv2.bitwise_and(frame_substraction , white_border, mask=None)
        
        ptsBlob=checkBlob(detector,frame_subBackground,frame1.copy(),idx,show=True)
        
        if ptsBlob is not None:
            allPoints=np.concatenate((allPoints,ptsBlob))
            blobPoint=np.concatenate((blobPoint,ptsBlob)) 
            
        #static detection ######################################################################################################################################
        if (box is None and not static) : #did not detected moving ball, static mode on
            static=True    
            box=detect_ball_static(frame_og_crop.copy(),ball_model)     
            static_box= box
        elif box is None and static_box is not None: # still not moving, use the last detection
            box=static_box
        else: # moving, static mode off
            static_box=None
            static=False   
        
        # add the 3D to the points
        if box is not None:
            pts=middle(box)
            
            # if same wavePoint, don't take it.
            if not np.array_equal(pts, wavePoint[-1][:2]) and ((pts[0]-wavePoint[-1][0])**2+(pts[1]-wavePoint[-1][1])**2)>70:
                if static:
                    P=[pts[0],pts[1],idx,1.5]
                else:
                    P=[pts[0],pts[1],idx,0.75]
                wavePoint=np.append(wavePoint,[P],axis=0) 
                allPoints=np.append(allPoints,[P],axis=0) 

            cv2.rectangle(frame_og_crop, (int(box[0]), int(box[1])),(int(box[0]+box[2]), int(box[1]+box[3])), (255,255,255), 2) 
            


        #Ransac ######################################################################################################################################
        
        #tab=wavePoint
        tab=allPoints
        
        Len=len(allPoints)
        #Len1=len(wavePoint)
        
        if Len%size_slice==0 and lastLen!=Len:
            
            if firstLoop:
                tab=tab[1:]

            sliced=tab[(Len-size_slice):]
            
            
            if model is not None:
                sliced=np.concatenate((corrected[len(corrected)-5:],sliced))
            elif firstLoop==False:
                sliced=np.concatenate((tab[(Len-5-size_slice):(Len-1-size_slice)],sliced))
            
            sliced=np.concatenate((tab[(Len-10-size_slice):(Len-1-size_slice)],sliced))
            
            model,droite=RANSACcoude(sliced,N=sliced.shape[0]**2,distanceMin=distMinRANSAC,wheightBranche=Wbranche,minPts=minPtsRansac)
            lastLen=Len

            if model is not None:
                corrected=np.concatenate((corrected,model))
                all_lines=np.append(all_lines,[droite],axis=0)
                

                if firstLoop:
                    corrected=corrected[1:]
                    all_lines=all_lines[1:]
                    
      
        # correct trajectory: if i is shorter to go to the jumpCth pts rather than the next then we jump ####################################################
        
        k=lastcorrected
        lenCorrect=len(corrected)
        jumpC=3
        """
        while k < (lenCorrect-5) :
            diff1=cv2.absdiff(corrected[k],corrected[k+1])
            diff2=cv2.absdiff(corrected[k],corrected[k+jumpC])
            corrected2=np.append(corrected2,[corrected[k]],axis=0)

            if diff1[0]+diff1[1]*1 < (diff2[0]+diff2[1]):
                k+=1
            else:
                k+=jumpC-1
        """
        if firstLoop and model is not None:
            corrected2=corrected2[1:]
            firstLoop=False
        if lenCorrect>5:
            lastcorrected = lenCorrect-jumpC
        
        


        #plot######################################################################################################################################
          
        for index, item in enumerate(particlePoint): 
            cv2.circle(frame_og_crop, (int(item[0]),int(item[1])), 2, [20, 225, 255], 5)
        for index, item in enumerate(wavePoint): 
            if item[3]==0.75: #green is moving
                cv2.circle(frame_og_crop, (int(item[0]),int(item[1])), 2, [20, 255, 20], 5)
            else: # blue is static
                cv2.circle(frame_og_crop, (int(item[0]),int(item[1])), 2, [255, 200, 50], 5)
        for index, item in enumerate(blobPoint): 
            cv2.circle(frame_og_crop, (int(item[0]),int(item[1])), 2, [200, 80, 255], 5)
            
        """
        for index, item in enumerate(allPoints): 
            cv2.circle(frame_og_crop, (int(item[0]),int(item[1])), 2, [20, 255, 20], 5)
        """    
             
        #plotLines(particlePointointoint,frame=frame_og_crop,disappear=True,limit=20) 
        #plotLines(wavePoint,frame=frame_og_crop,disappear=False)
        #plotLines(corrected,frame=frame_og_crop,disappear=False) 
        drawLines(all_lines,frame_og_crop,thick=distMinRANSAC)
        #plotLines(corrected2,frame=frame_og_crop,disappear=False,color= [255, 10, 10]) 
        
        for index, item in enumerate(corrected): 
            cv2.circle(frame_og_crop, (int(item[0]),int(item[1])), 2, [20, 20, 255], 5)

        display(frame_og_crop,name='detection')
        
        idx+=jump
        
        #if idx==58600:
        
            
        #next loop ######################################################################################################################################
        """
        #press key to go the next frame or Q to exit or S to save the frame
        key = cv2.waitKey(0)
        
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('s') or key == ord('S'):
            #box_cp = frame_og[box[1]:(box[1]+box[3]),box[0]:(box[0]+box[2]),:]
            #cv2.imwrite("object1_{}.png".format(idx), box_cp)
            cv2.imwrite("image.png", frame_og_crop)
        else:
            continue
        """
            
        
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

#ax.scatter(allPoints[:,1],allPoints[:,0],allPoints[:,2],marker="+",color='g')
#ax.scatter(corrected[:,1],corrected[:,0],corrected[:,2],marker="o",color='r')

ax.scatter(wavePoint[:,1],wavePoint[:,0],wavePoint[:,2],marker="o",color='g')
#ax.scatter(particlePoint[:,1],particlePoint[:,0],particlePoint[:,2],marker="+")
ax.scatter(blobPoint[:,1],blobPoint[:,0],blobPoint[:,2],marker=",")

#for index, item in enumerate(wavePoint): 
#     if (item[2]//jump)%15==0:
#          ax.scatter(item[1],item[0],item[2],marker="o",color='r')


ax.set_ylabel('x', labelpad=20)
ax.set_ylim(0, 1700)
ax.set_xlabel('y', labelpad=20)
ax.set_xlim(0, 1100)
ax.set_zlabel('t', labelpad=20)
#ax.set_zlim(0, 100)

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

"""
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