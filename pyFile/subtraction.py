import cv2
import numpy as np
import random as rng

def substraction(y0,y1,blur_type=0,threshold_type=0,threshold=50,blur=29,show=False,erode=0,dilate=0):
    """_summary_

    Args:
        y0 (array): one dimension of the frame 0
        y1 (array): one dimension of the frame 0
        blur_type (int, optional): 0= gaussian 1= median. Defaults to 0.
        threshold_type (int, optional): 0=binary 1=tozero 2=adaptive threshold. Defaults to 0.
        threshold (int, optional):  Defaults to 50.
        blur (int, optional): parameter for the blur. Defaults to 29.
        show (bool, optional):  Defaults to False.

    Returns:
        _type_: _description_
    """
    diff0=np.zeros((y0.shape[0],y0.shape[1]), dtype=int)
    diff1=np.zeros((y0.shape[0],y0.shape[1]), dtype=int)
    
    for i in range(y0.shape[2]):
        y0_chan=y0[:,:,i]
        y1_chan=y1[:,:,i]
        
        if blur_type==0:
            gb0=cv2.GaussianBlur(y0_chan,(blur,blur),0)
            gb1=cv2.GaussianBlur(y1_chan,(blur,blur),0)
        else:
            gb0=cv2.medianBlur(y0_chan, blur)
            gb1=cv2.medianBlur(y1_chan, blur)
        
        diff0+=cv2.absdiff(gb0,gb1)#+cv2.absdiff(ub0,ub1)+cv2.absdiff(vb0,vb1)
        diff1+=cv2.absdiff(gb1,gb0)#+cv2.absdiff(ub1,ub0)+cv2.absdiff(vb1,vb0)
        
    diff0 = 255 *(diff0 / diff0.max())
    diff0 = diff0.astype(np.uint8)
    diff1 = 255 *(diff1 / diff1.max())
    diff1 = diff1.astype(np.uint8)
    
    if threshold_type==0:
        _, diff0 = cv2.threshold(diff0, threshold, 255, cv2.THRESH_BINARY)
        _, diff1 = cv2.threshold(diff1, threshold, 255, cv2.THRESH_BINARY)
    elif threshold_type==1:
        _, diff0 = cv2.threshold(diff0, threshold, 255, cv2.THRESH_TOZERO  )
        _, diff1 = cv2.threshold(diff1, threshold, 255, cv2.THRESH_TOZERO  )
    else:
        diff0 = cv2.adaptiveThreshold(diff0, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5)
        diff1 = cv2.adaptiveThreshold(diff1, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5)
    
    frame_diff=cv2.add(diff0,diff1)
    frame_diff = cv2.erode(frame_diff, None, iterations=erode)
    frame_diff = cv2.dilate(frame_diff, None, iterations=dilate)
    
    var=int(np.var(frame_diff))
    sumTot=int(np.sum(frame_diff)/255)
     
    if show:
        displayFrame=frame_diff.copy()
        string="variance: {}".format(var)+"  sum pixel: {}".format(sumTot)
        #displayFrame = cv2.putText(displayFrame, string, (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (220,220,220), 4, cv2.LINE_AA)
        display(displayFrame,name='substraction1')

    return frame_diff,var,sumTot

######################################################################################################

def substractionSAD(background,frame0,frame1,time,show=False,blur=29,threshold=10.0):
    
    frame0=cv2.medianBlur(frame0, blur)
    frame0= cv2.cvtColor(frame0, cv2.COLOR_BGR2YCR_CB)
    frame0=frame0.astype('float32')
    #frame0=frame0[:,:,0]
    
    frame1=cv2.medianBlur(frame1, blur)
    frame1= cv2.cvtColor(frame1, cv2.COLOR_BGR2YCR_CB)
    frame1=frame1.astype('float32')
    #frame1=frame1[:,:,0]
    
    diff1=frame1
    diff0=frame0
    
    
    a=0.3+1/time #0.5
    #newBack=background.copy()
    newBack=(1.0-a)*background+a*frame1
    
    for i in range(3):
        newBack[:,:,i] = (1-a)*background[:,:,i] + frame1[:,:,i]*a
        diff1[:,:,i] = cv2.absdiff(newBack[:,:,i],frame1[:,:,i])
        diff0[:,:,i] =cv2.absdiff(newBack[:,:,i],frame0[:,:,i])
    
    for i in range(3):
        _, diff1[:,:,i] = cv2.threshold(diff1[:,:,i], threshold, 255.0, cv2.THRESH_BINARY)
        _, diff0[:,:,i] = cv2.threshold(diff0[:,:,i], threshold, 255.0, cv2.THRESH_BINARY)
        
    diff=diff1+diff0
    
    for i in range(3):
        _, diff[:,:,i] = cv2.threshold(diff[:,:,i], 1.0, 255, cv2.THRESH_BINARY)
        
    diff=np.sum(diff,axis=2)
    diff=diff.astype('uint8')
    
    diff = cv2.erode(diff, None, iterations=5)
    diff = cv2.dilate(diff, None, iterations=20)

    if show:
        display(diff,name='substractionSAD')

    #thresh=sum(cv2.absdiff(frame0,frame1)) / (frame0.shape[0]*frame0.shape[1])
     
    return newBack,diff


######################################################################################################

def display(frame,name=None):
    """ resize and display a frame

    Args:
        frame (array): frame that will be displayed
        name (string, optional): name of the window. Defaults to None.

    Returns:
        arrray: frame resized
    """
    scale_percent = 60 # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    
    if name is not None:
        cv2.imshow(winname=name, mat=resized)
        
    return resized

######################################################################################################

def drawContour(mask,kernelsize=15):
    """take a mask and draw the contour around the objects

    Args:
        mask (array): Mask with objects on it
        kernelsize (int, optional): size of the kernel. Defaults to 15.

    Returns:
        _type_: _description_
    """
    
    kernel6 = np.ones((kernelsize,kernelsize), np.uint8)
    dil = cv2.dilate(mask, kernel6, iterations=1)
    contours, _ = cv2.findContours(dil, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    return dil,contours


######################################################################################################

def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    
    return new_img

######################################################################################################

def correlation_coefficient(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product
    
######################################################################################################
    
def boundingBoxes(contours,width,height,show=False):
    
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    mask=np.zeros([height,width], dtype=np.int8 )
    
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 5, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
    
    tailleMax=32*4
    for item in boundRect:
        if item[2]<tailleMax and item[3]<tailleMax:
            cv2.rectangle(mask, (int(item[0]), int(item[1])),(int(item[0]+item[2]), int(item[1]+item[3])),1, -1)
        
    
    if show:
        
        drawing = np.zeros((height, width, 3), dtype=np.uint8)
        
        for i in range(len(contours)):
            color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
            cv2.drawContours(drawing, contours_poly, i, color)
            cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
            (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
    
        display(drawing,name='all boxes')

    return mask,boundRect