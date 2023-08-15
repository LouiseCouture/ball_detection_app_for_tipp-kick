import numpy as np
import cv2
from subtraction import display

def selectWhite(frame,show=False,erode=1,dilate=4,W_limit=165,B_limit=60,H_limit=1):
    """ select white using HLS

    Args:
        frame (array): frame
        show (bool, optional): Defaults to False.
        erode (int, optional): number of iteration for the erode method. Defaults to 1.
        dilate (int, optional): number of iteration for the erode method. Defaults to 4.
        less (int, optional): accuracy coeff. Defaults to 0.

    Returns:
        array: mask of white
    """

    frameHSV= cv2.cvtColor(frame,cv2.COLOR_BGR2HLS)
    blurred = cv2.GaussianBlur(frameHSV, (11, 11), 0)

    lower_white = np.array([H_limit,W_limit,0])
    upper_white = np.array([100,255,255])
    
    lower_black = np.array([H_limit,0,0])
    upper_black = np.array([100,B_limit,255])

    maskW = cv2.inRange(blurred, lower_white, upper_white)
    maskB = cv2.inRange(blurred, lower_black, upper_black)

    mask = cv2.bitwise_or(maskW,maskB)
    mask = cv2.erode(mask, None, iterations=erode)
    mask = cv2.dilate(mask, None, iterations=dilate)

    if show:
        display(mask,'select white')

    return mask

#######################################################################################################

def selectWhiteHSV(frame,show=False,erode=1,dilate=4,less=0):
    """ select white using HSV

    Args:
        frame (array): frame
        show (bool, optional): Defaults to False.
        erode (int, optional): number of iteration for the erode method. Defaults to 1.
        dilate (int, optional): number of iteration for the erode method. Defaults to 4.
        less (int, optional): accuracy coeff. Defaults to 0.

    Returns:
        array: mask of white
    """

    frameHSV= cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(frameHSV, (11, 11), 0)

    lower_white = np.array([20,0,160])
    upper_white = np.array([70,80,255])
    
    lower_black = np.array([20,0,0])
    upper_black = np.array([70,80,160])

    maskW = cv2.inRange(blurred, lower_white, upper_white)
    maskB = cv2.inRange(blurred, lower_black, upper_black)

    mask = cv2.bitwise_or(maskW,maskB)
    mask = cv2.erode(mask, None, iterations=erode)
    mask = cv2.dilate(mask, None, iterations=dilate)

    if show:
        display(mask,name='select white')

    return mask

#######################################################################################################

def selectGreenHSV(frame):

    frameHSV= cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(frameHSV, (11, 11), 0)

    lower_white = np.array([40, 50,50])
    upper_white = np.array([100, 255,255]) #70
    
    mask = cv2.inRange(blurred, lower_white, upper_white)/255
    display(mask,name="mask green")
    sumGreen=np.sum(mask)

    return sumGreen