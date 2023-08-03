import cv2
import numpy as np
import torch
# import tensorflow as tf
# from tensorflow import keras


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

# Load model 
ball_model = torch.hub.load('ultralytics/yolov5', 'custom', path='../03_Ball_Detection/models/ball_weights_V2.pt', force_reload=False)

def detect_ball_static(image,model=ball_model):

    height, width, _ = image.shape

    # start object detection
    # ball detection returns center coordinates from the ball
    ball_center, ball_bb = detect_ball(model, image)
    if ball_center:  
        ball_center = [ball_center[0] * (width / 360), ball_center[1] * (height / 360)] 
        ball_bb=[int(ball_bb[0]* (width / 360)),
                int(ball_bb[1]* (height / 360)),
                int(ball_bb[2]* (width / 360)),
                int(ball_bb[3]* (height / 360))]  
        
        return [ball_bb[0],ball_bb[1],ball_bb[2]-ball_bb[0]+10,ball_bb[3]-ball_bb[1]+10]
    return None


image=cv2.imread('../imgs/20220927_100820.jpg')

height, width, _ = image.shape

# start object detection
# ball detection returns center coordinates from the ball
ball_center, ball_bb = detect_ball(ball_model, image)
if ball_center:  # if ball was detected
    # calculate ball coordinates from 360x360 to current size
    ball_center = [ball_center[0] * (width / 360), ball_center[1] * (height / 360)] 
    ball_bb=[ball_bb[0]* (width / 360),
             ball_bb[1]* (height / 360),
             ball_bb[2]* (width / 360),
             ball_bb[3]* (height / 360)]  

    # # H-4 draw Ball Detection
    cv2.drawMarker(image, (int(ball_center[0]), int(ball_center[1])), (0, 0, 255), cv2.MARKER_CROSS, 10, 2, 8)
    cv2.rectangle(
            image, 
            (int(ball_bb[0]), int(ball_bb[1])), # start_point
            (int(ball_bb[2]), int(ball_bb[3])), # end_point
            (0, 0, 255), # color
            5 #thickness
        )
    cv2.putText(
            image, 
            'Ball', 
            (int(ball_bb[0])-10, int(ball_bb[1])- 10), 
            cv2.FONT_HERSHEY_DUPLEX, 
            2, # text_seize
            (0, 0, 255), # color
            2, # text_thickness 
            cv2.LINE_AA
        )

    cv2.namedWindow('TippKick', cv2.WINDOW_NORMAL)      
    cv2.imshow('TippKick', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()