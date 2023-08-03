import math 
import numpy as np
import random
import cv2

from subtraction import display
import matplotlib.pyplot as plt

def distanceDroite3D(droite,pts):
    """ distance between a line and a pts in 3D

    Args:
        droite (array): a line in 3D (2 pts: [x,y,t])
        pts (array): a point

    Returns:
        int: distance
    """
    
    droiteD=droite[0]-droite[1]
    droiteD=np.int64(droiteD)

    normeDroite=math.sqrt(droiteD[0]**2 + droiteD[1]**2 + droiteD[2]**2)
    AP=pts-droite[0]
    
    cross=np.cross(AP,droiteD)
    cross=np.int64(cross)

    normecross=math.sqrt(cross[0]**2 + cross[1]**2 + cross[2]**2)

    if normeDroite<0.000000000001:
        return 100000.0
    
    return normecross/normeDroite


def distanceDroite(droite,pts):
    """distance between a line and a pts in 2D

    Args:
        droite (array): line in 2D (2 pts)
        pts (array): a point in 2D
        
    Returns:
        int: distance
    """
    a=(droite[1][1]-droite[0][1])/(droite[1][0]-droite[0][0])
    b=droite[1][1]-a*droite[1][0]
    return abs(a*pts[0]-pts[1]+b)/math.sqrt(a**2+1)

def distanceCoude(droite,pts,type):
    """ distance between a pts and a coude (2 lines) in 3D

    Args:
        coude (array): [a,b][b,c] where abc = 3D pts
        pts (array): a point in 3D
        type (int): _description_


    Returns:
        _type_: _description_
    """

    a0=droite[0][0]
    a1=droite[0][1]

    b0=droite[1][0]
    b1=droite[1][1]

    m0=pts[0]
    m1=pts[1]

    X= ( (b0-a0)*(m0-b0)+(b1-a1)*(m1-b1) )/( (b0-a0)**2+(b1-a1)**2 )

    x=b0+(b0-a0)*X
    y=b1+(b1-a1)*X

    D=1000
    coeff=50

    # uh forgot how it work but it does, trust me dude
    if type==1 and ((a0<b0 and x<b0+coeff) or (a0>b0 and x>b0-coeff)) and ((a1<b1 and y<b1+coeff) or (a1>b1 and y>b1-coeff)):
        D= distanceDroite(droite,pts)

    if type==2 and ((a0<b0 and x>b0-coeff) or (a0>b0 and x<b0+coeff)) and ((a1<b1 and y>b1-coeff) or (a1>b1 and y<b1+coeff)):
        D= distanceDroite(droite,pts)


    #if ( (a0>b0 and x>b0-coeff and x<a0+coeff) or (a0<b0 and x<b0+coeff and x>a0-coeff) ) and ( (a1>b1 and x>b1-coeff and x<a1+coeff) or (a1<b1 and x<b1+coeff and x>a1-coeff) ) :
        #D= distanceDroite(droite,pts)

    return D

def distanceDroite3DleRetour(droite,pts,firstDroite,distanceMin=0):
    start=droite[0]
    end=droite[1]
    D=100000.0

    
    if firstDroite:
        # in firtsDroite: pts1  start -- pts2 --> end  pts3
        # pts3 has to be ignored
        if ( (start[0]<end[0] and pts[0]<=end[0]+distanceMin) or (start[0]>end[0] and pts[0]>=end[0]-distanceMin) )  and ( (start[1]<end[1] and pts[1]<=end[0]+distanceMin) or (start[1]>end[1] and pts[1]>=end[0]-distanceMin) ):
            D=distanceDroite3D(droite,pts[:3])
        
    else:
        # in secondDroite: pts1  start -- pts2 --> end  pts3
        # pts1 has to be ignored
        if ((start[0]<end[0] and pts[0]>=end[0]+distanceMin) or (start[0]>end[0] and pts[0]<=end[0]-distanceMin) )  and ( (start[1]<end[1] and pts[1]>=end[0]+distanceMin) or (start[1]>end[1] and pts[1]<=end[0]-distanceMin) ):
            D=distanceDroite3D(droite,pts[:3])
        
    return D

############################################################################################################################################################################################################################################
############################################################################################################################################################################################################################################
############################################################################################################################################################################################################################################

def RANSAC(pts,N=500,distanceMin=20,D3=False):
    best_score=0
    best_model=None
    best_droite=None
    
    for k in range(N):
        score=0
        
        start=random.randint(0, pts.shape[0]-1)
        end=random.randint(pts.shape[0]//2, pts.shape[0]-1)
        
        droite=[pts[start][:],pts[end][:]]
        
        model=np.array((pts[start],pts[end]))

        firstLoop=True
        for P in pts:
            P=P[:3]
            if D3:
                D=distanceDroite3D(droite,P)
            else:
                D=distanceDroite(droite,P)

            """if firstLoop and lastPosition is not None:
                Dlast=np.linalg.norm(P[:2]-lastPosition[:2])
                if firstLoop  and D<distanceMin and Dlast>distanceMin*15:
                    break"""

            if D<distanceMin:
                """if firstLoop:
                    score+=2"""
                score+=1
                model=np.vstack((model,P))
                
            firstLoop=False

        if score>pts.shape[0]//2 and score>best_score :
            best_droite=droite
            best_model=model
            best_score=score

        model=None
        
    if best_model is None:
        return None,None
    else:
        return best_model[2:],best_droite
    
def RANSACcoude(pts,N=500,distanceMin=20,wheightBranche=3,minPts=1/2):
    best_score=0
    best_model=None
    best_droite=None
    best_sum=140000000.0

    for k in range(N):
        
        """
        score=0
        
        start=random.randint(0, pts.shape[0]//2)
        end=random.randint(pts.shape[0]//2, pts.shape[0]-1)
        middle=random.randint(start, end)
            
        droite1=[pts[start][:3],pts[middle][:3]]
        droite2=[pts[middle][:3],pts[end][:3]]
        
        model=np.array((pts[start],pts[end]))
        
        branche1=0
        branche2=0
        
        Sum=0
        
        for P in pts:
            if P[2]<droite1[1][2]:
                D= distanceDroite3DleRetour(droite1, P,True)
                BR1=True
            else:
                D= distanceDroite3DleRetour(droite2, P,False)
                BR1=False
            Sum+=D
            
            if D<=distanceMin:
                score+=1+P[3]
                model=np.vstack((model,P))
                
                if BR1:
                    branche2+=1
                else:
                    branche1+=1
                    
        if  score>best_score and Sum<best_sum and branche1>3 and branche2>3:
            best_droite=[droite1,droite2]
            best_model=model
            best_score=score
            best_sum=Sum
                
        """
        
        start=random.randint(0, pts.shape[0]//2)
        end=random.randint(pts.shape[0]//2, pts.shape[0]-1)
        middle=random.randint(start, end)
    
        droite1=[pts[start][:3],pts[middle][:3]]
        droite2=[pts[middle][:3],pts[end][:3]]
        
        model=np.array((pts[start],pts[end]))
        
        branche1=0
        branche2=0
        score=0.0
        Sum=0

        for P in pts: 
        
            Dbr1=1000000.0
            Dbr2=1000000.0
            
            if pts[middle][2]>P[2]:
                Dbr1=distanceDroite3D(droite1,P[:3])
            else:
                Dbr2=distanceDroite3D(droite2,P[:3])
                
            D=min(Dbr1,Dbr2)
            Sum+=D

            if D<distanceMin or (P[3]==0.0 and D<(distanceMin*3/2)):
                Sum-=D/2
                score+=1.0+P[3]
                
                #if P[3]!=0.0:
                model=np.vstack((model,P))
                    
                if Dbr1>=Dbr2:
                    branche2+=1
                else:
                    branche1+=1
                
                    
        if  score>minPts and score>best_score and branche1>=wheightBranche and branche2>=wheightBranche:
            best_droite=[droite1,droite2]
            best_model=model
            best_score=score
            best_sum=Sum

        model=None
        
    if best_model is None:
        return None,None
    else:
        best_droite[0][0]=best_model[0][:3]
        best_droite[1][1]=best_model[-1][:3]
        return best_model[2:],best_droite
    
############################################################################################################################################################################################################################################
############################################################################################################################################################################################################################################
############################################################################################################################################################################################################################################

def plotLines(lines,frame=None,color= [10, 10, 255],disappear=False,limit=20):
    
    L=len(lines)
    
    if L<=limit or disappear==False:
        min=0
    else:
        min=L-limit

    for index in range(min,L-1): 
        cv2.line(frame, (int(lines[index ][0]),int(lines[index ][1])), (int(lines[index+1 ][0]),int(lines[index+1 ][1])), color, 3)  

def doRansac( position, frame,distMin=20,D3=False ):
    
    model,droite=RANSACcoude(position,N=position.shape[0]**3,distanceMin=distMin)

    
    if 0: #model is not None:
        cv2.line(frame, droite[0][1][:2], droite[0][0][:2], [0, 0, 0], 5) 
        cv2.line(frame, droite[1][1][:2], droite[1][0][:2], [0, 0, 0], 5) 
        #cv2.line(frame, droite[1][:2], droite[0][:2], [0, 0, 0], 5)
        

    return model,droite

def drawLines(all_lines,frame,thick=10):
    for item in all_lines:
        start_pts=item[0][0]
        middle_pts=item[0][1]
        end_pts=item[1][1]
        
        cv2.line(frame, (int(item[0][1][0]),int(item[0][1][1])), (int(item[0][0][0]),int(item[0][0][1])), [0, 0, 0], thick*2) 
        cv2.line(frame, (int(item[1][1][0]),int(item[1][1][1])), (int(item[1][0][0]),int(item[1][0][1])), [0, 0, 0], thick*2) 
        
        cv2.circle(frame, (int(middle_pts[0]),int(middle_pts[1])), 2, [20, 200, 20], 20)
        cv2.circle(frame, (int(start_pts[0]),int(start_pts[1])), 2, [20, 100, 255], 20)
        cv2.circle(frame, (int(end_pts[0]),int(end_pts[1])), 2, [255, 100, 20], 20)
    
    last=None
    for item in all_lines:
        start_pts=item[0][0]
        middle_pts=item[0][1]
        end_pts=item[1][1]
        
        if last is not None:
            cv2.line(frame, (int(last[0]),int(last[1])), (int(start_pts[0]),int(start_pts[1])), [255, 255, 255], 5) 
            
        cv2.line(frame, (int(start_pts[0]),int(start_pts[1])), (int(middle_pts[0]),int(middle_pts[1])), [255, 255, 255], 5) 
        cv2.arrowedLine(frame, (int(middle_pts[0]),int(middle_pts[1])), (int(end_pts[0]),int(end_pts[1])), [255, 255, 255], 5) 
        
        cv2.circle(frame, (int(middle_pts[0]),int(middle_pts[1])), 2,[255, 255, 255], 10)
        cv2.circle(frame, (int(start_pts[0]),int(start_pts[1])), 2, [255, 255, 255], 10)
        cv2.circle(frame, (int(end_pts[0]),int(end_pts[1])), 2,[255, 255, 255], 10)
        
        last=end_pts