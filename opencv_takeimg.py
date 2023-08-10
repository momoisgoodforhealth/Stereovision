# import the opencv library
import matplotlib
import cv2
from cv2 import *
import time


#time.sleep(2)
# define a video capture object
vidR = cv2.VideoCapture('rtsp://10.6.10.162/live_stream')
vidL = cv2.VideoCapture('rtsp://10.6.10.161/live_stream')
  
resultR, imageR = vidR.read()
resultL, imageL = vidL.read()
  
# If image will detected without any error, 
# show result
if resultL:
    imageLL=imageL[200:900, 600:1400]
    imageRR=imageR[200:900, 600:1400]
    cv2.imshow('s',imageLL)
    cv2.imshow('r',imageRR)
    cv2.waitKey(0)
    # showing result, it take frame name and image 
    # output
    #imshow("GeeksForGeeks", image)
  
    # saving image in local storage
    cv2.imwrite("R4.png", imageR)
    cv2.imwrite("L4.png", imageL)
  
    # If keyboard interrupt occurs, destroy image 
    # window
    waitKey(0)
    #destroyWindow("GeeksForGeeks")
  
# If captured image is corrupted, moving to else part
else:
    print("No image detected. Please! try again")