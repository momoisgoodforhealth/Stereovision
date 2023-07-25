# import the opencv library
from threading import Thread
import cv2
from cv2 import *
import time

time.sleep(4)
index=121


imageRR,imageLL=cv2.imread("", cv2.IMREAD_GRAYSCALE),cv2.imread("", cv2.IMREAD_GRAYSCALE)
def left():
    global imageLL
    vidL = cv2.VideoCapture("rtsp://10.6.10.161/live_stream")
    vidL.set(cv2.CAP_PROP_BUFFERSIZE, 0)
    resultL, imageL = vidL.read()
    imageLL=imageL[190:890, 560:1360]
    #cv2.imshow('l',imageLL)
    #cv2.waitKey(0)
    cv2.imwrite("cleanL/"+str(index)+".png", imageL)



def right():
    global imageRR
    vidR =cv2.VideoCapture("rtsp://10.6.10.162/live_stream")
    vidR.set(cv2.CAP_PROP_BUFFERSIZE, 0)
    resultR, imageR = vidR.read()
    imageRR=imageR[190:890, 560:1360]
    #cv2.imshow('r',imageRR)
    #cv2.waitKey(0)
    cv2.imwrite("cleanR/"+str(index)+".png", imageR)


thread1=Thread(target = left)
thread2=Thread(target = right)

thread1.start()
thread2.start()

thread1.join()
thread2.join()

cv2.imshow('l',imageLL)
cv2.imshow('r',imageRR)
cv2.waitKey(0)