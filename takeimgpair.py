# import the opencv library
from threading import Thread
import multiprocessing as mp
import cv2
from cv2 import *
import time



imageRR,imageLL=cv2.imread("", cv2.IMREAD_GRAYSCALE),cv2.imread("", cv2.IMREAD_GRAYSCALE)
def left():
    index=46
    global imageLL
    while index<=65:
        time.sleep(15)
        vidL = cv2.VideoCapture("rtsp://10.6.10.161/live_stream")
        vidL.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        resultL, imageL = vidL.read()
        imageLL=imageL[190:890, 560:1360]
        #cv2.imshow('l',imageLL)
        #cv2.waitKey(0)
        cv2.imwrite("underL/"+str(index)+".png", imageL)
        cv2.imshow('imagel',imageLL)
        index=index+1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break  



def right():
    index=46
    global imageRR
    while index<=65:
        time.sleep(15)
        vidR =cv2.VideoCapture("rtsp://10.6.10.162/live_stream")
        vidR.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        resultR, imageR = vidR.read()
        imageRR=imageR[190:890, 560:1360]
        #cv2.imshow('r',imageRR)
        #cv2.waitKey(0)
        cv2.imwrite("underR/"+str(index)+".png", imageR)
        cv2.imshow('imager',imageRR)
        index=index+1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break  

    #thread1=Thread(target = left)
    #thread2=Thread(target = right)

if __name__ == '__main__':
    capture_processL = mp.Process(target=left)
    capture_processR = mp.Process(target=right)


    capture_processL.start()
    capture_processR.start()
    #thread1.start()
    #thread2.start()
    capture_processL.join()
    capture_processR.join()
    #thread1.join()
    #thread2.join()
    #cv2.waitKey(0)
