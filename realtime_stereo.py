from threading import Thread
import cv2, time
import numpy as np
from matplotlib import pyplot as plt
from multiprocessing import Process, Queue

baseline=100
focal_length= 2.83711978e+02#4.77424789e+02
font = cv2.FONT_HERSHEY_DUPLEX

Q=np.float32([[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00, -3.34334817e+02],
 [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00, -3.14079409e+02],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  2.83711978e+02],
 [ 0.00000000e+00,  0.00000000e+00,  1.01527526e-02, -2.22518594e-01]])

npzfile = np.load('./calibration_data/{}p/stereo_camera_calibration.npz'.format(700))
cv_file = npzfile#cv2.FileStorage("improved_params3.xml", cv2.FILE_STORAGE_READ)
Left_Stereo_Map_x = cv_file['leftMapX']
Left_Stereo_Map_y = cv_file['leftMapY']
Right_Stereo_Map_x = cv_file['rightMapX']
Right_Stereo_Map_y = cv_file['rightMapY']

def left():
    vidL = cv2.VideoCapture("rtsp://10.6.10.161/live_stream")
    vidL.set(cv2.CAP_PROP_BUFFERSIZE, 0)
    resultL, imageL = vidL.read()
    while True:
        imageL=imageL[190:890, 560:1360]
        imageL = cv2.cvtColor(imageL, cv2.COLOR_BGR2GRAY)
        imageL= cv2.remap(imageL, Left_Stereo_Map_x,  Left_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        cv2.imshow('left',imageL)
        #cv2.imshow('left',imageL)
    #global imagell
    #imagell=imageL

        


def right():
    vidR =cv2.VideoCapture("rtsp://10.6.10.162/live_stream")
    vidR.set(cv2.CAP_PROP_BUFFERSIZE, 0)
    resultR, imageR = vidR.read()
    while True:
        imageR=imageR[190:890, 560:1360]
        imageR = cv2.cvtColor(imageR, cv2.COLOR_BGR2GRAY)
        imageR= cv2.remap(imageR, Right_Stereo_Map_x,   Right_Stereo_Map_y,cv2.INTER_LANCZOS4,cv2.BORDER_CONSTANT,0)
        cv2.imshow('right',imageR)
        #cv2.imshow('right',imageR)
    #global imagerr
    #imagerr=imageR


stereo = cv2.StereoBM.create(numDisparities=160, blockSize=15)

def disparity():
    global stereo
    global imagell
    global imagerr
    #cv2.imshow('lefty',imagell)
    #cv2.imshow('right',imagerr)
    
    disparity = stereo.compute(imagell,imagerr)
    cv2.imshow('disparity',disparity)
    #cv2.waitKey(0)


if __name__ == '__main__':
    l = Process(target=left)
    r = Process(target=right)
    l.start()
    r.start()
    l.join()
    r.join()

"""



thread1=Thread(target = left)#.start()
thread2=Thread(target = right)#.start()
thread3=Thread(target = disparity)
while True:


    thread1.start()
    thread2.start()


    thread1.join()
    thread2.join()



    thread3.start()
    thread3.join()
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
"""
#cv2.imshow('left',imagell)
#cv2.waitKey(0)
#while True:
    
    #cv2.imshow('left',imagell)
    #stereo = cv2.StereoBM.create(numDisparities=160, blockSize=15)
    #imagell=imagell.astype('uint8')
    #imagerr=imagerr.astype('uint8')
    #disparity = stereo.compute(imagell,imagerr)
    #cv2.imshow('disparity',disparity)