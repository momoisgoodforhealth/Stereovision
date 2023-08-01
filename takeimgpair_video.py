import multiprocessing as mp
from multiprocessing import Pipe, Queue

import cv2, time
import numpy as np
import io
import socket
import struct
import time
import pickle as cPickle
import zlib
from matplotlib import pyplot as plt







font = cv2.FONT_HERSHEY_DUPLEX

    #frame=conn.recv()




def capture_framesL(connection):
    scale_percent = 50
    width = int(3840 * scale_percent / 100)
    height = int(2160 * scale_percent / 100)
    dim = (width, height)
    src='./videos/calibL001.mov'#"rtsp://10.6.10.161/live_stream"#'./videos/L0007.mov'
    capture = cv2.VideoCapture(src)
    print("left fps="+str(capture.get(cv2.CAP_PROP_FPS)))
    print("l total frame="+str(capture.get(cv2.CAP_PROP_FRAME_COUNT)))

    capture.set(cv2.CAP_PROP_BUFFERSIZE,10)
    cv2.namedWindow("framel", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("framel", 480, 420)

    while True:
        # Ensure camera is connected
        if capture.isOpened():
            (status, frame) = capture.read()
            
            # Ensure valid frame
            if status:
               
                frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
                framec=frame[190:890, 560:1360]
                cv2.imshow('framel', framec)

                connection.send(frame)
                #queue.put(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    connection.send(None)
                    break
                    
            else:
                print("rip status")
                connection.send(None)
                capture = cv2.VideoCapture(src)
                capture.set(cv2.CAP_PROP_BUFFERSIZE,10)
                continue
                
        else:
            print("rip open")
            #time.sleep(FPS)

    capture.release()
    connection.send(None)
    cv2.destroyAllWindows()

def capture_framesR(connection):
    scale_percent = 50
    width = int(3840 * scale_percent / 100)
    height = int(2160 * scale_percent / 100)
    dim = (width, height)
    src='./videos/calibR001.mov'#"rtsp://10.6.10.162/live_stream"#'./videos/R0007.mov'
    capture = cv2.VideoCapture(src)
    print("right fps="+str(capture.get(cv2.CAP_PROP_FPS)))
    print("r total frame="+str(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    capture.set(cv2.CAP_PROP_BUFFERSIZE,10)
    cv2.namedWindow("framer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("framer", 480, 420)  


    while True:
        # Ensure camera is connected
        if capture.isOpened():
            (status, frame) = capture.read()
            
            # Ensure valid frame
            if status:
               
                frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
                framec=frame[190:890, 560:1360]
              
                cv2.imshow('framer', framec)

                connection.send(frame)

                #queue.put(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    connection.send(None)
                    break
                    
            else:
                print("rip status")
                capture = cv2.VideoCapture(src)
                capture.set(cv2.CAP_PROP_BUFFERSIZE,10)
                connection.send(None)
                continue
                
            #time.sleep(FPS)
        else:
            print("rip open")

    capture.release()
    connection.send(None)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    print('Starting video stream')
    connL1, connL2 = Pipe()
    connR1, connR2 = Pipe()
    
    capture_processL = mp.Process(target=capture_framesL, args=(connL2,))
    capture_processR = mp.Process(target=capture_framesR, args=(connR2,))
    #Ds = mp.Process(target=disparity, args=(connL1,connR1,))

    capture_processL.start()
    capture_processR.start()
    dummy=cv2.imread('sc.png')
    cv2.namedWindow('button press',cv2.WINDOW_AUTOSIZE)
    blank_image = np.zeros((100,100,3), np.uint8)
    index=185
    while True:
        frameL=connL1.recv()
        frameR=connR1.recv()
       
        cv2.imshow('button press',blank_image)
        if cv2.waitKey(33) == ord('a'):
            cv2.imwrite("underL/"+str(index)+".png", frameL)
            cv2.imwrite("underR/"+str(index)+".png", frameR)
            print('frame saved')
            index=index+1
        


    capture_processL.join()
    capture_processR.join()