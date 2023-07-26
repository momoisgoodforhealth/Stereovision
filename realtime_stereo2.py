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


baseline=98.5
focal_length= 2.83711978e+02#4.77424789e+02
msx,msy=0,0

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]


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

font = cv2.FONT_HERSHEY_DUPLEX

    #frame=conn.recv()




def capture_framesL(connection,queue):
    scale_percent = 50
    width = int(3840 * scale_percent / 100)
    height = int(2160 * scale_percent / 100)
    dim = (width, height)
    src='./videos/L0007.mov'#"rtsp://10.6.10.161/live_stream"#'./videos/L0007.mov'
    capture = cv2.VideoCapture(src)
    print("left fps="+str(capture.get(cv2.CAP_PROP_FPS)))
    capture.set(cv2.CAP_PROP_BUFFERSIZE,10)
    cv2.namedWindow("framel", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("framel", 480, 420)

    ctr = 0

    while True:
        # Ensure camera is connected
        if capture.isOpened():
            (status, frame) = capture.read()
            
            # Ensure valid frame
            if status:
               
                frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
                frame=frame[190:890, 560:1360]
                frame= cv2.remap(frame, Left_Stereo_Map_x,  Left_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
                cv2.imshow('framel', frame)
                ctr += 1
                if ctr == 9:
                    connection.send(frame)
                    ctr = 0
                
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

def capture_framesR(connection,queue):
    scale_percent = 50
    width = int(3840 * scale_percent / 100)
    height = int(2160 * scale_percent / 100)
    dim = (width, height)
    src='./videos/R0007.mov'#"rtsp://10.6.10.162/live_stream"#'./videos/R0007.mov'
    capture = cv2.VideoCapture(src)
    print("right fps="+str(capture.get(cv2.CAP_PROP_FPS)))
    capture.set(cv2.CAP_PROP_BUFFERSIZE,10)
    cv2.namedWindow("framer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("framer", 480, 420)  
    # FPS = 1/X, X = desired FPS
    FPS = 1/30
    FPS_MS = int(FPS * 1000)
    ctr = 0

    while True:
        # Ensure camera is connected
        if capture.isOpened():
            (status, frame) = capture.read()
            
            # Ensure valid frame
            if status:
               
                frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
                frame=frame[190:890, 560:1360]
                frame= cv2.remap(frame, Right_Stereo_Map_x,   Right_Stereo_Map_y,cv2.INTER_LANCZOS4,cv2.BORDER_CONSTANT,0)
              
                cv2.imshow('framer', frame)
                ctr += 1
                if ctr == 9:
                    connection.send(frame)
                    ctr = 0
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


def disparity(conL,conR):
    stereo = cv2.StereoBM.create(numDisparities=160, blockSize=15)
    framesL=[]
    framesR=[]
    i=0
    while True:
        frameL=conL.recv()
        frameR=conR.recv()
        #cv2.imshow('framlel',frameL)
        framesL.append(frameL)
        framesR.append(frameR)
        #cv2.imshow('recL',frameL)
        #cv2.imshow('recR',frameR)
        fraL=framesL[i]
        fraR=framesR[i]
        fraL=cv2.cvtColor(fraL, cv2.COLOR_BGR2GRAY)
        fraR=cv2.cvtColor(fraR, cv2.COLOR_BGR2GRAY)
        stereo = cv2.StereoBM.create(numDisparities=160, blockSize=15)
        disparity = stereo.compute(fraL,fraR)
        dispcolor=disparity.astype('uint8')

        points=cv2.reprojectImageTo3D(disparity, Q)
        distance = (baseline * focal_length) / disparity
        dispcolor=cv2.applyColorMap(dispcolor,cv2.COLORMAP_OCEAN)
        

        cv2.imshow('dis',dispcolor)  


        if cv2.waitKey(25) & 0xFF == ord('q'):
            break  
        #disparity = stereo.compute(frameL,frameR)
        #cv2.imshow('dis',disparity)

def sockett(conn):
    i=0
    TCP_IP = '127.0.0.1'
    TCP_PORT = 1234

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # establishing a tcp connection
    sock.bind((TCP_IP, TCP_PORT))
    sock.listen(5)
    print("sock listen pass")

    (client_socket, client_address) = sock.accept() # wait for client
    print ('client accepted')
    print (str(client_address))
    while True:
        frame=conn.recv()

        #print ('socket frame recieve')
        print(frame.shape)
        #by=bytearray(frame)
        #if i==5:
        frame = cPickle.dumps(frame)
        size = len(frame)
        print('packet size: '+str(size))
        p = struct.pack('I', size)
        frame = p + frame
        client_socket.sendall(frame)
        #    i=0
        #i=i+1
        #plt.imshow(frame)
        #plt.savefig('disparity_image.jpg')
        #frame = cv2.imencode('.jpg', frame)[1]
        #cv2.imshow('disparity_image.jpg',frame)
        #if cv2.waitKey(25) & 0xFF == ord('q'):
        #    break  

        #by=bytearray(frame)
        #print('byte size: '+str(by))
        #data=frame
        #frame = cPickle.dumps(frame)
        #size = len(frame)

        #print("frame size = "+ str(size))
        #client_socket.sendall(by)
        #frame = cPickle.dumps(frame)
        #size = len(frame)
        #print("frame size = "+ str(size))
        #p = struct.pack('I', size)
        #frame = p + frame
        #client_socket.sendall(frame)


if __name__ == '__main__':
    print('Starting video stream')
    connL1, connL2 = Pipe()
    connR1, connR2 = Pipe()
    connSR, connSS = Pipe()

    qL=mp.Queue
    qR=mp.Queue
    
    capture_processL = mp.Process(target=capture_framesL, args=(connL2,qL,))
    capture_processR = mp.Process(target=capture_framesR, args=(connR2,qR,))
    #Ds = mp.Process(target=disparity, args=(connL1,connR1,))
    disp=mp.Process(target=disparity, args=(connL1,connR1,))
    soc=mp.Process(target=sockett, args=(connSR,))

    capture_processL.start()
    capture_processR.start()
    #disp.start()
    
    #capture_processL.join()
    #capture_processR.join()    
    #disp.join()

    def nothing(x):
        pass


    cv2.namedWindow('dis',cv2.WINDOW_AUTOSIZE)
    #cv2.createTrackbar('numDisparities','dis',1,50,nothing)
    cv2.createTrackbar('blockSize','dis',5,50,nothing)
    #cv2.createTrackbar('preFilterType','dis',1,1,nothing)
    cv2.createTrackbar('preFilterSize','dis',2,25,nothing)
    #cv2.createTrackbar('preFilterCap','dis',5,62,nothing)
    #cv2.createTrackbar('textureThreshold','dis',10,100,nothing)
    cv2.createTrackbar('uniquenessRatio','dis',15,100,nothing)
    cv2.createTrackbar('speckleRange','dis',0,100,nothing)
    #cv2.createTrackbar('speckleWindowSize','dis',3,25,nothing)
    #cv2.createTrackbar('disp12MaxDiff','dis',5,25,nothing)
    cv2.createTrackbar('minDisparity','dis',0,25,nothing)

    stereo = cv2.StereoBM_create(numDisparities=160)
    soc.start()
    ctr=0
    while True:
        frameL=connL1.recv()
        frameR=connR1.recv()
        #frL=qL.get()
        #frR=qR.get()object has no attribute 'get'
        #cv2.imshow('recL',frL)
        #cv2.imshow('recR',frR)
    
        #cv2.imshow('recL',frameL)
        #cv2.imshow('recR',frameR)
        #if np.shape(frameL) == (): print("EMPTYL")
        #if np.shape(frameR) == (): print("EMPTYR")
        
        frameL=cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        frameR=cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
        #stereo = cv2.StereoBM.create(numDisparities=160, blockSize=15)
        

        


        #numDisparities = cv2.getTrackbarPos('numDisparities','dis')*16
        blockSize = cv2.getTrackbarPos('blockSize','dis')*2 + 5
        #preFilterType = cv2.getTrackbarPos('preFilterType','dis')
        preFilterSize = cv2.getTrackbarPos('preFilterSize','dis')*2 + 5
        #preFilterCap = cv2.getTrackbarPos('preFilterCap','dis')
        #textureThreshold = cv2.getTrackbarPos('textureThreshold','dis')
        uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio','dis')
        speckleRange = cv2.getTrackbarPos('speckleRange','dis')
        #speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize','dis')*2
        #disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff','dis')
        minDisparity = cv2.getTrackbarPos('minDisparity','dis')


        #stereo.setNumDisparities(numDisparities)
        stereo.setBlockSize(blockSize)
        #stereo.setPreFilterType(preFilterType)
        stereo.setPreFilterSize(preFilterSize)
        #stereo.setPreFilterCap(preFilterCap)
        #stereo.setTextureThreshold(textureThreshold)
        stereo.setUniquenessRatio(uniquenessRatio)
        stereo.setSpeckleRange(speckleRange)
        #stereo.setSpeckleWindowSize(speckleWindowSize)
        #stereo.setDisp12MaxDiff(disp12MaxDiff)
        stereo.setMinDisparity(minDisparity)
        disparity = stereo.compute(frameL,frameR)
        
        dispcolor=disparity.astype('uint8')
        points=cv2.reprojectImageTo3D(disparity, Q)
        distance = (baseline * focal_length) / disparity
        
        def mouse_click(event, x, y, flags, param): 
            global msx,msy
            if event == cv2.EVENT_LBUTTONDOWN:
                msx,msy=x,y
                
                cv2.putText(dispcolor, str((x,y))+"="+str(disparity[y,x]), (x, y), font, 0.4, (0, 0, 255), 2) 
                cv2.putText(dispcolor, str(points[y, x]), (x, y-14), font, 0.4, (0, 0, 255), 2) 
                cv2.putText(dispcolor, str(distance[y,x])+"mm", (x, y-28), font, 0.4, (0, 0, 255), 2) 
                cv2.imshow("dis", dispcolor)
        dispcolor=cv2.applyColorMap(dispcolor,cv2.COLORMAP_OCEAN)
        if ctr==20:
            connSS.send(dispcolor)
            ctr=0
        ctr=ctr+1
        cv2.putText(dispcolor, str((msx,msy))+"="+str(disparity[msy,msx]), (msx,msy), font, 0.4, (0, 0, 255), 2) 
        cv2.putText(dispcolor, str(points[msy, msx]), (msx, msy-14), font, 0.4, (0, 0, 255), 2) 
        cv2.putText(dispcolor, str(distance[msy,msx])+"mm", (msx, msy-28), font, 0.4, (0, 0, 255), 2) 
        cv2.imshow('dis',dispcolor)  

        cv2.setMouseCallback("dis", mouse_click)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break  
        #cv2.waitKey(0)   

    capture_processL.join()
    capture_processR.join()