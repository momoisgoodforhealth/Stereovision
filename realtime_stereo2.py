import multiprocessing as mp
from multiprocessing import Pipe, Queue

import cv2, time
import numpy as np
import io
import socket
import struct
import time
import pickle
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


#UNDER WATER Q

Q=np.float32([[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00, -4.75951664e+02],
 [ 0.00000000e+00,  1.00000000e+00 , 0.00000000e+00, -3.22210562e+02],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  2.84444017e+02],
 [ 0.00000000e+00,  0.00000000e+00,  9.96655670e-03, -2.78952270e-01]]
)




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
    src='./videos/escapeL0011.mov'#"rtsp://10.6.10.161/live_stream"#'./videos/L0011.mov'#"rtsp://10.6.10.161/live_stream"#'./videos/L0007.mov'
    capture = cv2.VideoCapture(src)
    print("left fps="+str(capture.get(cv2.CAP_PROP_FPS)))
    print("l total frame="+str(capture.get(cv2.CAP_PROP_FRAME_COUNT)))

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
    src='./videos/escapeL0011.mov'#"rtsp://10.6.10.162/live_stream"#'./videos/R0011.mov'#"rtsp://10.6.10.162/live_stream"#'./videos/R0007.mov'
    capture = cv2.VideoCapture(src)
    print("right fps="+str(capture.get(cv2.CAP_PROP_FPS)))
    print("r total frame="+str(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
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



#def point_cloud(points):



def sockett(conn):
    TCP_IP = '127.0.0.1'
    TCP_PORT = 1234

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # establishing a tcp connection
    sock.bind((TCP_IP, TCP_PORT))
    sock.listen(5)
    print("sock listen pass")


    while True:
        (client_socket, client_address) = sock.accept() # wait for client
        print ('client accepted')
        print (str(client_address))
        frame=conn.recv()
        print(frame.shape)
        #plt.imshow(frame)
        #plt.savefig('disparity_image.png')
        connection = client_socket.makefile('wb')
        image_bytes = cv2.imencode('.jpg', frame)[1].tobytes()

        #filee = open("disparity_image.png", "rb")
        #imgData = filee.read()
        print(len(image_bytes))
        client_socket.send(image_bytes)
        print('sent!')        
        client_socket.close()

        #frame = cv2.imencode('.jpg', frame)[1]
        #cv2.imshow('disparity_image.jpg',frame)


def yolo(framee,distancee):
    INPUT_WIDTH = 640
    INPUT_HEIGHT = 640
    SCORE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.45
    CONFIDENCE_THRESHOLD = 0.45
    distanceee=0
    # Text parameters.
    FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.3
    THICKNESS = 1

    # Colors
    BLACK  = (0,0,0)
    BLUE   = (255,178,50)
    YELLOW = (0,255,255)
    RED = (0,0,255)
    bx,by=0,0
    def draw_label(input_image, label, left, top):
        """Draw text onto image at location."""
        
        # Get text size.
        text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
        dim, baseline = text_size[0], text_size[1]
        # Use text size to create a BLACK rectangle. 
        cv2.rectangle(input_image, (left, top), (left + dim[0], top + dim[1] + baseline), BLACK, cv2.FILLED);
        # Display text inside the rectangle.
        cv2.putText(input_image, label, (left, top + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)
    


    def pre_process(input_image, net):
        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], 1, crop=False)
        #net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        #net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        # Sets the input to the network.
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers.
        output_layers = net.getUnconnectedOutLayersNames()
        outputs = net.forward(output_layers)
        # print(outputs[0].shape)

        return outputs
    

    def post_process(input_image, outputs):
        # Lists to hold respective values while unwrapping.
        class_ids = []
        confidences = []
        boxes = []
        global bx
        global by
        # Rows.
        rows = outputs[0].shape[1]

        image_height, image_width = input_image.shape[:2]

        # Resizing factor.
        x_factor = image_width / INPUT_WIDTH
        y_factor =  image_height / INPUT_HEIGHT

        # Iterate through 25200 detections.
        for r in range(rows):
            row = outputs[0][0][r]
            confidence = row[4]

            # Discard bad detections and continue.
            if confidence >= CONFIDENCE_THRESHOLD:
                classes_scores = row[5:]

                # Get the index of max class score.
                class_id = np.argmax(classes_scores)

                #  Continue if the class score is above threshold.
                if (classes_scores[class_id] > SCORE_THRESHOLD):
                    confidences.append(confidence)
                    class_ids.append(class_id)

                    cx, cy, w, h = row[0], row[1], row[2], row[3]

                    left = int((cx - w/2) * x_factor)
                    top = int((cy - h/2) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                
                    box = np.array([left, top, width, height])
                    boxes.append(box)

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 3*THICKNESS)
            bx=(left+left+width)/2
            by=(top+top+ height)/2
            print("center = "+str(bx)+","+str(by))
            cv2.putText(input_image,str(distanceee[ int(by),int(bx)])+"mm", (int(bx), int(by)), font, 0.4, (0, 0, 255), 2) 
            #    cv2.putText(input_image,"mm", (bx, by),FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)
            label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
            draw_label(input_image, label, left, top)

        return input_image

    classesFile = "./yolo/mbari-mb-benthic-33k.names"
    classes = None    
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
        

    # Load image.
    

    # Give the weight files to the model and load the network using them.
    modelWeights = "./yolo/models/mbari-mb-benthic-33k.onnx"
    net = cv2.dnn.readNet(modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    # Process image.
    while True:
        frame=framee.recv()
        distanceee=distancee.recv()
        detections = pre_process(frame, net)
        img = post_process(frame.copy(), detections)

        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        #label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        #print(label)
        #cv2.putText(img, label, (20, 40), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)

        cv2.imshow('Output', img)
        #cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    print('Starting video stream')
    connL1, connL2 = Pipe()
    connR1, connR2 = Pipe()
    #connSR, connSS = Pipe()
    connY1, connY2 = Pipe()
    connY11, connY22 = Pipe()

    qL=mp.Queue
    qR=mp.Queue
    
    capture_processL = mp.Process(target=capture_framesL, args=(connL2,qL,))
    capture_processR = mp.Process(target=capture_framesR, args=(connR2,qR,))
    #Ds = mp.Process(target=disparity, args=(connL1,connR1,))
    #disp=mp.Process(target=disparity, args=(connL1,connR1,))
    #soc=mp.Process(target=sockett, args=(connSR,))
    yolo_process = mp.Process(target=yolo, args=(connY1,connY11,))

    capture_processL.start()
    capture_processR.start()
    yolo_process.start()
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
    #soc.start()
    ctr=0
    arrayarea=[]
    global areaflag
    areaflag=False
    global areatrig
    areatrig=False
    global area
    area=0
    index=0
    i=0
    while True:
        frameL=connL1.recv()
        frameR=connR1.recv()
        framecopy=frameL
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
        #stereo.setPreFilterSize(preFilterSize)
        #stereo.setPreFilterCap(preFilterCap)
        #stereo.setTextureThreshold(textureThreshold)
        #stereo.setUniquenessRatio(uniquenessRatio)
        #stereo.setSpeckleRange(speckleRange)
        #stereo.setSpeckleWindowSize(speckleWindowSize)
        #stereo.setDisp12MaxDiff(disp12MaxDiff)
        stereo.setMinDisparity(minDisparity)
        disparity = stereo.compute(frameL,frameR)
        #disparity = (disparity/16.0 - minDisparity)/ 2
        
        dispcolor=disparity.astype('uint8')
        points=cv2.reprojectImageTo3D(disparity, Q)
        distance = ((baseline * focal_length) / disparity)*50

        if i==10:
            connY2.send(framecopy)
            connY22.send(distance)
            i=0
        i=i+1
        
        def mouse_click(event, x, y, flags, param): 
            global msx,msy
            if event == cv2.EVENT_LBUTTONDOWN:
                msx,msy=x,y
                
                cv2.putText(dispcolor, str((x,y))+"="+str(disparity[y,x]), (x, y), font, 0.4, (0, 0, 255), 2) 
                cv2.putText(dispcolor, str(points[y, x]), (x, y-14), font, 0.4, (0, 0, 255), 2) 
                cv2.putText(dispcolor, str(distance[y,x])+"mm", (x, y-28), font, 0.4, (0, 0, 255), 2) 
                cv2.imshow("dis", dispcolor)
        dispcolor=cv2.applyColorMap(dispcolor,cv2.COLORMAP_OCEAN)
        #if ctr==10:
        #connSS.send(dispcolor)
        #    ctr=0
        ctr=ctr+1
        cv2.putText(dispcolor, str((msx,msy))+"="+str(disparity[msy,msx]), (msx,msy), font, 0.4, (0, 0, 255), 2) 
        cv2.putText(dispcolor, str(points[msy, msx]), (msx, msy-14), font, 0.4, (0, 0, 255), 2) 
        cv2.putText(dispcolor, str(distance[msy,msx])+"mm", (msx, msy-28), font, 0.4, (0, 0, 255), 2) 
        if areaflag: cv2.putText(dispcolor,"Area="+str(area),(20, 30), font, 0.4, (0, 0, 255), 2)
        if areatrig: cv2.putText(dispcolor,"area trigger",(25, 35), font, 0.4, (0, 0, 255), 2)
        cv2.imshow('dis',dispcolor)  
        areatrig=False

        cv2.setMouseCallback("dis", mouse_click)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break  
        if cv2.waitKey(33) == ord('a'):
            arrayarea.append(points[msy, msx])
            areatrig=True
            if len(arrayarea)==4:
                area=(1/2)*abs(((arrayarea[0][0]*arrayarea[1][1])+(arrayarea[1][0]*arrayarea[2][1])+(arrayarea[2][0]*arrayarea[3][1]))-((arrayarea[0][1]*arrayarea[1][0])+(arrayarea[1][1]*arrayarea[2][0])+(arrayarea[2][1]*arrayarea[3][0])))
                print(area)
                arrayarea=[]
                areaflag=True
            #print(arrayarea)
        #cv2.waitKey(0)   

    capture_processL.join()
    capture_processR.join()
    yolo_process.join()