import cv2
import numpy as np
from matplotlib import pyplot as plt



baseline=100
focal_length= 2.83711978e+02#4.77424789e+02

Q=np.float32([[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00, -3.34334817e+02],
 [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00, -3.14079409e+02],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  2.83711978e+02],
 [ 0.00000000e+00,  0.00000000e+00,  1.01527526e-02, -2.22518594e-01]])

font = cv2.FONT_HERSHEY_DUPLEX

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
capL = cv2.VideoCapture('./videos/L0011.mov')
capR = cv2.VideoCapture('videos/R0011.mov') 

npzfile = np.load('./calibration_data/{}p/stereo_camera_calibration.npz'.format(700))
cv_file = npzfile#cv2.FileStorage("improved_params3.xml", cv2.FILE_STORAGE_READ)
Left_Stereo_Map_x = cv_file['leftMapX']
Left_Stereo_Map_y = cv_file['leftMapY']
Right_Stereo_Map_x = cv_file['rightMapX']
Right_Stereo_Map_y = cv_file['rightMapY']

# Check if camera opened successfully
if (capL.isOpened()== False): 
  print("Error opening video stream or file")
 
# Read until video is completed
while(capL.isOpened()):
  # Capture frame-by-frame
  ret, frameL = capL.read()

  scale_percent = 50 # percent of original size
  width = int(frameL.shape[1] * scale_percent / 100)
  height = int(frameL.shape[0] * scale_percent / 100)


  dim = (width, height)
    
# resize image
  frameL = cv2.resize(frameL, dim, interpolation = cv2.INTER_AREA)
  frameLr = cv2.resize(frameL, dim, interpolation = cv2.INTER_AREA)

  frameL=frameL[190:890, 560:1360]
  frameLr=frameLr[190:890, 560:1360]
  frameL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
  

  ret, frameR = capR.read()

  frameR = cv2.resize(frameR, dim, interpolation = cv2.INTER_AREA)
  frameR=frameR[190:890, 560:1360]
  frameR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)


  
  frameL= cv2.remap(frameL, Left_Stereo_Map_x,  Left_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
  frameLr= cv2.remap(frameLr, Left_Stereo_Map_x,  Left_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)      
  frameR= cv2.remap(frameR, Right_Stereo_Map_x,   Right_Stereo_Map_y,cv2.INTER_LANCZOS4,cv2.BORDER_CONSTANT,0)
  

  if ret == True:

    # Display the resulting frame
    stereo = cv2.StereoBM.create(numDisparities=160, blockSize=15)
    disparity = stereo.compute(frameL,frameR)
    dis=disparity.astype('uint8')
    stereoR=cv2.ximgproc.createRightMatcher(stereo)

    dispR= stereoR.compute(frameR,frameL)
    dispL= np.int16(disparity)
    dispR= np.int16(dispR)

    lmbda = 7000
    sigma = 1.8
    visual_multiplier = 1.0
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    filteredImg= wls_filter.filter(dispL,frameL,None,dispR)
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)
    filt_Color= cv2.applyColorMap(filteredImg,cv2.COLORMAP_OCEAN) 

    points=cv2.reprojectImageTo3D(disparity, Q)
    distance = (baseline * focal_length) / disparity
    def mouse_click(event, x, y, flags, param): 
      if event == cv2.EVENT_LBUTTONDOWN:

          cv2.putText(filt_Color, str((x,y)), (x, y), font, 0.4, (0, 0, 255), 2) 
          cv2.putText(filt_Color, str(points[y, x]), (x, y-14), font, 0.4, (0, 0, 255), 2) 
          cv2.putText(filt_Color, str(distance[y,x])+"mm", (x, y-28), font, 0.4, (0, 0, 255), 2) 
          cv2.imshow("Filtered", filt_Color)

    
    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video", 480, 420)
    cv2.imshow("Video",frameLr)
    cv2.putText(filt_Color, str((452,116)), (452,116), font, 0.4, (0, 0, 255), 2) 
    cv2.putText(filt_Color, str(points[116, 452]), (452, 116-14), font, 0.4, (0, 0, 255), 2) 
    cv2.putText(filt_Color, str(distance[116,452])+"mm", (452, 116-28), font, 0.4, (0, 0, 255), 2) 
    

    cv2.imshow("Filtered", filt_Color)
    dispcolor=cv2.applyColorMap(dis,cv2.COLORMAP_OCEAN)
    cv2.namedWindow("disparity", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("disparity", 480, 420)
    cv2.imshow("disparity", dispcolor)
    cv2.setMouseCallback("Filtered", mouse_click)
    #plt.imshow(disparity,'gray')
    #plt.show()
    #cv2.imshow('Frame',frameL)
 
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
capL.release()
capR.release()
 
# Closes all the frames
cv2.destroyAllWindows()