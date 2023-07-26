# Copyright (C) 2021 Eugene a.k.a. Realizator, stereopi.com, virt2real team
#
# This file is part of StereoPi tutorial scripts.
#
# StereoPi tutorial is free software: you can redistribute it 
# and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the 
# License, or (at your option) any later version.
#
# StereoPi tutorial is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with StereoPi tutorial.  
# If not, see <http://www.gnu.org/licenses/>.
#
#          <><><> SPECIAL THANKS: <><><>
#
# Thanks to Adrian and http://pyimagesearch.com, as a lot of
# code in this tutorial was taken from his lessons.
#  
# Thanks to RPi-tankbot project: https://github.com/Kheiden/RPi-tankbot
#
# Thanks to rakali project: https://github.com/sthysel/rakali


import os
import cv2
import numpy as np




# Global variables preset


# Camera resolution
photo_width = 800#1920
photo_height = 700#1080

# Image resolution for processing
img_width = 800#1920
img_height = 700#1080
image_size = (img_width,img_height)

# Chessboard parameters
rows = 11
columns = 11
#square_size = 2.5

# Visualization options
drawCorners = True
showSingleCamUndistortionResults = True
showStereoRectificationResults = True
writeUdistortedImages = True
imageToDisp = r'C:\Users\Benjamin\Documents\calibration\onemR.png'
distorted= cv2.imread(r'C:\Users\Benjamin\Documents\calibration\board52.png')[190:890, 560:1360]

# Calibration settings
CHECKERBOARD = (10,10)

subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 150, 0.001)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW

objp = np.zeros( (CHECKERBOARD[0]*CHECKERBOARD[1], 1, 3) , np.float64)
objp[:,0, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)*25.4 # 25.4 mm (1 inch) size of square

_img_shape = None
objpointsLeft = [] # 3d point in real world space
imgpointsLeft = [] # 2d points in image plane.

objpointsRight = [] # 3d point in real world space
imgpointsRight = [] # 2d points in image plane.

if (drawCorners):
    print("You can press 'Q' to quit this script.")


# Main processing cycle
# We process all calibration images and fill up 'imgpointsLeft' and 'objpointsRight'
# arrays with found coordinates of the chessboard
total_photos = 65
photo_counter = 0
print ('Main cycle start')

while photo_counter != total_photos:
  
  print ('Import pair No ' + str(photo_counter))
  leftName = './underL/'+str(photo_counter)+'.png'
  rightName = './underR/'+str(photo_counter)+'.png'
  leftExists = os.path.isfile(leftName)
  rightExists = os.path.isfile(rightName)
  photo_counter = photo_counter + 1
  # If pair has no left or right image - exit
  if ((leftExists == False) or (rightExists == False)) and (leftExists != rightExists):
      print ("Pair No ", photo_counter, "has only one image! Left:", leftExists, " Right:", rightExists )
      continue 
  
  # If stereopair is complete - go to processing 
  if (leftExists and rightExists):
      imgL = cv2.imread(leftName,1)[190:890, 560:1360]
      loadedY, loadedX, clrs  =  imgL.shape
      grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
      gray_small_left = cv2.resize (grayL, (img_width,img_height), interpolation = cv2.INTER_AREA)
      imgR = cv2.imread(rightName,1)[190:890, 560:1360]
      grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
      gray_small_right = cv2.resize (grayR, (img_width,img_height), interpolation = cv2.INTER_AREA)
      
      # Find the chessboard corners
      retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
      retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
      
      # Draw images with corners found
      if (drawCorners):
          cv2.drawChessboardCorners(imgL, (10,10), cornersL, retL)
          cv2.imshow('Corners LEFT', imgL)
          cv2.drawChessboardCorners(imgR, (10,10), cornersR, retR)
          cv2.imshow('Corners RIGHT', imgR)
          if cv2.waitKey(0) & 0xFF == ord('s'):
                    #if key == ord("q"):
                    #    exit(0)

                # Here is a fix for the OpenCV bug, which is causing this error:
                # error:(-215:Assertion failed) fabs(norm_u1) > 0 in function 'InitExtrinsics'
                # It means corners are too close to the side of the image. Let's filter them out
                
                SayMore = True; #Should we print additional debug info?
                if ((retL == True) and (retR == True)):
                    minRx = cornersR[:,:,0].min()
                    maxRx = cornersR[:,:,0].max()
                    minRy = cornersR[:,:,1].min()
                    maxRy = cornersR[:,:,1].max()

                    minLx = cornersL[:,:,0].min()
                    maxLx = cornersL[:,:,0].max()          
                    minLy = cornersL[:,:,1].min()
                    maxLy = cornersL[:,:,1].max()          
                        
                    border_threshold_x = loadedX/7
                    border_threshold_y = loadedY/7
                    if (SayMore): 
                        print ("thr_X: ", border_threshold_x, "thr_Y:", border_threshold_y)
                    x_thresh_bad = False
                    if ((minRx<border_threshold_x) or (minLx<border_threshold_x)): # or (loadedX-maxRx < border_threshold_x) or (loadedX-maxLx < border_threshold_x)):
                        x_thresh_bad = True
                    y_thresh_bad = False
                    if ((minRy<border_threshold_y) or (minLy<border_threshold_y)): # or (loadedY-maxRy < border_threshold_y) or (loadedY-maxLy < border_threshold_y)):
                        y_thresh_bad = True
                    if (y_thresh_bad==True) or (x_thresh_bad==True):
                        if (SayMore):
                            print("Chessboard too close to the side!", "X thresh: ", x_thresh_bad, "Y thresh: ", y_thresh_bad)
                            print ("minRx: ", minRx, "maxRx: ", maxRx, " minLx: ", minLx, "maxLx:", maxLx)      
                            print ("minRy: ", minRy, "maxRy: ", maxRy, " minLy: ", minLy, "maxLy:", maxLy) 
                        else: 
                            print("Chessboard too close to the side! Image ignored")
                        retL = False
                        retR = False
                        continue

                # Here is our scaling trick! Hi res for calibration, low res for real work!
                # Scale corners X and Y to our working resolution
                if ((retL == True) and (retR == True)) and (img_height <= photo_height):
                    scale_ratio = img_height/photo_height
                    print ("Scale ratio: ", scale_ratio)
                    cornersL = cornersL*scale_ratio #cornersL/2.0
                    cornersR = cornersR*scale_ratio #cornersR/2.0
                elif (img_height > photo_height):
                    print ("Image resolution is higher than photo resolution, upscale needed. Please check your photo and image parameters!")
                    exit (0)
                
                # Refine corners and add to array for processing
                if ((retL == True) and (retR == True)):
                    objpointsLeft.append(objp)
                    cv2.cornerSubPix(gray_small_left,cornersL,(3,3),(-1,-1),subpix_criteria)
                    imgpointsLeft.append(cornersL)
                    objpointsRight.append(objp)
                    cv2.cornerSubPix(gray_small_right,cornersR,(3,3),(-1,-1),subpix_criteria)
                    imgpointsRight.append(cornersR)
                else:
                    print ("Pair No", photo_counter, "ignored, as no chessboard found" )
                    continue
          else:
            print('Image pair not Used')
          
       
print ('End cycle')


# This function calibrates (undistort) a single camera
def calibrate_one_camera (objpoints, imgpoints, right_or_left):
  
    # Opencv sample code uses the var 'grey' from the last opened picture
    N_OK = len(objpoints)
    DIM= (img_width, img_height)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    
    # Single camera calibration (undistortion)
    rms, camera_matrix, distortion_coeff, _, _ = \
        cv2.fisheye.calibrate(
            objpoints,
                imgpoints,
            grayL.shape[::-1],
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 150, 0.001)
        )
    
    print ("<><>  Individual RMS is ", rms, " <><>")
    # Let's rectify our results
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    
    undistorted_img = cv2.remap(distorted, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow('undistorted',undistorted_img)
    cv2.waitKey(0)

    
    # Now we'll write our results to the file for the future use
    if (os.path.isdir('./under_calibration_data/{}p'.format(img_height))==False):
        os.makedirs('./under_calibration_data/{}p'.format(img_height))
    np.savez('./under_calibration_data/{}p/camera_calibration_{}.npz'.format(img_height, right_or_left),
        map1=map1, map2=map2, objpoints=objpoints, imgpoints=imgpoints,
        camera_matrix=camera_matrix, distortion_coeff=distortion_coeff)
    return (True)


# Stereoscopic calibration
def calibrate_stereo_cameras(res_x=img_width, res_y=img_height):
    # We need a lot of variables to calibrate the stereo camera
    """
    Based on code from:
    https://gist.github.com/aarmea/629e59ac7b640a60340145809b1c9013
    """
    processing_time01 = cv2.getTickCount()
    objectPoints = None
    DIM= (img_width, img_height)
    rightImagePoints = None
    rightCameraMatrix = None
    rightDistortionCoefficients = None

    leftImagePoints = None
    leftCameraMatrix = None
    leftDistortionCoefficients = None

    rotationMatrix = None
    translationVector = None

    imageSize= (res_x, res_y)

    TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 150, 0.001)
    OPTIMIZE_ALPHA = 0.25

    try:
        npz_file = np.load('./under_calibration_data/{}p/stereo_camera_calibration.npz'.format(res_y))
    except:
        pass

    for cam_num in [0, 1]:
        right_or_left = ["_right" if cam_num==1 else "_left"][0]

        try:
            print ('./under_calibration_data/{}p/camera_calibration{}.npz'.format(res_y, right_or_left))
            npz_file = np.load('./under_calibration_data/{}p/camera_calibration{}.npz'.format(res_y, right_or_left))

            list_of_vars = ['map1', 'map2', 'objpoints', 'imgpoints', 'camera_matrix', 'distortion_coeff']
            print(sorted(npz_file.files)) 

            if sorted(list_of_vars) == sorted(npz_file.files):
                print("Camera calibration data has been found in cache.")
                map1 = npz_file['map1']
                map2 = npz_file['map2']
                objectPoints = npz_file['objpoints']
                if right_or_left == "_right":
                    rightImagePoints = npz_file['imgpoints']
                    rightCameraMatrix = npz_file['camera_matrix']
                    rightDistortionCoefficients = npz_file['distortion_coeff']
                if right_or_left == "_left":
                    leftImagePoints = npz_file['imgpoints']
                    leftCameraMatrix = npz_file['camera_matrix']
                    leftDistortionCoefficients = npz_file['distortion_coeff']
            else:
                print("Camera data file found but data corrupted.")
        except:
            #If the file doesn't exists 
            print("Camera calibration data not found in cache.")
            return False


    print("Calibrating cameras together...")

    leftImagePoints = np.asarray(leftImagePoints, dtype=np.float64)
    rightImagePoints = np.asarray(rightImagePoints, dtype=np.float64)

    # Stereo calibration

    CALIBRATE_FLAGS = (
    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
    #+ cv2.fisheye.CALIB_CHECK_COND
    + cv2.fisheye.CALIB_FIX_SKEW
) #cv2.CALIB_FIX_INTRINSIC
    
    (RMS, _, _, _, _, rotationMatrix, translationVector) = cv2.fisheye.stereoCalibrate(
            objectPoints, leftImagePoints, rightImagePoints,
            leftCameraMatrix, leftDistortionCoefficients,
            rightCameraMatrix, rightDistortionCoefficients,
            DIM, None, None,
            cv2.CALIB_FIX_INTRINSIC, TERMINATION_CRITERIA)
    # Print RMS result (for calibration quality estimation)
    print ("<><>  Stereo RMS is ", RMS, " <><>")
    print("Rectifying cameras...")
    R1 = np.zeros([3,3])
    R2 = np.zeros([3,3])
    P1 = np.zeros([3,4])
    P2 = np.zeros([3,4])
    Q = np.zeros([4,4])
    
    # Rectify calibration results
    (leftRectification, rightRectification, leftProjection, rightProjection,
            dispartityToDepthMap) = cv2.fisheye.stereoRectify(
                    leftCameraMatrix, leftDistortionCoefficients,
                    rightCameraMatrix, rightDistortionCoefficients,
                    DIM, rotationMatrix, translationVector,
                    0, R2, P1, P2, Q,
                    cv2.CALIB_ZERO_DISPARITY, (0,0) , 0, 0)
    print("left rect: ")
    print(leftRectification)
    print("right rect: ")
    print(rightRectification)
    print("For Q..")
    print(dispartityToDepthMap)

    # Saving calibration results for the future use
    print("Saving calibration...")
    leftMapX, leftMapY = cv2.fisheye.initUndistortRectifyMap(
            leftCameraMatrix, leftDistortionCoefficients, leftRectification,
            leftProjection, DIM, cv2.CV_16SC2)
    rightMapX, rightMapY = cv2.fisheye.initUndistortRectifyMap(
            rightCameraMatrix, rightDistortionCoefficients, rightRectification,
            rightProjection, DIM, cv2.CV_16SC2)

    np.savez_compressed('./calibration_data/{}p/stereo_camera_calibration.npz'.format(res_y), imageSize=DIM,
            leftMapX=leftMapX, leftMapY=leftMapY,
            rightMapX=rightMapX, rightMapY=rightMapY, dispartityToDepthMap = dispartityToDepthMap)
    return True

# Now we have all we need to do stereoscopic fisheye calibration
# Let's calibrate each camera, and than calibrate them together
print ("Left camera calibration...")
result = calibrate_one_camera(objpointsLeft, imgpointsLeft, 'left')
print ("Right camera calibration...")
result = calibrate_one_camera(objpointsRight, imgpointsRight, 'right')
print ("Stereoscopic calibration...")
result = calibrate_stereo_cameras()
print ("Calibration complete!")

# The following code just shows you calibration results
# 
#

if (showSingleCamUndistortionResults):

    w = 800    
    h = 700
    print("Undistorting picture with (width, height):", (w, h))
    try:
        npz_file = np.load('./under_calibration_data/{}p/camera_calibration{}.npz'.format(h, '_left'))
        if 'map1' and 'map2' in npz_file.files:
            #print("Camera calibration data has been found in cache.")
            map1 = npz_file['map1']
            map2 = npz_file['map2']
        else:
            print("Camera data file found but data corrupted.")
            exit(0)
    except:
        #print("Camera calibration data not found in cache, file " & './calibration_data/{}p/camera_calibration{}.npz'.format(h, left))
        exit(0)

    # We didn't load a new image from file, but use last image loaded while calibration
    undistorted_left = cv2.remap(distorted, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    #h, w = imgR.shape[:2]
    print("Undistorting picture with (width, height):", (w, h))
    try:
        npz_file = np.load('./under_calibration_data/{}p/camera_calibration{}.npz'.format(h, '_right'))
        if 'map1' and 'map2' in npz_file.files:
            #print("Camera calibration data has been found in cache.")
            map1 = npz_file['map1']
            map2 = npz_file['map2']
        else:
            print("Camera data file found but data corrupted.")
            exit(0)
    except:
        print("Camera calibration RIGHT data not found in cache.")
        exit(0)

    undistorted_right = cv2.remap(distorted, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    cv2.imshow('Left UNDISTORTED', undistorted_left)
    cv2.imshow('Right UNDISTORTED', undistorted_right)
    cv2.waitKey(0)
    if (writeUdistortedImages):
        cv2.imwrite("undistorted_left.jpg",undistorted_left)
        cv2.imwrite("undistorted_right.jpg",undistorted_right)




if (showStereoRectificationResults):
    # lets rectify pair and look at the result
    # NOTICE: we use 320x240 as a working resolution, regardless of
    # calibration images resolution. So 320x240 parameters are hardcoded
    # now. 
    try:
        npzfile = np.load('./under_calibration_data/{}p/stereo_camera_calibration.npz'.format(1080))
    except:
       # print("Camera calibration data not found in cache, file " & './calibration_data/{}p/stereo_camera_calibration.npz'.format(240))
        exit(0)
    
    leftMapX = npzfile['leftMapX']
    leftMapY = npzfile['leftMapY']
    rightMapX = npzfile['rightMapX']
    rightMapY = npzfile['rightMapY']

    #read image to undistort
    photo_width = 1920#800#1920
    photo_height = 1080#700#1080
    image_width = 1920#800
    image_height = 1080#700
    image_size = (image_width,image_height)

    # Rectifying left and right images
    imgL = cv2.remap(distorted, leftMapX, leftMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    imgR = cv2.remap(distorted, rightMapX, rightMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    cv2.imshow('Left  STEREO CALIBRATED', imgL)
    cv2.imshow('Right STEREO CALIBRATED', imgR)
    cv2.imwrite("rectifyed_left.jpg",imgL)
    cv2.imwrite("rectifyed_right.jpg",imgR)
    cv2.waitKey(0)