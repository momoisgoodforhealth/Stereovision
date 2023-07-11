import cv2
assert cv2.__version__[0] >= '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
from tqdm import tqdm 
import os
import glob

# Set the path to the images captured by the left and right cameras
pathL = "./LU/"
pathR = "./RU/"
 
# Termination criteria for refining the detected corners
#subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 150, 0.001)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW 
 
objp = np.zeros((10*10,3), np.float32)
objp[:,:2] = np.mgrid[0:10,0:10].T.reshape(-1,2)*25.4 # 25.4 mm (1 inch) size of square

img_ptsL = []
img_ptsR = []
obj_pts = []
for i in tqdm(range(1,8)):
  imgL = cv2.imread(pathL+"l%d.png"%i)
  imgR = cv2.imread(pathR+"r%d.png"%i)
  imgL_gray = cv2.imread(pathL+"l%d.png"%i,0)
  imgR_gray = cv2.imread(pathR+"r%d.png"%i,0)  
 
  outputL = imgL.copy()
  outputR = imgR.copy()
 
  retR, cornersR =  cv2.findChessboardCorners(outputR,(10,10),None)
  retL, cornersL = cv2.findChessboardCorners(outputL,(10,10),None)
 
  if retR and retL:
    obj_pts.append(objp)
    cv2.cornerSubPix(imgR_gray,cornersR,(11,11),(-1,-1),criteria)
    cv2.cornerSubPix(imgL_gray,cornersL,(11,11),(-1,-1),criteria)
    cv2.drawChessboardCorners(outputR,(10,10),cornersR,retR)
    cv2.drawChessboardCorners(outputL,(10,10),cornersL,retL)
    cv2.imshow('cornersR',outputR)
    cv2.imshow('cornersL',outputL)
    cv2.waitKey(0) 
 
    img_ptsL.append(cornersL)
    img_ptsR.append(cornersR)


#print(obj_pts)
N_OK = len(obj_pts) 
K1 = np.zeros((3, 3))
D1 = np.zeros((4, 1))
K2 = np.zeros((3, 3))
D2 = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

# Calibrating left camera
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(obj_pts,img_ptsL,imgL_gray.shape[::-1],None,None)
print("retL="+str(retL))
#cv2.fisheye.calibrate(obj_pts,img_ptsL,imgL_gray.shape[::-1],K1,D1,rvecs,tvecs,calibration_flags,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
  

hL,wL= imgL_gray.shape[:2]
new_mtxL, roiL= cv2.getOptimalNewCameraMatrix(mtxL,distL,(wL,hL),1,(wL,hL))

dst = cv2.undistort(imgL_gray, mtxL, distL, None, new_mtxL)
cv2.imwrite('cakib.png',dst)

# Calibrating right camera
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(obj_pts,img_ptsR,imgR_gray.shape[::-1],None,None)

print("retR="+str(retR))
#cv2.fisheye.calibrate(obj_pts,img_ptsR,imgR_gray.shape[::-1],K2,D2,rvecs,tvecs,calibration_flags,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))

#print(cv2.fisheye.stereoCalibrate(obj_pts, img_ptsL, img_ptsR, K1,D1,K2,D2, new_mtxL, distL, new_mtxR, distR, imgL_gray.shape[::-1], criteria_stereo, flags))

hR,wR= imgR_gray.shape[:2]
new_mtxR, roiR= cv2.getOptimalNewCameraMatrix(mtxR,distR,(wR,hR),1,(wR,hR))


dst = cv2.undistort(imgR_gray, mtxR, distR, None, new_mtxR)
cv2.imwrite('cakibR.png',dst)

flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
# Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
# Hence intrinsic parameters are the same 
 
criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
 
 
# This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(obj_pts, img_ptsL, img_ptsR, new_mtxL, distL, new_mtxR, distR, imgL_gray.shape[::-1], criteria_stereo, flags)
print("retS="+str(retS))
#print(cv2.fisheye.stereoCalibrate(obj_pts, img_ptsL, img_ptsR, K1,D1,K2,D2, new_mtxL, distL, new_mtxR, distR, imgL_gray.shape[::-1], criteria_stereo, flags))
rectify_scale= 1


rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR= cv2.stereoRectify(new_mtxL, distL, new_mtxR, distR, imgL_gray.shape[::-1], Rot, Trns, rectify_scale,(0,0))
print("rect_l="+str(rect_l))
print("rect_r="+str(rect_r))
#rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR= cv2.fisheye.stereoRectify(K1,D1,K2,D2,new_mtxL, distL, new_mtxR, distR, imgL_gray.shape[::-1], Rot, Trns, rectify_scale,(0,0))
print("For Q..")
print(Q)


Left_Stereo_Map= cv2.initUndistortRectifyMap(new_mtxL, distL, rect_l, proj_mat_l,
                                             imgL_gray.shape[::-1], cv2.CV_16SC2)
Right_Stereo_Map= cv2.initUndistortRectifyMap(new_mtxR, distR, rect_r, proj_mat_r,
                                              imgR_gray.shape[::-1], cv2.CV_16SC2)
 
print("Saving parameters ......")
cv_file = cv2.FileStorage("improved_params3.xml", cv2.FILE_STORAGE_WRITE)
cv_file.write("Left_Stereo_Map_x",Left_Stereo_Map[0])
cv_file.write("Left_Stereo_Map_y",Left_Stereo_Map[1])
cv_file.write("Right_Stereo_Map_x",Right_Stereo_Map[0])
cv_file.write("Right_Stereo_Map_y",Right_Stereo_Map[1])
cv_file.release()
