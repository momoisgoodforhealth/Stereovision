# %%
import cv2
assert cv2.__version__[0] >= '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
import os
import glob
CHECKERBOARD = (10,10)
subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW)
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float64)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
_img_shape = None
objpoints= []
objpointsL= []
objpointsR= [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
imgpointsR=[]
imagesL = glob.glob(r'C:\Users\Benjamin\Documents\Stereo-Vision\cleanL\*.png')
imagesR = glob.glob(r'C:\Users\Benjamin\Documents\Stereo-Vision\cleanR\*.png')




for fname in imagesL:
    img = cv2.imread(fname)
    if _img_shape == None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    retL, cornersL = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    # If found, add object points, image points (after refining them)
    if retL == True:
        objpointsL.append(objp)
        cv2.cornerSubPix(gray,cornersL,(3,3),(-1,-1),subpix_criteria)
        imgpointsL.append(cornersL)

for fname in imagesR:
    img = cv2.imread(fname)
    if _img_shape == None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    retR, cornersR = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    # If found, add object points, image points (after refining them)
    if retR == True:
        objpointsR.append(objp)
        cv2.cornerSubPix(gray,cornersR,(3,3),(-1,-1),subpix_criteria)
        imgpointsR.append(cornersR)

N_OK = len(imgpointsL) #objpoints
K1 = np.zeros((3, 3))
K2 = np.zeros((3, 3))
D1 = np.zeros((4, 1))
D2 = np.zeros((4, 1))
rvecsL = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecsL = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
rvecsR = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecsR = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
R = np.zeros((1, 1, 3), dtype=np.float64)
T = np.zeros((1, 1, 3), dtype=np.float64)
#retL, K1, D1, rvecsL, tvecsL = 
rms, camera_matrixL, distortion_coeffL, _, _ = \
    cv2.fisheye.calibrate(
        objpointsL,
        imgpointsL,
        gray.shape[::-1],
        K1,
        D1,
        rvecsL,
        tvecsL,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
mapL1, mapL2 = cv2.fisheye.initUndistortRectifyMap(K1, D1, np.eye(3), K1, gray.shape[::-1], cv2.CV_16SC2)
np.savez('calib_fisheyeL.npz', map1=mapL1, map2=mapL2, objpoints=objpointsL, imgpoints=imgpointsL, camera_matrix=camera_matrixL, dist_coefs=distortion_coeffL)
print(len(objpointsL))
print(len(imgpointsL))
rms, camera_matrixR, distortion_coeffR, _, _ = \
    cv2.fisheye.calibrate(
        objpointsR,
        imgpointsR,
        gray.shape[::-1],
        K2,
        D2,
        rvecsR,
        tvecsR,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
print(len(objpointsR))
print(len(imgpointsR))
mapR1, mapR2 = cv2.fisheye.initUndistortRectifyMap(K2, D2, np.eye(3), K2, gray.shape[::-1], cv2.CV_16SC2)
np.savez('calib_fisheyeR.npz', map1=mapR1, map2=mapR2, objpoints=objpointsR, imgpoints=imgpointsR, camera_matrix=camera_matrixR, dist_coefs=distortion_coeffR)

print("image shape="+str(gray.shape[::-1]))

leftImagePoints = np.asarray(imgpointsL, dtype=np.float64)
rightImagePoints = np.asarray(imgpointsR, dtype=np.float64)

TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

print("K1=np.array(" + str(camera_matrixL.tolist()) + ")") # camera_matrix
print("D1=np.array(" + str(distortion_coeffL.tolist()) + ")") # dist_coefs
print("K2=np.array(" + str(camera_matrixR.tolist()) + ")") # camera_matrix
print("D2=np.array(" + str(distortion_coeffR.tolist()) + ")") # dist_coefs

board_area=CHECKERBOARD[0] * CHECKERBOARD[1]
objp = np.zeros((board_area, 1, 3), np.float64)
objp[:, 0, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = np.array([objp] * len(imgpointsL), dtype=np.float64)
#objpoints = np.reshape(objpoints, (N_OK, 1, board_area, 3))
#leftImagePoints = np.reshape(imgpointsL, (N_OK, 1, board_area, 2))
#rightImagePoints = np.reshape(imgpointsR, (N_OK, 1, board_area, 2))
objpointsL=np.array(objpointsL,dtype=np.float64)
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
CALIBRATE_FLAGS = flags #( cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW)

cv2.fisheye.stereoCalibrate(
                objpoints, leftImagePoints, rightImagePoints,
            camera_matrixL, distortion_coeffL,
            camera_matrixR, distortion_coeffR,
            (1920,1080), None, None,
            cv2.CALIB_FIX_INTRINSIC, TERMINATION_CRITERIA)
print("ok...")

'''
            objectPoints=objpointsR, imagePoints1= leftImagePoints,imagePoints2= rightImagePoints,
          #objectPoints=objpointsL, imagePoints1= imgpointsL,imagePoints2= imgpointsR,
          #  K1=K1,D1=D1,K2=K2,D2=D2,
            K1=camera_matrixL,D1=distortion_coeffL,
           K2=camera_matrixR, D2=distortion_coeffR,
            imageSize= (1920,1080), #R=R, T=T,
            flags=cv2.CALIB_FIX_INTRINSIC, criteria=TERMINATION_CRITERIA)'''
print("ok...")
# %%
"""

print("Found " + str(N_OK) + " valid images for calibration")
print("DIM=" + str(_img_shape[::-1]))
#print("\nRMS: " + str(rms))
print("K1=np.array(" + str(K1.tolist()) + ")") # camera_matrix
print("D1=np.array(" + str(D1.tolist()) + ")") # dist_coefs
print("K2=np.array(" + str(K1.tolist()) + ")") # camera_matrix
print("D2=np.array(" + str(D1.tolist()) + ")") # dist_coefs

error = 0
for i in range(len(objpointsL)):
    imgPoints2, _ = cv2.projectPoints(objpointsL[i], rvecsL[i], tvecsL[i], K1, D1)
    error += cv2.norm(imgpointsL[i], imgPoints2, cv2.NORM_L2) / len(imgpointsL)

print("\nTotal error(L): ", error / len(objpointsL))


error = 0
for i in range(len(objpointsR)):
    imgPoints2, _ = cv2.projectPoints(objpointsR[i], rvecsR[i], tvecsR[i], K2, D2)
    error += cv2.norm(imgpointsR[i], imgPoints2, cv2.NORM_L2) / len(imgpointsR)

print("\nTotal error(R): ", error / len(objpointsR))


flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
# Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
# Hence intrinsic parameters are the same 
 
criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

rms,_,_,_,_,_,_= \
objpoints=objpointsL+objpointsR
cv2.fisheye.stereoCalibrate(objpointsL,imgpointsL,imgpointsR,K1,D1,K2,D2,gray.shape[::-1])

"""


"""
print(    cv2.fisheye.stereoCalibrate(
        objpointsL,
        imgpointsL,
        imgpointsR, 
        K1,
        D1,
        K2,
        D2,
        gray.shape[::-1],R,T,
     #  rvecs,
     #   tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    ))

print("Found " + str(N_OK) + " valid images for calibration")
print("DIM=" + str(_img_shape[::-1]))
print("K1=np.array(" + str(K1.tolist()) + ")")
print("D1=np.array(" + str(D1.tolist()) + ")")
print("K1=np.array(" + str(K2.tolist()) + ")")
print("D1=np.array(" + str(D2.tolist()) + ")")

"""