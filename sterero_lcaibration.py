
import os
import cv2
import numpy as np




# Global variables preset
total_photos = 30

# Camera resolution
photo_width = 1920
photo_height = 1080

# Image resolution for processing
img_width = 1920
img_height = 1080
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
imageToDisp = './scenes/scene_1280x480_1.png'

# Calibration settings
CHECKERBOARD = (10,10)

subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW

objp = np.zeros( (CHECKERBOARD[0]*CHECKERBOARD[1], 1, 3) , np.float64)
objp[:,0, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)






processing_time01 = cv2.getTickCount()
objectPoints = None

rightImagePoints = None
rightCameraMatrix = None
rightDistortionCoefficients = None

leftImagePoints = None
leftCameraMatrix = None
leftDistortionCoefficients = None

rotationMatrix = None
translationVector = None

imageSize= (1920, 1080)

TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
OPTIMIZE_ALPHA = 0.25




npz_fileL = np.load('calib_fisheyeL.npz')
npz_fileR = np.load('calib_fisheyeL.npz')

list_of_vars = ['map1', 'map2', 'objpoints', 'imgpoints', 'camera_matrix', 'distortion_coeff']
print(sorted(npz_fileL.files))

if sorted(list_of_vars) == sorted(npz_fileL.files):
    print("Camera calibration data has been found in cache.")
    map1 = npz_fileL['map1']
    map2 = npz_fileL['map2']
    objectPoints = npz_fileL['objpoints']
    rightImagePoints = npz_fileL['imgpoints']
    rightCameraMatrix = npz_fileL['camera_matrix']
    rightDistortionCoefficients = npz_fileL['distortion_coeff']
    leftImagePoints = npz_fileL['imgpoints']
    leftCameraMatrix = npz_fileL['camera_matrix']
    leftDistortionCoefficients = npz_fileL['distortion_coeff']

print("Calibrating cameras together...")

leftImagePoints = np.asarray(leftImagePoints, dtype=np.float64)
rightImagePoints = np.asarray(rightImagePoints, dtype=np.float64)

# Stereo calibration
(RMS, _, _, _, _, rotationMatrix, translationVector) = cv2.fisheye.stereoCalibrate(
            objectPoints, leftImagePoints, rightImagePoints,
            leftCameraMatrix, leftDistortionCoefficients,
            rightCameraMatrix, rightDistortionCoefficients,
            imageSize, None, None,
            cv2.CALIB_FIX_INTRINSIC, TERMINATION_CRITERIA)
