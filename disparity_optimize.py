import numpy as np 
import cv2
from matplotlib import pyplot as plt
from open3d import *

# Check for left and right camera IDs
# These values can change depending on the system
#CamL_id = 2 # Camera ID for left camera
#CamR_id = 0 # Camera ID for right camera
 
#CamL= cv2.VideoCapture(CamL_id)
#CamR= cv2.VideoCapture(CamR_id)
 
baseline=100
focal_length= 2.85650723e+02#4.77424789e+02

Q=np.float32([[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00, -4.39381522e+02],
 [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00, -3.12274867e+02],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  2.85650723e+02],
 [ 0.00000000e+00,  0.00000000e+00,  7.76600100e-03, -3.72237318e-02]])




# Reading the mapping values for stereo image rectification
npzfile = np.load('./calibration_data/{}p/stereo_camera_calibration.npz'.format(700))
cv_file = npzfile#cv2.FileStorage("improved_params3.xml", cv2.FILE_STORAGE_READ)
Left_Stereo_Map_x = cv_file['leftMapX']
Left_Stereo_Map_y = cv_file['leftMapY']
Right_Stereo_Map_x = cv_file['rightMapX']
Right_Stereo_Map_y = cv_file['rightMapY']
 
def write_ply(fn, verts, colors):
    ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

def display_pc():
    cloud = io.read_point_cloud("fisheyePoint.ply")
    visualization.draw_geometries([cloud])


while True:
  # Capturing and storing left and right camera images
    rawimgL=cv2.imread(r'C:\Users\Benjamin\Documents\calibration\oneeml.png')[200:850, 600:1400]
    rawimgR=cv2.imread(r'C:\Users\Benjamin\Documents\calibration\oneemr.png')[200:850, 600:1400]
    imgL= cv2.imread(r'C:\Users\Benjamin\Documents\calibration\dive11L.png',cv2.IMREAD_GRAYSCALE)[200:850, 600:1400]
    imgR= cv2.imread(r'C:\Users\Benjamin\Documents\calibration\dive11R.png',cv2.IMREAD_GRAYSCALE)[200:850, 600:1400]

    Left_nice_raw= cv2.remap(rawimgL,
                Left_Stereo_Map_x,
                Left_Stereo_Map_y,
                cv2.INTER_LANCZOS4,
                cv2.BORDER_CONSTANT,
                0)
    
    Right_nice_raw= cv2.remap(rawimgR,
                Right_Stereo_Map_x,
                Right_Stereo_Map_y,
                cv2.INTER_LANCZOS4,
                cv2.BORDER_CONSTANT,
                0)
    Left_nice= cv2.remap(imgL,
                Left_Stereo_Map_x,
                Left_Stereo_Map_y,
                cv2.INTER_LANCZOS4,
                cv2.BORDER_CONSTANT,
                0)
        
        # Applying stereo image rectification on the right image
    Right_nice= cv2.remap(imgR,
                Right_Stereo_Map_x,
                Right_Stereo_Map_y,
                cv2.INTER_LANCZOS4,
                cv2.BORDER_CONSTANT,
                0)
    
    Left_nice = cv2.GaussianBlur(Left_nice,(3,3),cv2.BORDER_DEFAULT)
    Right_nice = cv2.GaussianBlur(Right_nice,(3,3),cv2.BORDER_DEFAULT)
    Left_niceC = cv2.Canny(image=Left_nice, threshold1=100, threshold2=200) 
    Right_niceC = cv2.Canny(image=Right_nice, threshold1=100, threshold2=200) 
    
    Left_nice = cv2.addWeighted(Left_nice, 0.5, Left_niceC, 0.7, 0)
    Right_nice = cv2.addWeighted(Right_nice, 0.5, Right_niceC, 0.7, 0)
    cv2.imshow('raw left nice',Left_nice_raw)
    cv2.imshow('left nice',Left_nice)
    cv2.imshow('right nice',Right_nice)
    cv2.waitKey(0)


    stereo = cv2.StereoSGBM_create(numDisparities =160, blockSize =15)
    
    # computes disparity

  
    disparity = stereo.compute(Left_nice, Right_nice)#.astype(np.float32) / 16.0

    #distance = (baseline * focal_length) / disparity

    #dispL=disparity
    #cv2.imshow('disparity',disparity)
    plt.imshow(disparity)
    plt.show()
    #cv2.waitKey(0)