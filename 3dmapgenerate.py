import numpy as np 
import cv2
from matplotlib import pyplot as plt
# Check for left and right camera IDs
# These values can change depending on the system
#CamL_id = 2 # Camera ID for left camera
#CamR_id = 0 # Camera ID for right camera
 
#CamL= cv2.VideoCapture(CamL_id)
#CamR= cv2.VideoCapture(CamR_id)
 
# Reading the mapping values for stereo image rectification
cv_file = cv2.FileStorage("improved_params2.xml", cv2.FILE_STORAGE_READ)
Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
cv_file.release()
 




  # Capturing and storing left and right camera images
imgL= cv2.imread('onemLu.png', cv2.IMREAD_GRAYSCALE)
imgR= cv2.imread('onemRu.png', cv2.IMREAD_GRAYSCALE)

    
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
        

    # creates StereoBm object 
stereo = cv2.StereoSGBM_create(numDisparities =160,
                                blockSize =1)
    
    # computes disparity
disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
 

#print "\nGenerating the 3D map ..."
h, w = imgL.shape[:2]
focal_length = 0.8*w                          

    # Perspective transformation matrix
Q = np.float32([[1, 0, 0, -w/2.0],
                    [0,-1, 0,  h/2.0], 
                    [0, 0, 0, -focal_length], 
                    [0, 0, 1, 0]])

points_3D = cv2.reprojectImageTo3D(disparity, Q)
colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
mask_map = disparity > disparity.min()
output_points = points_3D[mask_map]
output_colors = colors[mask_map]

def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1,3), colors])

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

    with open(filename, 'w') as f:
        f.write(ply_header % dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')

output_file = 'output_file.ply'
#print "\nCreating the output file ...\n"
create_output(output_points, output_colors, output_file)
  
# displays image as grayscale and plotted
plt.imshow(disparity)
plt.show()

from open3d import *    


cloud = io.read_point_cloud("output_file.ply") # Read point cloud
visualization.draw_geometries([cloud])    # Visualize point cloud 