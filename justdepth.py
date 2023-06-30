import numpy as np 
import cv2
from matplotlib import pyplot as plt
# Check for left and right camera IDs
# These values can change depending on the system
#CamL_id = 2 # Camera ID for left camera
#CamR_id = 0 # Camera ID for right camera
 
#CamL= cv2.VideoCapture(CamL_id)
#CamR= cv2.VideoCapture(CamR_id)
 
baseline=100
focal_length= 4.77424789e+02
#Q=[[ 1.00000000e+00  0.00000000e+00  0.00000000e+00 -1.24098755e+03]
#   [ 0.00000000e+00  1.00000000e+00  0.00000000e+00 -1.07122999e+03]
#   [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  4.77424789e+02]
#   [ 0.00000000e+00  0.00000000e+00  9.03260810e-03 -0.00000000e+00]]

Q=np.float32([[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.07952681e+01],
 [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00, -1.54576640e+03],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  9.16574783e+02],
 [ 0.00000000e+00,  0.00000000e+00, -2.16772271e-02,  0.00000000e+00]])

Q=np.float32([[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00, -2.67034456e+02],
 [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00, -1.03445789e+03],
 [ 0.00000000e+00,  0.00000000e+00 , 0.00000000e+00,  9.16584610e+02],
 [ 0.00000000e+00,  0.00000000e+00, -8.44895915e-04,  0.00000000e+00]])

Q=np.float32([[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00, -1.47927154e+03],
 [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00, -6.69240590e+02],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  3.60831146e+02],
 [ 0.00000000e+00,  0.00000000e+00,  1.27373443e-03, -0.00000000e+00]])


# Reading the mapping values for stereo image rectification
cv_file = cv2.FileStorage("improved_params3.xml", cv2.FILE_STORAGE_READ)
Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
cv_file.release()
 
def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


while True:
  # Capturing and storing left and right camera images
    imgL= cv2.imread(r'C:\Users\Benjamin\Documents\calibration\oneemlu.png', cv2.IMREAD_GRAYSCALE)#[60:900, 230:1500]
    imgR= cv2.imread(r'C:\Users\Benjamin\Documents\calibration\oneemru.png', cv2.IMREAD_GRAYSCALE)#[60:900, 230:1500]

    
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
    disparity = stereo.compute(imgL, imgR)#.astype(np.float32) / 16.0

    distance = (baseline * focal_length) / disparity

    dispL=disparity
    stereoR=cv2.ximgproc.createRightMatcher(stereo)

    dispR= stereoR.compute(imgR,imgL)
    dispL= np.int16(dispL)
    dispR= np.int16(dispR)

        # Using the WLS filter
    lmbda = 80000
    sigma = 1.8
    visual_multiplier = 1.0
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)


    filteredImg= wls_filter.filter(dispL,imgL,None,dispR)
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)
    #cv2.imshow('Disparity Map', filteredImg)
    disp= ((disparity.astype(np.float32)/ 16)-2)/128 # Calculation allowing us to have 0 for the most distant object able to detect

    ##    # Resize the image for faster executions
    #dispR= cv2.resize(disp,None,fx=0.7, fy=0.7, interpolation = cv2.INTER_AREA)

        # Filtering the Results with a closing filter
    kernel= np.ones((3,3),np.uint8)
    closing= cv2.morphologyEx(disp,cv2.MORPH_CLOSE, kernel) # Apply an morphological filter for closing little "black" holes in the picture(Remove noise) 

        # Colors map
    dispc= (closing-closing.min())*255
    dispC= dispc.astype(np.uint8)                                   # Convert the type of the matrix from float32 to uint8, this way you can show the results with the function cv2.imshow()
    disp_Color= cv2.applyColorMap(dispC,cv2.COLORMAP_OCEAN)         # Change the Color of the Picture into an Ocean Color_Map
    filt_Color= cv2.applyColorMap(filteredImg,cv2.COLORMAP_OCEAN) 
    print("filt color shape="+str(filt_Color.shape))
    print("dispC shape="+str(dispC.shape))
    print("distance shape="+str(distance.shape))
        # Show the result for the Depth_image
        #cv2.imshow('Disparity', disp)
        #cv2.imshow('Closing',closing)
        #cv2.imshow('Color Depth',disp_Color)



    points=cv2.reprojectImageTo3D(dispC, Q)
    #file1 = open("points.txt","w")
    #for x in points:
    #    print()
    #    for y in x:
    #        for z in y:
                
    #file1.close()
    print(points.shape)
    print(points)


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



    print('generating 3d point cloud...',)
    points=cv2.reprojectImageTo3D(dispC, Q)
    print(points.shape)
    #print(points)
    #points = cv2.reprojectImageTo3D(dispC, Q)
    colors = cv2.cvtColor(filteredImg, cv2.COLOR_BGR2RGB)
    mask = dispC > dispC.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = 'out.ply'
    #write_ply(out_fn, out_points, out_colors)
    #print('%s saved' % out_fn)



    """
    def coords_mouse_disp(event,x,y,flags,param):
        average=0
        Distance=0
        if event == cv2.EVENT_LBUTTONDBLCLK:n
            #print x,y,disp[y,x],filteredImg[y,x]
            average=0
            for u in range (-1,2):
                for v in range (-1,2):
                    average += disp[y+u,x+v]
            average=average/9
            Distance= -593.97*average**(3) + 1506.8*average**(2) - 1373.1*average + 522.06
            Distance= np.around(Distance*0.01,decimals=2)
            print('Distance: '+ str(Distance)+' m')
        return 'Distance: '+ str(Distance)+' m'
    """

    def mouse_click(event, x, y, flags, param): 
        # to check if left mouse 
        # button was clicked
        if event == cv2.EVENT_LBUTTONDOWN:
          
        # font for left click event
            
            font = cv2.FONT_HERSHEY_DUPLEX
            LB = 'Left Button'
          
        # display that left button 
        # was clicked.
            cv2.putText(filt_Color, str((x,y)), (x, y), font, 0.4, (0, 0, 255), 2) 
            cv2.putText(filt_Color, str(points[y, x]), (x, y-14), font, 0.4, (0, 0, 255), 2) 
            cv2.putText(filt_Color, str(distance[y,x])+"mm", (x, y-28), font, 0.4, (0, 0, 255), 2) 
            #cv2.resize(filt_Color, (500, 700))
            cv2.imshow('Filtered Color Depth', filt_Color)
            cv2.waitKey(0)

    
    """
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,500)
    fontScale              = 1
    fontColor              = (255,255,255)
    thickness              = 1
    lineType               = 2

    cv2.putText(filt_Color,coords_mouse_disp(), 
    bottomLeftCornerOfText, 
    font, 
    fontScale,
    fontColor,
    thickness,
    lineType)
    """
    cv2.imwrite('sc.png',filt_Color)
    cv2.rectangle(filt_Color, (30, 30), (300, 200), (0, 255, 0), 5)
    cv2.circle(filt_Color, (200, 200), 80, (255, 0, 0), 3)
    cv2.imwrite('sc2.png',filt_Color)
    cv2.imshow('Filtered Color Depth',filt_Color)
    
    cv2.setMouseCallback("Filtered Color Depth", mouse_click)

    #cv2.setMouseCallback("Filtered Color Depth",coords_mouse_disp,filt_Color)

        # Mouse click
    
        # End the Programme
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cv2.destroyAllWindows()
"""
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
"""
"""
def nothing(x):
    pass
 
cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp',600,600)
 
cv2.createTrackbar('numDisparities','disp',1,233,nothing)
cv2.createTrackbar('blockSize','disp',5,50,nothing)
cv2.createTrackbar('preFilterType','disp',1,1,nothing)
cv2.createTrackbar('preFilterSize','disp',2,25,nothing)
cv2.createTrackbar('preFilterCap','disp',5,62,nothing)
cv2.createTrackbar('textureThreshold','disp',10,100,nothing)
cv2.createTrackbar('uniquenessRatio','disp',15,100,nothing)
cv2.createTrackbar('speckleRange','disp',0,100,nothing)
cv2.createTrackbar('speckleWindowSize','disp',3,25,nothing)
cv2.createTrackbar('disp12MaxDiff','disp',5,25,nothing)
cv2.createTrackbar('minDisparity','disp',5,25,nothing)
 
# Creating an object of StereoBM algorithm
stereo = cv2.StereoSGBM_create()
 
while True:
 


 
    # Updating the parameters based on the trackbar positions
    numDisparities = cv2.getTrackbarPos('numDisparities','disp')*16
    blockSize = cv2.getTrackbarPos('blockSize','disp')*2 + 5
    #preFilterType = cv2.getTrackbarPos('preFilterType','disp')
    #preFilterSize = cv2.getTrackbarPos('preFilterSize','disp')*2 + 5
    preFilterCap = cv2.getTrackbarPos('preFilterCap','disp')
    #textureThreshold = cv2.getTrackbarPos('textureThreshold','disp')
    uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio','disp')
    speckleRange = cv2.getTrackbarPos('speckleRange','disp')
    speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize','disp')*2
    disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff','disp')
    minDisparity = cv2.getTrackbarPos('minDisparity','disp')
     
    # Setting the updated parameters before computing disparity map
    stereo.setNumDisparities(numDisparities)
    stereo.setBlockSize(blockSize)
    #stereo.setPreFilterType(preFilterType)
    #stereo.setPreFilterSize(preFilterSize)
    stereo.setPreFilterCap(preFilterCap)
    #stereo.setTextureThreshold(textureThreshold)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setSpeckleRange(speckleRange)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setDisp12MaxDiff(disp12MaxDiff)
    stereo.setMinDisparity(minDisparity)
 
    # Calculating disparity using the StereoBM algorithm
    disparity = stereo.compute(Left_nice,Right_nice)
    # NOTE: Code returns a 16bit signed single channel image,
    # CV_16S containing a disparity map scaled by 16. Hence it 
    # is essential to convert it to CV_32F and scale it down 16 times.
 
    # Converting to float32 
    disparity = disparity.astype(np.float32)
 
    # Scaling down the disparity values and normalizing them 
    disparity = (disparity/16.0 - minDisparity)/numDisparities
 
    # Displaying the disparity map
    cv2.imshow("disp",disparity)
 
    # Close window using esc key
    if cv2.waitKey(1) == 27:
      break
   
  #else:
    #CamL= cv2.VideoCapture(CamL_id)
    #CamR= cv2.VideoCapture(CamR_id)
"""