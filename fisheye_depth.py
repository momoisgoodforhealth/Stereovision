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
focal_length= 2.83711978e+02#4.77424789e+02

Q=np.float32([[ 1.00000000e+00,  0.00000000e+00 , 0.00000000e+00 ,-6.43447407e+02],
 [ 0.00000000e+00 , 1.00000000e+00,  0.00000000e+00, -6.47744417e+02],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  3.06869370e+02],
 [ 0.00000000e+00,  0.00000000e+00 ,-4.21595460e-03,  0.00000000e+00]])



Q=np.float32([[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00, -4.39381522e+02],
 [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00, -3.12274867e+02],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  2.85650723e+02],
 [ 0.00000000e+00,  0.00000000e+00,  7.76600100e-03, -3.72237318e-02]])

Q=np.float32([[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00, -3.34334817e+02],
 [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00, -3.14079409e+02],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  2.83711978e+02],
 [ 0.00000000e+00,  0.00000000e+00,  1.01527526e-02, -2.22518594e-01]])




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
    rawimgL=cv2.imread(r'C:\Users\Benjamin\Documents\calibration\handL.png')[190:890, 560:1360]
    rawimgR=cv2.imread(r'C:\Users\Benjamin\Documents\calibration\handR.png')[190:890, 560:1360]
    imgL= cv2.imread(r'C:\Users\Benjamin\Documents\calibration\handL.png', cv2.IMREAD_GRAYSCALE)[190:890, 560:1360]
    imgR= cv2.imread(r'C:\Users\Benjamin\Documents\calibration\handR.png', cv2.IMREAD_GRAYSCALE)[190:890, 560:1360]

    Left_nice_raw= cv2.remap(rawimgL,
                Left_Stereo_Map_x,
                Left_Stereo_Map_y,
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
    
    """
    x1=250
    x2=450
    y1=250
    y2=330

    blankl = np.zeros((700,800), dtype=np.uint8)
    blankr = np.zeros((700,800), dtype=np.uint8)
    cropl=Left_nice[x1:x2, y1:y2]
    cropr=Right_nice[x1:x2, y1:y2]
    blankl[x1:x2, y1:y2]=cropl
    blankr[x1:x2, y1:y2]=cropr

    Left_nice=blankl
    Right_nice=blankr
    cv2.imshow('blankl',blankl)
    cv2.imshow('blankr',blankr)
    
    """

    cv2.waitKey(0)
    
    cv2.imshow('left nice',Left_nice)
    cv2.imshow('right nice',Right_nice)
    cv2.waitKey(0)
        
    kernel = np.ones((10,10),np.uint8)
    closing1 = cv2.morphologyEx(Left_nice, cv2.MORPH_CLOSE, kernel)
    closing2 = cv2.morphologyEx(Right_nice, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('morphologyexL',closing1)
    cv2.imshow('morphologyexR',closing2)
    cv2.waitKey(0)
    # creates StereoBm object 

    
    stereo =cv2.StereoSGBM.create(numDisparities=160, blockSize=1)
    stereo2 =cv2.StereoSGBM.create(numDisparities=160, blockSize=16)
    stereo3 =cv2.StereoBM.create(numDisparities=160, blockSize=5)
    disparity2 = stereo2.compute(Left_nice, Right_nice).astype(np.float32) / 16.0
    disparity3 = stereo3.compute(Left_nice, Right_nice).astype(np.float32) / 16.0
    dis2=disparity2.astype('uint8')
    dispcolor2=cv2.applyColorMap(dis2,cv2.COLORMAP_OCEAN)  



    dis3=disparity3.astype('uint8')
    dispcolor3=cv2.applyColorMap(dis3,cv2.COLORMAP_OCEAN)  


    # computes disparity
    disparity = stereo.compute(Left_nice, Right_nice).astype(np.float32) / 16.0
    dis=disparity.astype('uint8')
    dispcolor=cv2.applyColorMap(dis,cv2.COLORMAP_OCEAN)

    #plt.show()
    #np.savez('disp.npz',disparity=disparity)

    

    dispL=disparity

    #cv2.waitKey(0)
    stereoR=cv2.ximgproc.createRightMatcher(stereo)

    dispR= stereoR.compute(Right_nice,Left_nice)
    dispL= np.int16(dispL)
    dispR= np.int16(dispR)

    # Using the WLS filter
    lmbda = 80000
    sigma = 1.8
    visual_multiplier = 1.0
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)


    filteredImg= wls_filter.filter(dispL,Left_nice,None,dispR)
    
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)
    #plt.imshow(filteredImg)
   # plt.show()
    #cv2.imshow('Disparity Map', filteredImg)
    disp= disparity#.astype(np.float32)#/ 16)-2)/128 # Calculation allowing us to have 0 for the most distant object able to detect

    ##    # Resize the image for faster executions
    #dispR= cv2.resize(disp,None,fx=0.7, fy=0.7, interpolation = cv2.INTER_AREA)

        # Filtering the Results with a closing filter
    kernel= np.ones((3,3),np.uint8)
    closing= cv2.morphologyEx(disp,cv2.MORPH_CLOSE, kernel) # Apply an morphological filter for closing little "black" holes in the picture(Remove noise) 
    closing = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    distance = (baseline * focal_length) / dis
    #plt.imshow(closing)
    #plt.show()
        # Colors map
    closing=closing.astype(np.uint8)
    #dispcolor=cv2.applyColorMap(closing,cv2.COLORMAP_OCEAN)
    dispc= (closing-closing.min())*255
    dispC= dispc.astype(np.uint8)       
    #plt.imshow(closing)     
                           # Convert the type of the matrix from float32 to uint8, this way you can show the results with the function cv2.imshow()
    disp_Color= cv2.applyColorMap(dispC,cv2.COLORMAP_RAINBOW)         # Change the Color of the Picture into an Ocean Color_Map
    filt_Color= cv2.applyColorMap(filteredImg,cv2.COLORMAP_OCEAN) 
    #print("filt color shape="+str(filt_Color.shape))
    print("dispC shape="+str(dispC.shape))
    print("distance shape="+str(distance.shape))
        # Show the result for the Depth_image
        #cv2.imshow('Disparity', disp)
        #cv2.imshow('Closing',closing)
    cv2.imshow('Color Depth',disp_Color)
    Q = np.float32([[1, 0, 0, 0],
               [0, -1, 0, 0],
               [0, 0, focal_length * 0.05, 0],
               [0, 0, 0, 1]])
    points=cv2.reprojectImageTo3D(disparity2, Q)
    #file1 = open("points.txt","w")
    #for x in points:
    #    print()
    #    for y in x:
    #        for z in y:
                
    #file1.close()
    print(points.shape)
    print(points)



    print('generating 3d point cloud...')
    points=cv2.reprojectImageTo3D(disparity2, Q)
    print(points.shape)
    #print(points)
    #points = cv2.reprojectImageTo3D(dispC, Q)
    colors = cv2.cvtColor(Left_nice_raw, cv2.COLOR_BGR2RGB)
    mask = disparity > disparity.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = 'fish.ply'
    print(out_colors.shape)
    print(out_points.shape)
    write_ply(out_fn, out_points, out_colors)
    print('%s saved' % out_fn)



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
            cv2.putText(dispcolor, str((x,y)), (x, y), font, 0.4, (0, 0, 255), 2) 
            cv2.putText(dispcolor, str(points[y, x]), (x, y-14), font, 0.4, (0, 0, 255), 2) 
            cv2.putText(dispcolor, str(distance[y,x])+"mm", (x, y-28), font, 0.4, (0, 0, 255), 2) 
            #cv2.resize(filt_Color, (500, 700))
            cv2.imshow('dis', dispcolor)
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
    #cv2.imwrite('sc.png',filt_Color)
    #cv2.rectangle(filt_Color, (30, 30), (300, 200), (0, 255, 0), 5)
    #cv2.circle(filt_Color, (200, 200), 80, (255, 0, 0), 3)
    #cv2.imwrite('sc2.png',filt_Color)

    #dispcolor[300:400,300:400] = (128,128,23)
    dispcolor=cv2.applyColorMap(dis,cv2.COLORMAP_OCEAN)
    cv2.imshow('dis',dispcolor)
    cv2.imshow('dis2',dispcolor2)
    cv2.imshow('dis3',dispcolor3)
    cv2.setMouseCallback('dis', mouse_click)
    #plt.imshow(filt_Color)
    plt.show()

    #cv2.setMouseCallback("Filtered Color Depth", mouse_click)

    #cv2.setMouseCallback("Filtered Color Depth",coords_mouse_disp,filt_Color)

        # Mouse click
    
        # End the Programme
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cv2.destroyAllWindows()
