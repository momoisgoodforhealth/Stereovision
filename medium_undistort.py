# You should replace these 3 lines with the output in calibration step


import numpy as np
import cv2
import sys
import os
import glob

""" LEFT
DIM=(1920, 1080)
K=np.array([[386.86272580887766, 0.0, 961.3496948184803], [0.0, 386.4098425816344, 507.1367485438685], [0.0, 0.0, 1.0]])
D=np.array([[0.025075441107852738], [-0.026474731140917737], [0.02878644374021592], [-0.011533604444589816]])
"""

""" RIGHT
DIM=(1920, 1080)
K=np.array([[386.49404392457035, 0.0, 950.7834022296813], [0.0, 386.17955232335163, 534.966578693153], [0.0, 0.0, 1.0]])
D=np.array([[0.027946380281796915], [-0.02052588351194629], [0.015438289890456498], [-0.005880227411641979]])
"""

""" LEFT V2
DIM=(1920, 1080)
K=np.array([[390.04427555700215, 0.0, 961.6136195957941], [0.0, 389.91220362222515, 508.6686936437803], [0.0, 0.0, 1.0]])
D=np.array([[0.025635339869543264], [-0.037264630055349614], [0.03804086063442726], [-0.01399852912413158]])
"""

""" RIGHT V2
DIM=(1920, 1080)
K=np.array([[386.386322313026, 0.0, 950.8192461891825], [0.0, 386.0537491228412, 534.9458738253207], [0.0, 0.0, 1.0]])
D=np.array([[0.028665799058537656], [-0.022179141581777077], [0.017364406587187752], [-0.006659607916889329]])
"""


"""LEFT V3
DIM=(1920, 1080)
K=np.array([[391.3848450284422, 0.0, 962.3775216439046], [0.0, 391.0778105961325, 508.34751216462564], [0.0, 0.0, 1.0]])
D=np.array([[0.006236998786723406], [0.005462819175278735], [-0.003998179526054345], [0.0005439273824285558]])
"""

#RIGHT V3
DIM=(1920, 1080)
KR=np.array([[384.67485064332175, 0.0, 950.8595665068705], [0.0, 384.8128042580504, 539.3097801677955], [0.0, 0.0, 1.0]])
DR=np.array([[0.025630777892378356], [-0.017527660784374195], [0.008867163745251873], [-0.001989176145429497]])

DIM=(1920, 1080)
KL=np.array([[391.3848450284422, 0.0, 962.3775216439046], [0.0, 391.0778105961325, 508.34751216462564], [0.0, 0.0, 1.0]])
DL=np.array([[0.006236998786723406], [0.005462819175278735], [-0.003998179526054345], [0.0005439273824285558]])
#def undistort(img_path):
#img_path=
images = glob.glob(r'C:\Users\Benjamin\Documents\Stereo-Vision\LL\*.png')
pathL = "./LL/"
pathR = "./RR/"
j=1
for i in (range(1,73)):
    imgL = cv2.imread(pathL+"chessboardd-L%d.png"%i)
    imgR = cv2.imread(pathR+"chessboardd-R%d.png"%i)
    #h,w = imgL.shape[:2]
    map1L, map2L = cv2.fisheye.initUndistortRectifyMap(KL, DL, np.eye(3), KL, DIM, cv2.CV_16SC2)
    undistorted_imgL = cv2.remap(imgL, map1L, map2L, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    map1R, map2R = cv2.fisheye.initUndistortRectifyMap(KR, DR, np.eye(3), KR, DIM, cv2.CV_16SC2)
    undistorted_imgR = cv2.remap(imgR, map1R, map2R, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow("L"+str(i), undistorted_imgL)
    cv2.imshow("R"+str(i), undistorted_imgR)
    if cv2.waitKey(0) & 0xFF == ord('s'):
        cv2.imwrite('l'+str(j)+'.png',undistorted_imgL)
        cv2.imwrite('r'+str(j)+'.png',undistorted_imgR)
        j=j+1
    else:
        print('Img not saved')
    cv2.destroyAllWindows()

"""
for fname in images:
    img = cv2.imread(fname)
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow("arm", undistorted_img)
    if cv2.waitKey(0) & 0xFF == ord('s'):
        cv2.imwrite('r'+str(i)+'.png',undistorted_img)
        i=i+1
    else:
        print('Img not saved')
    cv2.destroyAllWindows()
"""
"""
img = cv2.imread(os.path.join('plainR','board51.png'))
h,w = img.shape[:2]
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
cv2.imshow("arm", undistorted_img)
cv2.imwrite('armlu.png',undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
#if __name__ == '__main__':
 #   for p in sys.argv[1:]:
  #      undistort(p)