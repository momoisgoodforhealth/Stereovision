# You should replace these 3 lines with the output in calibration step


import numpy as np
import cv2
import sys
import os

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

DIM=(1920, 1080)
K=np.array([[386.86272580887766, 0.0, 961.3496948184803], [0.0, 386.4098425816344, 507.1367485438685], [0.0, 0.0, 1.0]])
D=np.array([[0.025075441107852738], [-0.026474731140917737], [0.02878644374021592], [-0.011533604444589816]])
#def undistort(img_path):
#img_path=
img = cv2.imread(os.path.join('plainR','board51.png'))
h,w = img.shape[:2]
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
cv2.imshow("arm", undistorted_img)
cv2.imwrite('armlu.png',undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#if __name__ == '__main__':
 #   for p in sys.argv[1:]:
  #      undistort(p)