import cv2

cap = cv2.VideoCapture('rtsp://localhost:8554')
#cap.open('https://videos3.earthcam.com/fecnetwork/9974.flv/chunklist_w1421640637.m3u8')

while (True):
    _, frame = cap.read()
    cv2.imshow("camCapture", frame)
    cv2.waitKey(1)