import cv2
import io
import socket
import struct
import time
import pickle
import zlib

TCP_IP = '127.0.0.1'
TCP_PORT = 1234

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#client_socket.connect(('127.0.0.1', 1234))
sock.bind((TCP_IP, TCP_PORT))
sock.listen(5)
print("sock listen pass")
while True:
    (client_socket, client_address) = sock.accept() # wait for client
    print ('client accepted')
    print (str(client_address)) 
    connection = client_socket.makefile('wb')
    #src='./videos/L0007.mov'
    filee = open("trucL.png", "rb")
    imgData = filee.read()
    print(len(imgData))
    time.sleep(5)
    client_socket.send(imgData)
    print('sent!')
    #print(imgData)
    client_socket.close()

"""
cam = cv2.VideoCapture(src)

cam.set(3, 320);
cam.set(4, 240);

img_counter = 0

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

while True:
    ret, frame = cam.read()
    cv2.imshow('frame',frame)
    cv2.waitKey(0)
    result, frame = cv2.imencode('.jpg', frame, encode_param)
#    data = zlib.compress(pickle.dumps(frame, 0))
    data = pickle.dumps(frame, 0)
    size = len(data)


    print("{}: {}".format(img_counter, size))
    client_socket.sendall(struct.pack(">L", size) + data)
    img_counter += 1

cam.release()

"""