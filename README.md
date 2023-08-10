## Depth Map Generation for Fisheye Stereo Camera with Distance Estimation

### To Run Depth Map on Camera Stream / Pre-recorded Videos
**realtime_stereo2.py**

Uses multiprocessing/Processes and Pipes for intercommunication,

Individual videos/camera feed are handled by seperate Processes.

Socket communication with Unity is also handled by a seperate Process. These can be 'turned off' by commenting out the respective Pipe.send command at line 605. (connSS.send(dispcolor))

YOLOv5 Object Detection uses MBARI benethic model. This also handled by a seperate Process, can be 'turned off' by commenting out line 567 (connPC2.send(disparity))

Point Cloud Generation is also another seperate Process. Comment out  "connPC2.send(disparity), connPC22.send(colors)" to turn off. (line 568)

*Changing video source:* Change src variable in 'def capture_framesL' and 'def capture_framesR' to livestream or video file. Livestream link = "rtsp://10.6.10.161/live_stream", video link ='./videos/L0008.mov'

*mouse click handling:* by def mouse_click


### Stereo Fisheye Calibration Script:
 fisheye_calibration.py
### To get a Disparity Map on Input Image Pairs With Distance on Point Click:
 fisheye_depth.py
### Disparity on videos with Distance on Point Click:
vid_disparity.py
### Disparity Map using juststereo.py calibration files:
justdepth.py

## Calibration Folder is repoed here:
https://github.com/shovcha/CalibrationStuff



## Images for Final Calibraation
underL, underR

## Calibration Data
is inside calibration_data/700p/


# These sources really helped

https://github.com/LearnTechWithUs/Stereo-Vision
https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0
https://learnopencv.com/depth-perception-using-stereo-camera-python-c/
https://learnopencv.com/making-a-low-cost-stereo-camera-using-opencv/
