'''
@Author: your name
@Date: 2019-11-18 10:22:05
@LastEditTime: 2019-12-07 10:56:44
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \V_vsCode\.vscode\20191118_dualCam\videoStart.py
'''
# -*- coding: utf-8 -*-
import cv2


class DualCamera(object):
    def __init__(self):
        super().__init__()

        self.cap = None

    def start_sync(self):
        self.cap = cv2.VideoCapture(0)
        ret, frame = self.cap.read()
        
        return ret, frame

    def stop_cam(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    d_camera = DualCamera()
    
    while True:
        ret, frame = d_camera.start_sync() 
        cv2.imshow("image", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    d_camera.stop_cam()
# 创建实例对象变量 cap，类 VideoCapture，打开一个视频捕获设备
# cap = cv.VideoCapture(0)

# while(True):

#     ret, frame = cap.read()

#     # Our operations on the frame come here
#     # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

#     # Display the resulting frame
#     cv.imshow('org', frame)
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break

# # When everything done, release the capture
# cap.release()
# cv.destroyAllWindows()
