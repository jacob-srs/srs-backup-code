# -*- coding: utf-8 -*-
# @Time    : 2019/7/3 11:44
# @Author  : DengHong
# @Email   : deng.hong@seiriosai.com
# @File    : Checkpoint.py
# @Software: PyCharm

import cv2
import numpy as np
import time


class Camera():

    def __init__(self):#初始化，打开摄像头
        self.cap = cv2.VideoCapture(0)
        # self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.cap.set(3, 3040)  # width
        self.cap.set(4, 1520)  # height        
        # self.cap.set(3, 1520)  # width
        # self.cap.set(4, 1520)  # height
        # self.cap.set(3, 2560)  # width  1280 x 2
        # self.cap.set(4, 960)  # height 
        # self.cap.set(3, 3840)  # width  1280 x 2
        # self.cap.set(4, 1080)  # height 

        self.save_path = r"C:\Users\Administrator\Desktop\20200506_dual_cam\calib_img"

        self.ret = None
        self.shame = None

    def play_single(self):

        while(True):
            self.ret, self.shame = self.cap.read()  # 读取摄像头的图像
            print(self.shame.shape)
            # cv2.namedWindow('Together',0)
            cv2.imshow('Together', self.shame)  # 显示摄像头原像 (行数 row height，列数 col width)
            if cv2.waitKey(1) == ord('s'):  # 按s键，截图保存
                break
            if cv2.waitKey(1) == 27:
                break

    def play(self):
        image_count = 0
        while(True):
            self.ret,self.shame = self.cap.read()  #读取摄像头的图像
            # shame_tmp_720h = self.shame[400:1120, :]
            # shame_tmp_720h_2560w = cv2.resize(shame_tmp_720h, (2560, 720))
            # print(self.shame.shape, shame_tmp_720h.shape, shame_tmp_720h_2560w.shape)
            print(self.shame.shape)
            # cv2.namedWindow('Together', 0)
            cv2.imshow('Together', self.shame)      #显示摄像头原像
            # cv2.namedWindow('Together_720h', 0)
            # cv2.imshow('Together_720h', shame_tmp_720h)
            # cv2.imshow('Together_720h_2560w', shame_tmp_720h_2560w)

            # cv2.imshow('Left',self.show_left())    #显示左边摄像头
            # cv2.imshow('Right',self.show_right())  #显示右边摄像头
            if cv2.waitKey(1) == ord('s'):         #按s键，截图保存
                s = time.strftime('%H-%M-%S', time.localtime()) #提取当前时间
                print(s)
                # cv2.imwrite(r'D:\Data\20200427\1-0-%s.jpg' % s, self.shame)
                # cv2.imwrite(self.save_path + r'\%d-%s.jpg' % (image_count, s), self.shame)
                cv2.imwrite(self.save_path + r'\%d-%s.jpg' % (image_count, s), self.shame)  # 保存 crop + resize 后的图像 shame_tmp_720h_2560w
                # cv2.imwrite(r'D:\Data\20200427\1-0-%s.jpg' % s, self.show_left())  #以时间命名文件名
                # cv2.imwrite(r'D:\Data\20200427\2-0-%s.jpg' % s, self.show_right())
                image_count += 1
            if cv2.waitKey(1) == 27:
                break
#获取左边摄像头的视频帧信息，从摄像头输出的图像中剥离出来
    def show_left(self):
        self.shame1 = np.zeros((1520, 1520, 3), np.uint8)
        self.shame1 = self.shame[0:1520,0:1520,:]
        # for i in range(0, 1080):
        #     for j in range(0, 1980):
        #         self.shame1[i, j, :] = self.shame[0:1080,0:1980,:]
        return self.shame1
# 获取右边摄像头的视频帧信息，从摄像头输出的图像中剥离出来
    def show_right(self):
        self.shame2 = np.zeros((1520,1520, 3), np.uint8)
        self.shame2 = self.shame[0:1520, 1520:3040, :]
        # for i in range(0, 240):
        #     for j in range(0, 320):
        #         self.shame2[i, j, :] = self.shame[359-i,639-j,:]
        return self.shame2
#结束
    def cameraexit(self):
        self.cap.release()
        cv2.destroyAllWindows()

    # def seveMedia(self):
    #     self.out = cv2.VideoWriter('output.avi', self.fourcc, self.cap.get(5), (self.cap.get(3), self.cap.get(4)))



if __name__ == '__main__':

    camera = Camera()
    camera.play()
    # camera.play_single()
    camera.cameraexit()
