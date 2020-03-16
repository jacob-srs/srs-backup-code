#
# 作者：jacob-srs
# 文件：学习 realsense 3
# 模块：
# 版本：v1.0
# 功能：保存图像
# 


import cv2
import numpy as np

import pyrealsense2 as rs


class Realsense():
    def __init__(self, resolution=(640, 480), fps=30):
        """ 可自定义相机分辨率 (col x row)
        
        intelRealsense fact sheet:
        - Depth Stream Output Resolution: Up to 1280 x 720
        - Depth Stream Output Frame Rate: Up to 90 fps
        - Minimum Depth Distance (Min-Z): 0.16 m
        - Maximum Range: Approx. 10 meters
        - RGB Sensor Resolution & Frame Rate: 1920 x 1080 at 30 fps
        """
        col = resolution[0]
        row = resolution[1]
        self.resolution = (col, row)
        self.fps = fps
        self.config = None
        self.pipeline = None
        # self.depth_frame = None
        # self.color_frame = None
       
        self._get_realsense_device(self.resolution)
        self._configure_stream(col, row, fps)
        self._start_pipeline()

    def _configure_stream(self, col, row, fps):
        """ 配置深度和彩色数据流 """
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, col, row, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, col, row, rs.format.bgr8, fps)

    def _start_pipeline(self):
        """ 创建 realsense 管道，简化使用者与设备的交互 """
        self.pipeline = rs.pipeline()
        self.pipeline.start()

    def _get_realsense_device(self, resolution):
        """ 查找连接的 realsense 数目 """
        
        import os
        import sys

        # Get a snapshot of currently connected devices
        ctx = rs.context()
        device_list = ctx.query_devices()
        # # if len(device_list) == 0:
        # print(device_list)  # <pyrealsense2.pyrealsense2.device_list object at 0x0000014C6F4F3298>
        # print(len(device_list))  # 1
        # for dev in device_list:
        #     print(dev)  # <pyrealsense2.device: Intel RealSense D415 (S/N: 915422060024)>
        # print(device_list.size())  # 1，size() numpy 函数
        # print(type(device_list))  # <class 'pyrealsense2.pyrealsense2.device_list'>
        if device_list.size == 0:
            print("no realsense devices found... ")
            os.system("pause")
            sys.exit(0)
        else:  # 打印设备信息
            for i, device in enumerate(device_list):
                i += 1
                device_name = device.get_info(rs.camera_info.name)
                print("found [{0}] device(s) required :: [{1}]".format(i, device_name))
                print("resolution :: {0}".format(resolution))
    
    def rs_read_frames(self):
        """ 读取 realsense 的视频帧，返回帧:颜色 + 深度 """
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        # depth_frame, color_frame 格式?
        return (depth_frame, color_frame)
    
    def rs_get_image(self, color_frame, depth_frame):
        """ 得到 realsense 的图像，返回图像：颜色 + 深度 + 聚合图 """
        # 得到 depth_data, color_data
        depth_data = depth_frame.get_data()
        color_data = color_frame.get_data()
        
        # 转换成数组并得到图像
        depth_image = np.asanyarray(depth_data)
        color_image = np.asanyarray(color_data)

        # 得到深度图的伪彩色图
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET)

        # 聚合图像
        stack_image = np.hstack((color_image, depth_colormap))

        return (color_image, depth_colormap, stack_image)
         

    def rs_read_frames_align(self):
        """ 读取 realsense 对齐后的视频帧，返回帧：颜色 + RGB """
        pass

    def rs_get_image_align(self, )
        """ 得到 realsense 的对齐后的图像，返回图像：颜色 + 深度 + 聚合图 """

if __name__ == "__main__":
    realsense_camera = Realsense()
