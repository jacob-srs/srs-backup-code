import os
import sys
import time

import numpy as np
import cv2 as cv


class IntegratedCamera:
    """ synchronized stereo camera operation """
    def __init__(self, cam_index):
        self.cap = cv.VideoCapture(cam_index)
        self.cap_open_flag = False
        self.camera_properity_dic = {
            "prop_position":cv.CAP_PROP_POS_MSEC,
            "prop_width":cv.CAP_PROP_FRAME_WIDTH,
            "prop_height":cv.CAP_PROP_FRAME_HEIGHT,
            "prop_fps":cv.CAP_PROP_FPS,
            "prop_fourcc":cv.CAP_PROP_FOURCC,
            "prop_format":cv.CAP_PROP_FORMAT,
            "prop_capture_mode":cv.CAP_PROP_MODE,
            "prop_brightness":cv.CAP_PROP_BRIGHTNESS,
            "prop_contrast":cv.CAP_PROP_CONTRAST,
            "prop_saturation":cv.CAP_PROP_SATURATION,
            "prop_hue":cv.CAP_PROP_HUE,
            "prop_gain":cv.CAP_PROP_GAIN,
            "prop_exposure":cv.CAP_PROP_EXPOSURE,
            "prop_bool_auto_exposure":cv.CAP_PROP_AUTO_EXPOSURE,
            "prop_gamma":cv.CAP_PROP_GAMMA,
            "prop_sample_aspect_ratio":cv.CAP_PROP_SAR_NUM,
            "prop_bool_auto_white_balance":cv.CAP_PROP_AUTO_WB,
            "prop_white_balance_temperature":cv.CAP_PROP_WB_TEMPERATURE,
        }
        self.error_dic = {
            "error_0":"properity not found",
            "error_1":"frame not grabbed",
            "error_2":"cannot open camera",
            "error_3":"set resolution failed",
            "error_4":"properity value is none",
        }


    def get_width(self):
        """ test function """
        return self.cap.get(cv.CAP_PROP_FRAME_WIDTH)

    def get_all_properities(self):
        """ return all camera properity """
        for prop_key, prop_value in self.camera_properity_dic.items():
            print("{0} :: {1}".format(prop_key, self.cap.get(prop_value)))
    
    def get_camera_properity(self, input_properity):
        if input_properity not in [key for key in self.camera_properity_dic.keys()]:
            # print("..error_0...properity not found")
            print("..{0}...{1}".format("error_0", self.error_dic["error_0"]))
            return  # 结束函数
        else:
            return self.cap.get(self.camera_properity_dic[input_properity])

    def set_camera_properity(self, input_properity, value):
        """ set a new cam. prop. to overwrite its old value """
        if input_properity not in [key for key in self.camera_properity_dic.keys()]:
            # print("..error_0...properity not found")
            print("..{0}...{1}".format("error_0", self.error_dic["error_0"]))
            return
        elif value is None:
            print("..{0}...{1}".format("error_4", self.error_dic["error_4"]))
            return
        else:
            return self.cap.set(self.camera_properity_dic[input_properity], value)
    
    def set_camera_properity_height_width(self, height=480, width=1280):
        """ property order: height, width  """
        ret_height = self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        ret_width = self.cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        if ret_height and ret_height:
            print("set resolution :: height={0}, width={1}".format(height, width))
        else:
            # print("..error_3...set resolution failed")
            print("..{0}...{1}".format("error_3", self.error_dic["error_3"]))

    def start_camera(self):
        self.cap_open_flag = self.cap.isOpened()
        if not self.cap_open_flag:
            print("..{0}...{1}".format("error_2", self.error_dic["error_2"]))
            return self.cap_open_flag

        return self.cap_open_flag

    def get_frame(self, enable_line="", line_num=3):
        """ paras:: enable_line="", line_num=3 """
        ret, frame = self.cap.read()
        if not ret:
            # print("..error_1...frame not grabbed")
            print("..{0}...{1}".format("error_1", self.error_dic["error_1"]))
            return
        
        height, width = frame.shape[:2]
        frame_left = frame[:, int(width/2):]
        frame_right = frame[:, :int(width/2)]

        if enable_line == "true":
            fame = self.draw_line(frame, line_num)
            fame_left = self.draw_line(frame_left, line_num)
            fame_right = self.draw_line(frame_right, line_num)
        else:
            pass
        
        return frame, frame_left, frame_right

    def draw_line(self, input_frame, line_num):
        height, width = input_frame.shape[:2]

        interval = int(height/(line_num+1))
        height_accumulate = 0
        for i in range(line_num):
            if height_accumulate == height:
                break
            
            height_accumulate += interval
            input_frame = cv.line(input_frame, (0, height_accumulate), (width, height_accumulate), (0, 0, 255), 1)

        return input_frame


    def close_camera(self):
        self.cap.release()
        self.cap_open_flag = False
        cv.destroyAllWindows()


def demo_stereo_cam_simple():
    """ stereo camera demo """

    ic = IntegratedCamera(0)

    save_path = r".\image_save"

    ic.set_camera_properity_height_width(1520, 3040)
    start_flag = ic.start_camera()

    print("="*40)
    ic.get_all_properities()
    print("="*40)

    save_img_count = 0
    frame_interval = 33  # 33ms
    while start_flag:
        frame, frame_left, frame_right = ic.get_frame("true", 15)
        
        cv.namedWindow("frame", cv.WINDOW_NORMAL)
        cv.namedWindow("frame_left", cv.WINDOW_NORMAL)
        cv.namedWindow("frame_right", cv.WINDOW_NORMAL)

        cv.imshow("frame", frame)
        cv.imshow("frame_left", frame_left)
        cv.imshow("frame_right", frame_right)

        if cv.waitKey(frame_interval) == ord('q'):  # change to 33ms = 30fps
            print("camera exit with q")
            break
        elif cv.waitKey(frame_interval) == ord('s'):
            print("frame save with s")
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            cv.imwrite(save_path + r"\{}.png".format(save_img_count), frame)
            save_img_count += 1
    
    ic.close_camera()


def demo_multi_threading():
    import threading
    pass



from threading import Thread
import cv2, time
import sys

class VideoStreamWidget(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(.01)

    def show_frame(self):
        # Display frames in main program
        cv2.imshow('frame', self.frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            # exit(1)

if __name__ == '__main__':
    video_stream_widget = VideoStreamWidget()
    while True:
        try:
            video_stream_widget.show_frame()
        except AttributeError:
            pass



# if __name__ == "__main__":
#     #!/usr/bin/python3
    
#     import threading
#     import time
    
#     class myThread (threading.Thread):
#         def __init__(self, threadID, name, counter):
#             threading.Thread.__init__(self)
#             self.threadID = threadID
#             self.name = name
#             self.counter = counter
#         def run(self):
#             print ("开启线程： " + self.name)
#             # 获取锁，用于线程同步
#             threadLock.acquire()
#             print_time(self.name, self.counter, 3)
#             # 释放锁，开启下一个线程
#             threadLock.release()
    
#     def print_time(threadName, delay, counter):
#         while counter:
#             time.sleep(delay)
#             print ("%s: %s" % (threadName, time.ctime(time.time())))
#             counter -= 1
    
#     threadLock = threading.Lock()
#     threads = []
    
#     # 创建新线程
#     thread1 = myThread(1, "Thread-1", 1)
#     thread2 = myThread(2, "Thread-2", 2)
    
#     # 开启新线程
#     thread1.start()
#     thread2.start()
    
#     # 添加线程到线程列表
#     threads.append(thread1)
#     threads.append(thread2)
    
#     # 等待所有线程完成
#     for t in threads:
#         t.join()
#     print ("退出主线程")