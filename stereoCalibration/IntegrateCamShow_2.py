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


    def get_width(self):
        """ test function """
        return self.cap.get(cv.CAP_PROP_FRAME_WIDTH)

    def get_camera_properity(self, input_properity):
        if input_properity not in [key for key in self.camera_properity_dic.keys()]:
            print("..error_0...properity not found")
            return  # 结束函数
        else:
            return self.cap.get(self.camera_properity_dic[input_properity])

    def set_camera_properity(self, input_properity):
        pass
    
    def set_camera_properity_height_width(self, height=480, width=1280):
        """ property order: height, width  """
        ret_height = self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        ret_width = self.cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        if ret_height and ret_height:
            print("set resolution :: height={0}, width={1}".format(height, width))
        else:
            print("..error_3...set resolution failed")

    def start_camera(self):
        if not self.cap.isOpened():
            print("..error_2...cannot open camera")
            self.cap_open_flag = False
            return self.cap_open_flag

        self.cap_open_flag = True
        return self.cap_open_flag

    def get_frame(self, enable_line="", line_num=3):
        ret, frame = self.cap.read()
        if not ret:
            print("..error_1...frame not grabbed")
            return
        
        height, width = frame.shape[:2]
        frame_left = frame[:, int(width/2):]
        frame_right = frame[:, :int(width/2)]

        if enable_line == "true":
            # enable_line = True
            
            fame = self.draw_line(frame, line_num)
            fame_left = self.draw_line(frame_left, line_num)
            fame_right = self.draw_line(frame_right, line_num)
        
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


if __name__ == "__main__":
    # import cv2 as cv

    ic = IntegratedCamera(0)

    save_path = r".\image_save"

    ic.set_camera_properity_height_width(480, 1280)
    start_flag = ic.start_camera()

    height = ic.get_camera_properity("prop_height")
    width = ic.get_camera_properity("prop_width")
    print("get resolution :: height={0}, width={1}".format(height, width))

    save_img_count = 0
    while start_flag:
        frame, frame_left, frame_right = ic.get_frame("true", 15)
        cv.imshow("frame", frame)
        cv.imshow("frame_left", frame_left)
        cv.imshow("frame_right", frame_right)

        if cv.waitKey(1) == ord('q'):
            print("camera exit with q")
            break
        elif cv.waitKey(1) == ord('s'):
            print("frame save with s")
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            cv.imwrite(save_path + r"\{}.png".format(save_img_count), frame)
            save_img_count += 1

    ic.close_camera()