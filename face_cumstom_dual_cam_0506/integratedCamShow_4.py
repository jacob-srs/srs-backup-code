from threading import Thread
import time
import sys
import cv2
import numpy as np


class VideoStreamWidget(object):
    def __init__(self, src=0, fps=30):
        """ init the device with device index(=0) and fps(=30)"""
        self.frame_delay = int((1/fps)*1000)  # miniseconds

        self.capture = cv2.VideoCapture(src)
        self._init_sys_params()
        
        # self.frame = None
        # self.status = None
        self._init_frame()

        # Start the thread to read frames from the video stream in background
        # self.thread = Thread(target=self._update, args=())
        # self.thread.daemon = True  # set self.thread as daemon thread so when program quits, daemon threads are killed automatically
        # self.thread.start()
        self.thread = None

        self.save_count = 0
        self.fps_show = 0
        # self.frame_count = 0

    def _init_frame(self):
        width = self.get_camera_properity("prop_width")
        height = self.get_camera_properity("prop_height")

        self.frame = np.zeros((int(height), int(width), 3), np.uint8)
        self.status = None

    def start_daemon_thread(self):
        # Start the thread to read frames from the video stream in background
        self.thread = Thread(target=self._update, args=())
        self.thread.daemon = True  # set self.thread as daemon thread so when program quits, daemon threads are killed automatically
        self.thread.start()

    def _init_sys_params(self):
        self.camera_properity_dic = {
            "prop_width":cv2.CAP_PROP_FRAME_WIDTH,
            "prop_height":cv2.CAP_PROP_FRAME_HEIGHT,
            "prop_fps":cv2.CAP_PROP_FPS,
            "prop_fourcc":cv2.CAP_PROP_FOURCC,
            "prop_brightness":cv2.CAP_PROP_BRIGHTNESS,
            "prop_contrast":cv2.CAP_PROP_CONTRAST,
            "prop_exposure":cv2.CAP_PROP_EXPOSURE,
            "prop_bool_auto_exposure":cv2.CAP_PROP_AUTO_EXPOSURE,
        }

        self.error_dic = {
            "error_0":"properity not found",
            "error_1":"frame not grabbed",
            "error_2":"cannot open camera",
            "error_3":"set resolution failed",
            "error_4":"properity value is none",
        }

    def _update(self):
        # start background frame grabbing, daemon thread
        # Read the next frame from the stream in a different thread
        # frame_count = 0
        start = time.perf_counter()
        while True:
            if self.capture.isOpened():
                # (self.status, self.frame) = self.capture.read()
                (self.status, self.frame) = self.capture.read()
                # print(self.frame)                
                # frame_count += 1

                end = time.perf_counter()
                seconds = end - start
                fps = 1 / seconds
                print("[update] fps = {0}".format(fps))

                frame_count = 0
                start = time.perf_counter()

                # cv2.imshow("update_frame", self.frame)
            time.sleep(.01)  # interval between each frame .01 sec for main thread (?)
            # cv2.waitKey(1)
            # cv2.destroyAllWindows()

    def show_frame(self, enable_fun=False, enable_text=False, enable_line=False):
        # Display frames in main program, main thread
        # if not enable_fun:
        #     return False

        IMG_SAVE_PATH = r".\image_save"
        
        start = time.perf_counter()
        # self.fps_show = 0
        
        if enable_text:
            self._put_text(self.frame, self.fps_show)
        
        if enable_line:
            self._draw_line_by_pix(self.frame)

        cv2.namedWindow('show_frame', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow('show_frame', self.frame)
        
        key = cv2.waitKey(self.frame_delay)  # control the frame interval
        if key == ord('q'):
            # self.capture.release()
            # cv2.destroyAllWindows()
            cv2.destroyWindow('show_frame')
            exit(0)  # exit with success
            return False
        if key == ord('s'):
            cv2.imwrite(IMG_SAVE_PATH + "\{}.png".format(self.save_count), self.frame)
            print("save with s {0}".format(self.save_count))
            self.save_count += 1

        end = time.perf_counter()
        seconds = end - start
        self.fps_show = 1 / seconds
        print("--[show] fps = {0}".format(self.fps_show), seconds, self.frame_delay)

        start.time.perf_counter()

    def _put_text(self, input_frame, input_fps=0):
        text = "fps={0}".format(input_fps)
        # org = (input_frame.shape[0], 0)
        # print(input_frame.shape[0], input_frame.shape[1])  # 960, 960
        org = (50, 150)
        cv2.putText(input_frame, text, org, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    def _draw_line(self, input_frame, line_num):
        height, width = input_frame.shape[:2]

        interval = int(height/(line_num+1))
        height_accumulate = 0
        for i in range(line_num):
            if height_accumulate == height:
                break
            
            height_accumulate += interval
            input_frame = cv2.line(input_frame, (0, height_accumulate), (width, height_accumulate), (0, 0, 255), 1)

        return input_frame

    def _draw_line_by_pix(self, input_frame):
        # print("input_frame.shape[0]={0}, input_frame.shape[1]={1}".format(input_frame.shape[0], input_frame.shape[1]))
        for i in range(0, input_frame.shape[0], 20):
            # print("i={0}".format(i))
            cv2.line(input_frame, (0, i), (input_frame.shape[1], i), (0, 0, 255), 1)

    def get_all_properities(self):
        """ return all camera properity """
        for prop_key, prop_value in self.camera_properity_dic.items():
            print("{0} :: {1}".format(prop_key, self.capture.get(prop_value)))

    def get_camera_properity(self, input_properity):
        if input_properity not in [key for key in self.camera_properity_dic.keys()]:
            # print("..error_0...properity not found")
            print("..{0}...{1}".format("error_0", self.error_dic["error_0"]))
            return  # 结束函数
        else:
            return self.capture.get(self.camera_properity_dic[input_properity])

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
            return self.capture.set(self.camera_properity_dic[input_properity], value)

def main():

    video_stream_widget = VideoStreamWidget(0, 15.5)
    
    video_stream_widget.set_camera_properity("prop_height", 720)
    video_stream_widget.set_camera_properity("prop_width", 2560)
    
    video_stream_widget.start_daemon_thread()

    while True:
        # continue  # 保证 update 线程运行
        
        # show_frame_flag = False
        # key = cv2.waitKey(1)

        try:
            # start = time.perf_counter()
            # if key == ord('r'):
            #     print("r pressed")
            #     show_frame_flag = True

            video_stream_widget.show_frame(enable_text=True, enable_line=True)
            
            # show_frame_flag = video_stream_widget.show_frame(enable_text=True)  # show_frame 线程运行
            # # end = time.perf_counter()
            # seconds = end - start
            # fps = 1 / seconds
            # print("fps = {}".format(fps), seconds)
            # start.time.perf_counter()
            # if stop_flag:  # 这种
            #     break

        except AttributeError:
            pass
        # video_stream_widget.show_frame()


if __name__ == "__main__":
    main()