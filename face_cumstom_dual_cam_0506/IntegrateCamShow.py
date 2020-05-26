import copy
import os
import sys
import time

import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

# prop_position = cap.get(cv.CAP_PROP_POS_MSEC)
# prop_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
# prop_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
# prop_fps = cap.get(cv.CAP_PROP_FPS)
# prop_fourcc = cap.get(cv.CAP_PROP_FOURCC)
# prop_format = cap.get(cv.CAP_PROP_FORMAT)
# prop_capture_mode = cap.get(cv.CAP_PROP_MODE)
# prop_brightness = cap.get(cv.CAP_PROP_BRIGHTNESS)
# prop_contrast = cap.get(cv.CAP_PROP_CONTRAST)
# prop_saturation = cap.get(cv.CAP_PROP_SATURATION)
# prop_hue = cap.get(cv.CAP_PROP_HUE)
# prop_gain = cap.get(cv.CAP_PROP_GAIN)
# prop_exposure = cap.get(cv.CAP_PROP_EXPOSURE)
# prop_bool_auto_exposure = cap.get(cv.CAP_PROP_AUTO_EXPOSURE)
# prop_gamma = cap.get(cv.CAP_PROP_GAMMA)
# prop_sample_aspect_ratio = cap.get(cv.CAP_PROP_SAR_NUM)
# prop_bool_auto_white_balance = cap.get(cv.CAP_PROP_AUTO_WB)
# prop_white_balance_temperature = cap.get(cv.CAP_PROP_WB_TEMPERATURE)

camera_property_dic = {
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


for prop_key, prop_cv_value in camera_property_dic.items():
    prop_value = cap.get(prop_cv_value)
    print("{0} :: {1}".format(prop_key, prop_value))


# cap.set(cv.CAP_PROP_FRAME_WIDTH, 3040)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1520)

cap.set(cv.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 960)

# cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, 960)

# cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
print(cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT))


if not cap.isOpened():
    print("Cannot open camera")
    sys.exit()



# def lag_checker():
#     pass

# LOG_PATH = r"D:\P_project_SRS\face_cumstom_dual_cam_0506\LOG"
# def record_lag(LAG_input):
#     if not os.path.exists(LOG_PATH):
#         os.mkdir(LOG_PATH)


start = time.perf_counter()
FRAME_COUNT = 0
FRAME_TOTAL = 0
LAG = 0
OK = 0

line_start_pt_1 = ()
line_end_pt_1 = ()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    frame_width = frame.shape[0]
    frame_height = frame.shape[1]

    # frame_left = frame[:, :1280]
    # frame_right = frame[:, 1280:]

    # frame_left = frame[:, :960]
    # frame_right = frame[:, 960:]

    line_start_pt_1 = (0, int(frame_height/2))
    line_end_pt_1 = (frame_width, int(frame_height/2))
    # frame_line = cv.line(frame, (0, int(frame_height/2)), (frame_width, int(frame_height/2)), (255, 0, 0), 5)
    frame_line_0 = cv.line(frame, (0, 480), (2560, 480), (0, 0, 255), 2)
    frame_line_1 = cv.line(frame_line_0, (0, 240), (2560, 240), (0, 0, 255), 2)
    frame_line_2 = cv.line(frame_line_1, (0, 720), (2560, 720), (0, 0, 255), 2)

    # for 

    # for i in range(10):
    #     frame_line 

    # frame_reverse = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    # frame_reverse = copy.deepcopy(frame)
    # frame_reverse[:, :960] = frame_right
    # frame_reverse[:, 960:] = frame_left

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    # cv.imshow('frame', gray)
    cv.namedWindow('frame_rgb', 0)
    # cv.imshow('frame_rgb', frame)
    cv.imshow('frame_rgb', frame)
    # cv.imshow('frame_rgb_reverse', frame_reverse)
    # cv.imshow('frame_rgb_left', frame_left)
    # cv.imshow('frame_rgb_right', frame_right)
    if cv.waitKey(1) == ord('q'):
        break

    FRAME_COUNT += 1
    FRAME_TOTAL += 1

    end = time.perf_counter()
    frame_time = end - start
    # print("frame_time =", frame_time)
    
    if frame_time > 0.06:
        LAG += 1
        # record_lag(LAG, frame_time, )
        print("frame_total =", FRAME_TOTAL, "frame_time =", frame_time, "LAG_times =", LAG)

    start = time.perf_counter()


    # if total > 1:
    #     print("fps =", FRAME_COUNT, ", frame_time =", (1/FRAME_COUNT))

    #     start = time.perf_counter()
    #     FRAME_COUNT = 0

        # if FRAME_COUNT > 25:
        #     LAG += 1
        #     record_lag(LAG)
        # else:
        #     OK += 1
        #     # record_ok()
        # print("ok_times::{0}, lag_times::{1}".format(OK, LAG))

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
