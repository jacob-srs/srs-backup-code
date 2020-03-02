'''
@Author: your name
@Date: 2019-12-12 09:10:49
@LastEditTime : 2019-12-18 16:20:59
@LastEditors  : Please set LastEditors
@Description: 这个程序是对 08 的 拷贝 与 整理，提高稳健性。
@reference: 关联文件夹：C:\LeeSRSPrgoramFile\V_vsCode\.vscode\20191118_dualCam\QTUI_DOC\QTUI_DOC_TEST12_OCR03_FT03
@FilePath: \V_vsCode\.vscode\20191118_dualCam\opencv_CameraRectification_09.py
'''
import os
import sys

import cv2
import glob
import numpy as np
import re

from matplotlib import pyplot as plt


class Dual_Camera_Single_Frame(object):
    """ 对0，2，3，5，6，10，11双目图像对进行 CR """
    # 类常量
    ROOT_PATH = (  # 主路径，修改后影响程序所有路径
        # r".vscode\20191118_dualCam\QTUI_DOC\DUEL_CAM_DOC"
        r"C:\Users\lijin\OneDrive\srs-backup-code\stereoCalibration"
    )
    IMG_ORG_PATH = glob.glob(
        ROOT_PATH +
        # r"\image_dichotomy\cam_org_*.jpg"
        r"\image_dichotomy\cam_org_*.png"
    )
    IMG_ORG_VALID_PATH = (
        ROOT_PATH +
        r"\cam_org_valid"
    )
    IMG_CORNER_PATH = (
        ROOT_PATH +
        r"\cam_corner"
    )
    INFO_SAVE_PATH = (
        ROOT_PATH +
        r"\save_info"
    )

    pattern_left_path = re.compile(r'[\w\s\\.]*left[\w\s\\.]*')
    pattern_right_path = re.compile(r'[\w\s\\.]*right[\w\s\\.]*')

    # pattern_size = (7, 6) # 对于每一个不同的棋盘格，格内尺寸需要修改，org教程
    pattern_size = (7, 7) # 对于每一个不同的棋盘格，格内尺寸需要修改，mindvision

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    # objp = objp * 30  # mm 对于每一个不同的棋盘格，每个格子的实际尺寸必须注明，这个是 opencv org 教程中的棋盘格尺寸
    objp = objp * 34  # mm 对于每一个不同的棋盘格，每个格子的实际尺寸必须注明，这个是 opencv org 教程中的棋盘格尺寸，mindvision
    # objp = objp

    # pattern_size = (10, 7)

    def __init__(self):
        super().__init__()

        # 实例变量区
        self.image_size = None

        # 初始化函数区
        # self.make_directory()

    def make_directory(self, relative_path):
        """ 创建必要的文件路径 """
        # if not os.path.exists(self.INFO_SAVE_PATH):
        #     os.mkdir(self.INFO_SAVE_PATH)
        # else:
        #     pass

        if not os.path.exists(relative_path):
            os.mkdir(relative_path)
        else:
            pass

    def custom_imwrite(self, input_image, relative_path):
        if not input_image.data:  # 只用查看image有无
            print("\n{0}\n{1}".format(
                "...in self-defined imwrite",
                "...input_image error"
            ))

        directory_list = relative_path.split('\\')[:-1]
        directory = None
        for i in range(len(directory_list)):
            # if directory_list[i+1]:
            #     directory = directory + directory_list[i] + '\\' + directory_list[i+1]
            if i == 0:
                directory = directory_list[0]
            else:
                directory = directory + '\\' + directory_list[i]
        self.make_directory(directory)
        print("in custom_imwrite, input_image.dtype = ", input_image.dtype)
        cv2.imwrite(relative_path, input_image)

    # 有 icon save函数，完全不需要为 icon 重新写一个函数，python 太牛逼
    def save_camera_info(self, icon=None, **keys_values):
        """ 保存任意长度计算机数据信息"""
        if icon is not None:
            for key, value in keys_values.items():
                self.make_directory(self.INFO_SAVE_PATH)
                np.save(
                    self.INFO_SAVE_PATH + r"\{0}_{1}.npy"  # key, direction
                    .format(key, icon),
                    value
                )
        elif icon is None:
            for key, value in keys_values.items():
                self.make_directory(self.INFO_SAVE_PATH)
                np.save(
                    self.INFO_SAVE_PATH + r"\{0}.npy"  # key, direction
                    .format(key),
                    value
                )

    def read_camera_info(self, *name_string):
        """ 读取任意的保存了的相机信息 """
        info_paths = glob.glob(self.INFO_SAVE_PATH + r"\*.npy")

        # test
        print("\nname_string::\n", name_string)
        return_list = []
        count = 0
        for input_name in name_string:
            # print("\ninput_name:: ", input_name)
            for info_path in info_paths:
                info_name = info_path.split('\\')[-1].split('.')[-2]
                # print("\ninfo_name:: ", info_name)
                if input_name == info_name:
                    temp_value = np.load(info_path)
                    return_list.append(temp_value)
                    # count += 1
                else:
                    # count += 1
                    # print("\n{} input value not found" .format(count))
                    pass
            count += 1

        if len(return_list) != len(name_string):
            print(
                "\nfound {0} infos out of {1}, read_camera_info error"
                .format(len(return_list), len(name_string)))
            return -1
        else:
            return return_list

    def pair_dual_images(self):
        """ 对采集的双目图像按名称进行匹配 """
        # 函数变量区
        image_path_list_left = [None] * 200
        image_path_list_right = [None] * 200

        for path in self.IMG_ORG_PATH:
            image_name = path.split('\\')[-1]
            image_direction = image_name.split('_')[-2]
            image_index = image_name.split('_')[-1].split('.')[0]

            if image_direction == "left":
                image_path_list_left[int(image_index)] = path
                # print("\nleft_path::", path)  # 没问题，没 None
            elif image_direction == "right":
                image_path_list_right[int(image_index)] = path

        # 写成 return 形式
        return (image_path_list_left, image_path_list_right)

    def find_corners(self):
        """ 对双目图像采集的棋盘标定格进行角点查找
        只保存匹配图像都有角点的图像对点坐标, 这个函数得到
        1. img_corner_points_left
        2. img_corner_points_right
        3. obj_points"""
        # 函数变量区
        (
            image_path_list_left,
            image_path_list_right
        ) = self.pair_dual_images()

        # print(
        #     "\nimage_left_path_list_length::{0}\n" +
        #     "image_right_path_list_length::{1}"
        #     .format(len(image_path_list_left), len(image_path_list_right))
        # )
        print(
            "\nimage_left_path_list_length::{0}"
            .format(len(image_path_list_left))
        )
        print(
            "\nimage_right_path_list_length::{0}"
            .format(len(image_path_list_right))
        )

        img_corner_points_left = []
        img_corner_points_right = []
        obj_points = []  # 定义 obj_points

        cornerSubPix_criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )

        img_combined_path_list = zip(
            image_path_list_left,  # 200 个，有 None
            image_path_list_right  # 同上
        )

        count = 0
        count_both_found = 0
        flag_save = []
        for image_path in img_combined_path_list:
            """ 读取双目图像，找每个图像的角点，并且将都有角点的图像对角点存储 """
            # print("{}\n{}\n" .format(image_path[0], image_path[1]))

            img_path_left = None
            img_path_right = None
            if image_path[0] and image_path[1] is not None:
                img_path_left = image_path[0]
                img_path_right = image_path[1]
                # print("{}\n{}\n" .format(img_path_left, img_path_right))
            else:
                count += 1
                continue
                # break

            img_left = cv2.imread(img_path_left)
            img_right = cv2.imread(img_path_right)
            img_gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            img_gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

            # cv2.imshow("left img {}" .format(count), img_left)
            # cv2.imshow("right img {}" .format(count), img_right)

            # if count == 0:  # 只保存一次就够了,哈哈，merge 是插值存储的，所以没有存上
            #     self.image_size = img_gray_left.shape  # 这个没存
            #     self.save_camera_info(image_size=self.image_size)
            # else:
            #     pass

            # flag = count
            if img_left is not None:
                flag_save.append(count)
                if len(flag_save) == 1:
                    self.image_size = img_gray_left.shape  # 这个没存
                    self.save_camera_info(image_size=self.image_size)
                else:
                    pass

            # findChessboardCorners() for dual camera
            find_chessboard_flags = (
                cv2.CALIB_CB_ADAPTIVE_THRESH |
                cv2.CALIB_CB_NORMALIZE_IMAGE |
                cv2.CALIB_CB_FAST_CHECK
            )

            # find_chessboard_flags = None

            found_left, corners_left = cv2.findChessboardCorners(
                img_gray_left,
                self.pattern_size,  # 全局定义 pattern_size
                find_chessboard_flags
            )
            found_right, corners_right = cv2.findChessboardCorners(
                img_gray_right,
                self.pattern_size,
                find_chessboard_flags
            )

            # cornerSubPix() for dual camera
            if found_left:
                left_corners_2 = cv2.cornerSubPix(
                    img_gray_left,
                    corners_left,
                    (11, 11),
                    (-1, -1),
                    cornerSubPix_criteria
                )
            if found_right:
                right_corners_2 = cv2.cornerSubPix(
                    img_gray_right,
                    corners_right,
                    (11, 11),
                    (-1, -1),
                    cornerSubPix_criteria
                )

            # 这里不用放 drawChessboardCOrners,因为会在两图不同时有
            # 角点的情况下绘图，浅拷贝 img1 = img2，深拷贝 img1 = img2.copy()
            img_corner_left = img_left.copy()
            img_corner_right = img_right.copy()

            if found_left and found_right:
                # 如果图像对都有角点，就需要检测角点的有效性
                # 逻辑: 有效性条件
                # (left_start_x - left_end_x) -
                # (right_start_x - right_end_x) = +/- 40
                # s -> ([0][0][0] - [0][0][1]) - ([41][0][0] - [41][0][1])
                # (left_start_y - left_end_y) -
                # (right_start_y - right_end_y) = +/- 40
                # R -> ([0][0][0] - [0][0][1]) - ([41][0][0] - [41][0][1])
                first_value = (
                    (corners_left[0][0][0] - corners_left[41][0][0]) -
                    (corners_right[0][0][0] - corners_right[41][0][0])
                )
                second_value = (
                    (corners_left[0][0][0] - corners_left[41][0][0]) -
                    (corners_right[0][0][0] - corners_right[41][0][0])
                )
                if abs(first_value) <= 40 and abs(second_value) <= 40:
                    img_corner_points_left.append(corners_left)
                    img_corner_points_right.append(corners_right)
                    obj_points.append(self.objp)  # 定义 obj_points

                    # cv2.imshow("left", img_left)
                    cv2.drawChessboardCorners(
                        img_corner_left,
                        self.pattern_size,
                        left_corners_2,
                        found_left
                    )
                    cv2.drawChessboardCorners(
                        img_corner_right,
                        self.pattern_size,
                        right_corners_2,
                        found_right
                    )

                    # 保存原图，保存角点图。
                    self.make_directory(self.IMG_ORG_VALID_PATH)
                    self.make_directory(self.IMG_CORNER_PATH)
                    # cv2.imwrite(
                    #     self.IMG_ORG_VALID_PATH +
                    #     r"\cam_org_valid_left_{}.jpg" .format(count),
                    #     img_left
                    # )
                    # cv2.imwrite(
                    #     self.IMG_ORG_VALID_PATH +
                    #     r"\cam_org_valid_right_{}.jpg" .format(count),
                    #     img_right
                    # )
                    # cv2.imwrite(
                    #     self.IMG_CORNER_PATH +
                    #     r"\cam_corner_left_{}.jpg" .format(count),
                    #     img_corner_left
                    # )
                    # cv2.imwrite(
                    #     self.IMG_CORNER_PATH +
                    #     r"\cam_corner_right_{}.jpg" .format(count),
                    #     img_corner_right
                    # )
                    cv2.imwrite(
                        self.IMG_ORG_VALID_PATH +
                        r"\cam_org_valid_left_{}.png" .format(count),
                        img_left
                    )
                    cv2.imwrite(
                        self.IMG_ORG_VALID_PATH +
                        r"\cam_org_valid_right_{}.png" .format(count),
                        img_right
                    )
                    cv2.imwrite(
                        self.IMG_CORNER_PATH +
                        r"\cam_corner_left_{}.png" .format(count),
                        img_corner_left
                    )
                    cv2.imwrite(
                        self.IMG_CORNER_PATH +
                        r"\cam_corner_right_{}.png" .format(count),
                        img_corner_right
                    )

                    count_both_found += 1

            cv2.imshow("left chess {}" .format(count), img_corner_left)
            cv2.imshow("right chess {}" .format(count), img_corner_right)

            count += 1
            cv2.waitKey(100)
            cv2.destroyAllWindows()
        print("count_both_found::", count_both_found)

        # 保存 此函数 的 output
        self.save_camera_info(
            img_corner_points_left=img_corner_points_left
        )
        self.save_camera_info(
            img_corner_points_right=img_corner_points_right
        )
        self.save_camera_info(obj_points=obj_points)
        # 返回 此函数 的 output
        return (img_corner_points_left, img_corner_points_right, obj_points)

    def stereo_calibrate(self):
        """ 立体相机标度步骤
        1. calibrateCamera() --> left
            in--> obj_points
            in--> img_corner_points_left
            in--> image_size
            out--> mtx_left, dist_left
        2. calibrateCamera() --> right
            in--> obj_points
            in--> img_corner_points_right
            in--> image_size
            out--> mtx_left, dist_left
        3. stereoCalibrate()
            in--> obj_points, i_l_c_p, i_r_c_p
            in--> mtx_left, mtx_right, dist_left, dist_right
            in--> img.size(w, h), None, None, None, None,
            flag, criteria
            out--> rvecs, tvecs
        Calibrates the stereo camera.
        标定立体相机"""
        # height, width = self.image_size[:2]  # 实例变量

        # 读取所需变量，输出顺序跟输入顺序一致，规范程序变量命名
        image_size = None
        obj_points = None
        img_corner_points_left = None
        img_corner_points_right = None

        info_list = dcsf.read_camera_info(
            "image_size",
            "obj_points",
            "img_corner_points_left",
            "img_corner_points_right"
        )
        if not info_list:
            print("\nin stereo_calibrate, read_camera_info error")
            sys.exit(0)
        else:
            image_size = info_list[0]
            obj_points = info_list[1]
            img_corner_points_left = info_list[2]
            img_corner_points_right = info_list[3]
            print("\nall found")

        height, width = image_size[:2]  # 实例变量

        # calibrateCamera() left
        (
            ret_calib_left,
            mtx_left,
            dist_left,
            _,  # rvecs_left
            _  # tvecs_left
        ) = cv2.calibrateCamera(
            obj_points,  # 通过 return/read 取得
            img_corner_points_left,  # 通过 return/read 取得
            (width, height),
            None,
            None
        )
        # print("\nmtx_left=\n", mtx_left)
        # print("\ndist_left=\n", dist_left)

        if not ret_calib_left:
            print(
                "\ncalibrateCamera left error!"
            )
        # calibrateCamera() right
        (
            ret_calib_right,
            mtx_right,
            dist_right,
            _,  # rvecs_right
            _  # tvecs_rigth
        ) = cv2.calibrateCamera(
            obj_points,  # 通过 return/read 取得
            img_corner_points_right,  # 通过 return/read 取得
            (width, height),
            None,
            None
        )
        if not ret_calib_left:
            print(
                "\ncalibrateCamera left error!"
            )
        # stereoCalibrate()
        stereocalib_criteria = (
            cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
            100,
            1e-5
        )
        stereocalib_flags = (
            cv2.CALIB_FIX_ASPECT_RATIO |
            cv2.CALIB_ZERO_TANGENT_DIST |
            cv2.CALIB_SAME_FOCAL_LENGTH |
            cv2.CALIB_RATIONAL_MODEL |
            cv2.CALIB_FIX_K3 |
            cv2.CALIB_FIX_K4 |
            cv2.CALIB_FIX_K5
        )
        # stereoCalibrate() 函数区
        (
            stereocalib_retval,
            _,
            _,
            _,
            _,
            stereoCalibrate_R,
            stereoCalibrate_T,
            _,
            _
        ) = cv2.stereoCalibrate(
            obj_points,  # 通过 return/read 取得
            img_corner_points_left,  # 通过 return/read 取得
            img_corner_points_right,  # 通过 return/read 取得
            mtx_left,  # calibrate left o/p
            dist_left,  # calibrate left o/p
            mtx_right,  # calibrate right o/p
            dist_right,  # calibrate right o/p
            (width, height),
            None,  # R:output rotation mtx b/w cam1 and cam2
            None,  # T:output translation vector b/w coord
            None,  # E:output essential matrix
            None,  # F:output fundamental mtx
            criteria=stereocalib_criteria,
            flags=stereocalib_flags
        )
        if not stereocalib_retval:
            print("\nstereoCalibrate error")

        self.save_camera_info(mtx_left=mtx_left, dist_left=dist_left)
        self.save_camera_info(mtx_right=mtx_right, dist_right=dist_right)
        self.save_camera_info(
            stereoCalibrate_R=stereoCalibrate_R,
            stereoCalibrate_T=stereoCalibrate_T
        )

        return (
            mtx_left, dist_left, mtx_right, dist_right,
            stereoCalibrate_R, stereoCalibrate_T
        )

    # 在这个函数 修改 rectify_scale 的值
    def stereo_rectify(self):
        """ 对立体相机进行校正
        Computes rectification transforms for each head of a calibrated stereo camera.
        计算 矫正变换，对每一个校准后的立体摄像机镜头 """
        print("\nin stereo_rectify")

        # 读取所需变量，输出顺序跟输入顺序一致，规范程序变量命名
        mtx_left = None  # input
        dist_left = None  # input
        mtx_right = None  # input
        dist_right = None  # input
        stereoCalibrate_R = None  # input
        stereoCalibrate_T = None  # input
        image_size = None

        # 读取相机参数信息
        info_list = dcsf.read_camera_info(
            "mtx_left",
            "dist_left",
            "mtx_right",
            "dist_right",
            "stereoCalibrate_R",
            "stereoCalibrate_T",
            "image_size"
        )
        if not info_list:
            print("\nin stereo_calibrate, read_camera_info error")
            sys.exit(0)
        else:
            mtx_left = info_list[0]
            dist_left = info_list[1]
            mtx_right = info_list[2]
            dist_right = info_list[3]
            stereoCalibrate_R = info_list[4]
            stereoCalibrate_T = info_list[5]
            image_size = info_list[6]
            print("\nall found")

        height, width = image_size[:2]
        """ 重要参数 """
        rectify_scale = 0  # 0=full crop, 1=no crop 这个参数太重要了
        # rectify_scale = 1  # 0=full crop, 1=no crop 这个参数太重要了
        """ 重要参数 """
        # stereoRectify process
        (
            stereoRectify_R1,  # 3x3 rectification left
            stereoRectify_R2,  # 3x3 rectification right
            stereoRectify_P1,  # 3x4 projection left
            stereoRectify_P2,  # 3x4 projection right
            stereoRectify_Q,  # 4x4 disparity-to-depth
            stereoRectify_roi1,  # green rectangles roi left
            stereoRectify_roi2  # green rectangles roi right
        ) = cv2.stereoRectify(
            mtx_left,  # input
            dist_left,  # input
            mtx_right,  # input
            dist_right,  # input
            (width, height),  # 这个必须是tuple input
            stereoCalibrate_R,  # input
            stereoCalibrate_T,  # input
            None,  # R1:
            None,  # R2:
            None,  # P1:
            None,  # P2:
            None,  # Q:
            cv2.CALIB_ZERO_DISPARITY,
            alpha=rectify_scale
        )

        # 保存与返回
        self.save_camera_info(
            stereoRectify_R1=stereoRectify_R1,
            stereoRectify_R2=stereoRectify_R2,
            stereoRectify_P1=stereoRectify_P1,
            stereoRectify_P2=stereoRectify_P2,
            stereoRectify_Q=stereoRectify_Q,
            stereoRectify_roi1=stereoRectify_roi1,
            stereoRectify_roi2=stereoRectify_roi2,
        )

        return (
            stereoRectify_R1, stereoRectify_R2,
            stereoRectify_P1, stereoRectify_P2,
            stereoRectify_Q, stereoRectify_roi1,
            stereoRectify_roi2
        )

    def stereo_undistortion(self):
        # 读取所需变量，输出顺序跟输入顺序一致，规范程序变量命名
        mtx_left = None  # input
        dist_left = None  # input
        mtx_right = None  # input
        dist_right = None  # input
        stereoRectify_R1 = None  # input
        stereoRectify_P1 = None  # input
        stereoRectify_R2 = None  # input
        stereoRectify_P2 = None  # input
        image_size = None

        info_list = dcsf.read_camera_info(
            "mtx_left",
            "dist_left",
            "mtx_right",
            "dist_right",
            "stereoRectify_R1",
            "stereoRectify_P1",
            "stereoRectify_R2",
            "stereoRectify_P2",
            "image_size"
        )
        if not info_list:
            print("\nin stereo_calibrate, read_camera_info error")
            sys.exit(0)
        else:
            mtx_left = info_list[0]
            dist_left = info_list[1]
            mtx_right = info_list[2]
            dist_right = info_list[3]
            stereoRectify_R1 = info_list[4]
            stereoRectify_P1 = info_list[5]
            stereoRectify_R2 = info_list[6]
            stereoRectify_P2 = info_list[7]
            image_size = info_list[8]
            print("\nall found")

        height, width = image_size[:2]  # 480, 640
        # Computes the undistortion and rectification transformation map.
        maps_x_left, maps_y_left = cv2.initUndistortRectifyMap(
            mtx_left,  # input
            dist_left,  # input
            stereoRectify_R1,  # input
            stereoRectify_P1,  # input
            (width, height),
            cv2.CV_32FC2
        )
        maps_x_right, maps_y_right = cv2.initUndistortRectifyMap(
            mtx_right,  # input
            dist_right,  # input
            stereoRectify_R2,  # input
            stereoRectify_P2,  # input
            (width, height),
            cv2.CV_32FC2
        )

        self.save_camera_info(
            maps_x_left=maps_x_left,
            maps_y_left=maps_y_left,
            maps_x_right=maps_x_right,
            maps_y_right=maps_y_right
        )

        return maps_x_left, maps_y_left, maps_x_right, maps_y_right

    def stereo_remapping(self):
        (
            maps_x_left, maps_y_left,
            maps_x_right, maps_y_right
        ) = self.stereo_undistortion()

        # 读取图像对
        (
            image_path_list_left,
            image_path_list_right
        ) = self.pair_dual_images()

        img_combined_path_list = zip(
            image_path_list_left,  # 200 个，有 None
            image_path_list_right  # 同上
        )

        # count = 0
        count_remap = 0
        for image_path in img_combined_path_list:
            """ 读取双目图像，找每个图像的角点，并且将都有角点的图像对角点存储 """
            # print("{}\n{}\n" .format(item[0], item[1]))
            print("{}\n{}\n" .format(image_path[0], image_path[1]))

            img_path_left = None
            img_path_right = None
            if image_path[0] and image_path[1] is not None:
                img_path_left = image_path[0]
                img_path_right = image_path[1]
                # print("{}\n{}\n" .format(img_path_left, img_path_right))
            else:  # 这到底是用 break 还是 continue
                count_remap += 1
                continue
                # break

            img_left = cv2.imread(img_path_left)
            img_right = cv2.imread(img_path_right)

            img_gray_left = cv2.cvtColor(
                img_left,
                cv2.COLOR_BGR2GRAY)
            img_gray_right = cv2.cvtColor(
                img_right,
                cv2.COLOR_BGR2GRAY)

            # remap each image pair
            # Applies a generic geometrical transformation to an image.
            img_remap_left = cv2.remap(
                img_gray_left,
                maps_x_left,
                maps_y_left,
                cv2.INTER_LANCZOS4
                # cv2.INTER_LINEAR
            )
            img_remap_right = cv2.remap(
                img_gray_right,
                maps_x_right,
                maps_y_right,
                cv2.INTER_LANCZOS4
                # cv2.INTER_LINEAR
            )

            # 将 gray 2 bgr
            img_remap_left = cv2.cvtColor(img_remap_left, cv2.COLOR_GRAY2BGR)
            img_remap_right = cv2.cvtColor(img_remap_right, cv2.COLOR_GRAY2BGR)

            (
                stereoRectify_roi1,
                stereoRectify_roi2
            ) = self.read_camera_info(
                "stereoRectify_roi1",
                "stereoRectify_roi2"
            )

            img_remap_left = cv2.rectangle(
                img_remap_left,
                (stereoRectify_roi1[0], stereoRectify_roi1[1]),
                (stereoRectify_roi1[2], stereoRectify_roi1[3]),
                (0, 255, 0),
                2
            )
            img_remap_right = cv2.rectangle(
                img_remap_right,
                (stereoRectify_roi2[0], stereoRectify_roi2[1]),
                (stereoRectify_roi2[2], stereoRectify_roi2[3]),
                (0, 255, 0),
                2
            )

            print(
                "\nimg_remap_right shape::{}\ttype::{}"
                .format(
                    img_remap_right.shape, img_remap_right.dtype
                )
            )
            
            # 融合图像 20200228
            row_number_left = img_remap_left.shape[0]
            row_number_right = img_remap_left.shape[0]
            col_number_left = img_remap_right.shape[1]
            col_number_right = img_remap_right.shape[1]
            image_type = img_remap_left.dtype

            img_remap_merge = np.zeros((row_number_left, col_number_left + col_number_right), image_type)
            # img_remap_merge = np.stack((img_remap_merge,)*3, axis=1)  # 转化成 3 通道
            img_remap_merge = cv2.cvtColor(img_remap_merge, cv2.COLOR_GRAY2BGR)

            img_remap_merge[:, :col_number_left] = img_remap_left
            img_remap_merge[:, col_number_left:] = img_remap_right

            # 开始画线 60 条
            line_number = 60
            pixel_interval = row_number_left / line_number
            for i in range(line_number):
                if i == 0:
                    continue
                cv2.line(
                    img_remap_merge,
                    (0, int(i*pixel_interval)),
                    (img_remap_merge.shape[1], int(i*pixel_interval)),
                    (0, 0, 255),
                    1
                )

            # img_remap_merge = np.zeros(
            #     (
            #         img_remap_left.shape[0],
            #         img_remap_left.shape[1] + img_remap_right.shape[1],
            #         3
            #     ),
            #     np.uint8
            # )
            # img_remap_merge = np.zeros(
            #     (
            #         img_remap_left.shape[0],
            #         2000,
            #         3
            #     ),
            #     np.uint8
            # )
            # img_remap_merge.fill(255)  # 填色 white
            # # 使用 boardcast 拷贝图像
            # # img_remap_merge[:, :640, :] = img_remap_left
            # # img_remap_merge[:, 640:, :] = img_remap_right
            # img_remap_merge[:, :1000, :] = img_remap_left[:, 140:1140, :]
            # img_remap_merge[:, 1000:, :] = img_remap_right[:, 140:1140, :]
            # # 对 img_remap_merge 进行 基本图形绘制
            # # 绘制 46 条线，0没有，48不包括
            # loop_index = img_remap_merge.shape[0]
            # for i in range(int(loop_index / 10)):
            #     if i == 0:
            #         continue

            #     cv2.line(
            #         img_remap_merge,
            #         (0, i * 10),  # pt1
            #         (img_remap_merge.shape[1], i * 10),  # pt2
            #         (0, 0, 255),
            #         1
            #     )
            # cv2.imshow("img_remap_merge", img_remap_merge)
            # print("\nimg_remap_merge shape::", img_remap_merge.shape)

            # 还是没有得到结果
            cv2.imshow(
                "img_left{}"
                .format(count_remap), img_gray_left)
            cv2.imshow(
                "img_right{}"
                .format(count_remap), img_gray_right)
            cv2.imshow(
                "img_remap_left_{}"
                .format(count_remap), img_remap_left)
            cv2.imshow(
                "img_remap_right_{}"
                .format(count_remap), img_remap_right)
            cv2.imshow(
                "img_remap_merge_{}"
                .format(count_remap), img_remap_merge)

            self.custom_imwrite(
                img_remap_left,
                (
                    self.ROOT_PATH
                    + r"\cam_remap\cam_remap_left_{}.png".format(count_remap)
                )
            )
            self.custom_imwrite(
                img_remap_right,
                (
                    self.ROOT_PATH
                    + r"\cam_remap\cam_remap_right_{}.png".format(count_remap)
                )
            )
            # depth map for 1080p
            # image_depth_left = np.zeros(
            #     img_remap_left[:, 140:1140, :].shape,
            #     np.uint8
            # )
            # image_depth_right = np.zeros(
            #     img_remap_right[:, 140:1140, :].shape,
            #     np.uint8
            # )

            # image_depth_left = img_remap_left[:, 140:1140, :].copy()
            # image_depth_right = img_remap_right[:, 140:1140, :].copy()

            # image_depth_left = np.zeros(
            #     img_remap_left[:, 140:1140, :].shape,
            #     np.uint8
            # )
            # image_depth_right = np.zeros(
            #     img_remap_right[:, 140:1140, :].shape,
            #     np.uint8
            # )

            image_depth_left = np.zeros(
                img_remap_left.shape,
                np.uint8
            )
            image_depth_right = np.zeros(
                img_remap_right.shape,
                np.uint8
            )

            image_depth_left = img_remap_left.copy()
            image_depth_right = img_remap_right.copy()

            """ stereoBM 算法"""
            # stereoMatcher = cv2.StereoBM_create()

            # stereoMatcher.setMinDisparity(4)
            # stereoMatcher.setNumDisparities(16)
            # stereoMatcher.setBlockSize(15)
            # stereoMatcher.set
            # # stereoMatcher.setSpeckleRange(16)
            # # stereoMatcher.setSpeckleWindowSize(45)

            # image_depth_left_gray = cv2.cvtColor(
            #     image_depth_left,
            #     cv2.COLOR_BGR2GRAY
            # )
            # image_depth_right_gray = cv2.cvtColor(
            #     image_depth_right,
            #     cv2.COLOR_BGR2GRAY
            # )

            image_depth_left_gray = img_remap_left.copy() # uint8
            image_depth_right_gray = img_remap_right.copy()

            cv2.imshow("image_depth_left_gray", image_depth_left_gray)
            cv2.imshow("image_depth_right_gray", image_depth_right_gray)

            # depth = stereoMatcher.compute(  # 这里的 depth.dtype = int16
            #     image_depth_left_gray,
            #     image_depth_right_gray
            # )

            # depth = depth.astype(np.uint16)  # int16 -> uint16

            # disparity range tuning
            self.do_stereoSBGM(
                image_depth_left_gray, image_depth_right_gray,
                count_remap)

            # cv2.waitKey(50)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # plt.imshow(depth, 'gray')
            # plt.show()

            count_remap += 1

    # stereoSBGM (semi-block-global-matching) 函数
    def do_stereoSBGM(self, input_left, input_right, input_count):
            image_depth_left_gray = input_left
            image_depth_right_gray = input_right
            count_remap = input_count

            window_size = 3
            min_disp = 0
            num_disp = 320 - min_disp

            stereoMatcher = cv2.StereoSGBM_create(
                # 最小视差
                minDisparity=0,
                # 视差范围，即最大视差值和最小视差值之差，必须是16的倍数。
                numDisparities=16*10,
                # 匹配块大小，大于1的奇数  
                blockSize=3,
                # P1, P2控制视差图的光滑度
                # 惩罚系数，一般：P1 = 8 * 通道数*SADWindowSize*SADWindowSize，
                # P2 = 4 * P
                P1=8 * 3 * window_size ** 2,
                # wsize default 3; 5; 7 for SGBM reduced size image; 
	            # 15 for SGBM full size image (1300px and above); 
                # 5 Works nicely
                P2=32 * 3 * window_size ** 2,
                # 左右视差图的最大容许差异（超过将被清零），默认为 - 1，
                # 即不执行左右视差检查
                disp12MaxDiff=1,
                # 视差唯一性百分比， 视差窗口范围内最低代价是次低代价的
                # (1 + uniquenessRatio / 100)倍时，最低代价对应的视差值才是
                # 该像素点的视差，否则该像素点的视差为 0，通常为5~15.
                uniquenessRatio=15,
                # 平滑视差区域的最大尺寸，以考虑其噪声斑点和无效。将其设
                # 置为0可禁用斑点过滤。否则，将其设置在50 - 200的范围内。
                speckleWindowSize=0,
                # 视差变化阈值，每个连接组件内的最大视差变化。如果你做斑点过滤，
                # 将参数设置为正值，它将被隐式乘以16.通常，1或2就足够好了
                speckleRange=2,
                # preFilterCap=63,
                preFilterCap=0,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )

            # stereoMatcher = cv2.StereoSGBM_create(
            #     # 最小视差
            #     minDisparity=0,
            #     # 视差范围，即最大视差值和最小视差值之差，必须是16的倍数。
            #     numDisparities=16*12,
            #     # 匹配块大小，大于1的奇数  
            #     blockSize=3,
            #     # P1, P2控制视差图的光滑度
            #     # 惩罚系数，一般：P1 = 8 * 通道数*SADWindowSize*SADWindowSize，
            #     # P2 = 4 * P
            #     P1=8 * 3 * window_size ** 2,
            #     # wsize default 3; 5; 7 for SGBM reduced size image; 
	        #     # 15 for SGBM full size image (1300px and above); 
            #     # 5 Works nicely
            #     P2=32 * 3 * window_size ** 2,
            #     # 左右视差图的最大容许差异（超过将被清零），默认为 - 1，
            #     # 即不执行左右视差检查
            #     disp12MaxDiff=1,
            #     # 视差唯一性百分比， 视差窗口范围内最低代价是次低代价的
            #     # (1 + uniquenessRatio / 100)倍时，最低代价对应的视差值才是
            #     # 该像素点的视差，否则该像素点的视差为 0，通常为5~15.
            #     uniquenessRatio=15,
            #     # 平滑视差区域的最大尺寸，以考虑其噪声斑点和无效。将其设
            #     # 置为0可禁用斑点过滤。否则，将其设置在50 - 200的范围内。
            #     speckleWindowSize=0,
            #     # 视差变化阈值，每个连接组件内的最大视差变化。如果你做斑点过滤，
            #     # 将参数设置为正值，它将被隐式乘以16.通常，1或2就足够好了
            #     speckleRange=2,
            #     # preFilterCap=63,
            #     preFilterCap=0,
            #     mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            # )

            # numOfDisparities = (image_depth_left_gray.shape[1] / 8 + 15) & -16
            # stereoMatcher = cv2.StereoSGBM_create(
            #     # 最小视差
            #     minDisparity=0,
            #     # 视差范围，即最大视差值和最小视差值之差，必须是16的倍数。
            #     numDisparities=numOfDisparities,
            #     # 匹配块大小，大于1的奇数  
            #     blockSize=0,
            #     # P1, P2控制视差图的光滑度
            #     # 惩罚系数，一般：P1 = 8 * 通道数*SADWindowSize*SADWindowSize，
            #     # P2 = 4 * P
            #     P1=0,
            #     # wsize default 3; 5; 7 for SGBM reduced size image; 
	        #     # 15 for SGBM full size image (1300px and above); 
            #     # 5 Works nicely
            #     P2=0,
            #     # 左右视差图的最大容许差异（超过将被清零），默认为 - 1，
            #     # 即不执行左右视差检查
            #     disp12MaxDiff=1,
            #     # 视差唯一性百分比， 视差窗口范围内最低代价是次低代价的
            #     # (1 + uniquenessRatio / 100)倍时，最低代价对应的视差值才是
            #     # 该像素点的视差，否则该像素点的视差为 0，通常为5~15.
            #     uniquenessRatio=10,
            #     # 平滑视差区域的最大尺寸，以考虑其噪声斑点和无效。将其设
            #     # 置为0可禁用斑点过滤。否则，将其设置在50 - 200的范围内。
            #     speckleWindowSize=100,
            #     # 视差变化阈值，每个连接组件内的最大视差变化。如果你做斑点过滤，
            #     # 将参数设置为正值，它将被隐式乘以16.通常，1或2就足够好了
            #     speckleRange=32,
            #     # preFilterCap=63,
            #     preFilterCap=0,
            #     mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            # )

            # compute disparity
            depth = stereoMatcher.compute(
                image_depth_left_gray,
                image_depth_right_gray
            ).astype(np.float32) / 16.0

            # depth = stereoMatcher.compute(
            #     image_depth_left_gray,
            #     image_depth_right_gray
            # ).astype(np.uint8)

            self.custom_imwrite(
                depth,
                (
                    self.ROOT_PATH
                    + r"\cam_depth\cam_depth_{}.png".format(count_remap)
                )
            )
            print("depth.dtype:", depth.dtype)

            # DEPTH_VISUALIZATION_SCALE = 550
            # DEPTH_VISUALIZATION_SCALE = 1
            # cv2.imshow("depth_stereoBM", depth / DEPTH_VISUALIZATION_SCALE)
            # cv2.imshow("depth_stereoSGBM", depth)

            from matplotlib import pyplot as plt
            plt.imshow(depth, 'gray')
            plt.show()

    # def stereo_video(self):
    def this_stereo_BM_video(self):
        (
            maps_x_left, maps_y_left,
            maps_x_right, maps_y_right
        ) = self.stereo_undistortion()

        cap_left = cv2.VideoCapture(1)
        cap_right = cv2.VideoCapture(0)

        CAMERA_WIDTH = 1280
        CAMERA_HEIGHT = 720

        cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        # video read
        while cap_left.isOpened() and cap_right.isOpened():
            ret_left, frame_rgb_left = cap_left.read()
            ret_right, frame_rgb_right = cap_right.read()

            frame_gray_left = cv2.cvtColor(
                frame_rgb_left,
                cv2.COLOR_BGR2GRAY
            )
            frame_rgb_right = cv2.cvtColor(
                frame_rgb_right,
                cv2.COLOR_BGR2GRAY
            )

            cv2.imshow("frame_rgb_left", frame_rgb_left)
            cv2.imshow("frame_rgb_right", frame_rgb_right)
            cv2.imshow("frame_gray_left", frame_rgb_left)
            cv2.imshow("frame_gray_right", frame_rgb_right)
            img_remap_left = cv2.remap(
                frame_gray_left,
                maps_x_left,
                maps_y_left,
                cv2.INTER_LANCZOS4
                # cv2.INTER_LINEAR
            )
            img_remap_right = cv2.remap(
                frame_rgb_right,
                maps_x_right,
                maps_y_right,
                cv2.INTER_LANCZOS4
                # cv2.INTER_LINEAR
            )
            cv2.imshow("img_remap_left", img_remap_left)
            cv2.imshow("img_remap_right", img_remap_right)
            (
                stereoRectify_roi1,
                stereoRectify_roi2
            ) = self.read_camera_info(
                "stereoRectify_roi1",
                "stereoRectify_roi2"
            )
            image_depth_left = np.zeros(
                img_remap_left.shape,
                np.uint8
            )
            image_depth_right = np.zeros(
                img_remap_right.shape,
                np.uint8
            )

            image_depth_left = img_remap_left.copy()
            image_depth_right = img_remap_right.copy()

            stereoMatcher = cv2.StereoBM_create()

            stereoMatcher.setMinDisparity(4)
            stereoMatcher.setNumDisparities(128)
            stereoMatcher.setBlockSize(21)
            stereoMatcher.setSpeckleRange(16)
            stereoMatcher.setSpeckleWindowSize(45)

            # image_depth_left_gray = cv2.cvtColor(
            #     image_depth_left,
            #     cv2.COLOR_BGR2GRAY
            # )
            # image_depth_right_gray = cv2.cvtColor(
            #     image_depth_right,
            #     cv2.COLOR_BGR2GRAY
            # )
            image_depth_left_gray = image_depth_left.copy()
            image_depth_right_gray = image_depth_right.copy()

            depth = stereoMatcher.compute(
                image_depth_left_gray,
                image_depth_right_gray
            )

            # DEPTH_VISUALIZATION_SCALE = 2048
            DEPTH_VISUALIZATION_SCALE = 400

            cv2.imshow("depth", depth / DEPTH_VISUALIZATION_SCALE)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap_left.release()
        cap_right.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    dcsf = Dual_Camera_Single_Frame()
    # dcsf.find_corners()
    # dcsf.stereo_calibrate()
    # dcsf.stereo_rectify()
    # dcsf.stereo_undistortion()
    dcsf.stereo_remapping() # 直接将新的图像放入 image_dichotomy，其他的文件不要动
    # dcsf.stereo_video()