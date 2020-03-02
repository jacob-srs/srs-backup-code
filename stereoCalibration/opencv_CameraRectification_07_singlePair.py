'''
@Author: your name
@Date: 2019-12-11 15:25:18
@LastEditTime: 2019-12-13 17:01:06
@LastEditors: Please set LastEditors
@Description: 这个代码固定，于20191212，尽量不要修改，实现立体相机的标度与校正
@FilePath: \V_vsCode\.vscode\20191118_dualCam\opencv_CameraRectification_07_singlePair.py
'''
# import os
import cv2
import glob
import numpy as np
import re


class Dual_Camera_Single_Frame(object):
    """ 对0，2，3，5，6，10，11双目图像对进行 CR """
    # 类常量
    ROOT_PATH = (
        r".vscode\20191118_dualCam\QTUI_DOC" +
        r"\QTUI_DOC_TEST12_OCR03_FIlETEST"
    )
    IMG_ORG_PATH = glob.glob(
        ROOT_PATH +
        r"\cam_org_2\cam_org_*.jpg"
    )
    # INFO_SAVE_PATH = (
    #     ROOT_PATH +
    #     r"\save_info"
    # )

    pattern_left_path = re.compile(r'[\w\s\\.]*left[\w\s\\.]*')
    pattern_right_path = re.compile(r'[\w\s\\.]*right[\w\s\\.]*')

    pattern_size = (7, 6)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    def __init__(self):
        super().__init__()

        self.obj_points = []  # 3d point in real world space
        self.img_left_corner_points = []  # 2d points in image plane.
        self.img_right_corner_points = []  # 2d points in image plane.
        self.image_size = None

        self.img_left_path_list = [None] * 200
        self.img_right_path_list = [None] * 200

        self.mtx_left = None
        self.dist_left = None
        self.rvecs_left = None
        self.tvecs_left = None

        self.mtx_right = None
        self.dist_right = None
        self.rvecs_right = None
        self.tvecs_right = None

        self.stereoCalibrate_cameraMatrix1 = None
        self.stereoCalibrate_distCoeffs1 = None
        self.stereoCalibrate_cameraMatrix2 = None
        self.stereoCalibrate_distCoeffs2 = None
        self.stereoCalibrate_R = None
        self.stereoCalibrate_T = None

        self.stereoRectify_R1 = None
        self.stereoRectify_R2 = None
        self.stereoRectify_P1 = None
        self.stereoRectify_P2 = None
        self.stereoRectify_Q = None
        self.stereoRectify_roi1 = None
        self.stereoRectify_roi2 = None

    def pair_dual_images(self):
        """ 对采集的双目图像按名称进行匹配 """
        for path in self.IMG_ORG_PATH:
            image_name = path.split('\\')[-1]
            image_direction = image_name.split('_')[-2]
            image_index = image_name.split('_')[-1].split('.')[0]

            if image_direction == "left":
                self.img_left_path_list[int(image_index)] = path
                # print("\nleft_path::", path)  # 没问题，没 None
            elif image_direction == "right":
                self.img_right_path_list[int(image_index)] = path

    def find_corners(self):

        """ 对双目图像采集的棋盘标定格进行角点查找
        只保存匹配图像都有角点的图像对点坐标, 这个函数得到
        1. img_left_corner_point
        2. img_right_corner_point
        3. obj_points"""
        self.pair_dual_images()

        cornerSubPix_criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )

        img_combined_path_list = zip(
            self.img_left_path_list,  # 200 个，有 None
            self.img_right_path_list  # 同上
        )

        count = 0
        count_both_found = 0
        for image_path in img_combined_path_list:
            """ 读取双目图像，找每个图像的角点，并且将都有角点的图像对角点存储 """
            # print("{}\n{}\n" .format(item[0], item[1]))
            print("{}\n{}\n" .format(image_path[0], image_path[1]))

            img_left_path = None
            img_right_path = None
            if image_path[0] and image_path[1] is not None:
                img_left_path = image_path[0]
                img_right_path = image_path[1]
                # print("{}\n{}\n" .format(img_left_path, img_right_path))
            else:
                break

            left_img = cv2.imread(img_left_path)  # 保存在同一个文件夹下
            right_img = cv2.imread(img_right_path)  # 保存在同一个文件夹下
            left_img_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_img_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

            # cv2.imshow("left img {}" .format(count), left_img)
            # cv2.imshow("right img {}" .format(count), right_img)

            if count == 0:  # 只保存一次就够了
                self.image_size = left_img_gray.shape  # 这个没存
                print("\nimage_size_type::{}\nimgage_size::{}" .format(
                    type(self.image_size), self.image_size
                ))
            else:
                pass

            # findChessboardCorners() for dual camera
            # find_chessboard_flags = (
            #     cv2.CALIB_CB_ADAPTIVE_THRESH |
            #     cv2.CALIB_CB_NORMALIZE_IMAGE |
            #     cv2.CALIB_CB_FAST_CHECK
            # )

            find_chessboard_flags = None

            left_found, left_corners = cv2.findChessboardCorners(
                left_img_gray,
                self.pattern_size,  # 全局定义 pattern_size
                find_chessboard_flags
            )
            right_found, right_corners = cv2.findChessboardCorners(
                right_img_gray,
                self.pattern_size,
                find_chessboard_flags
            )

            # cornerSubPix() for dual camera
            if left_found:
                left_corners_2 = cv2.cornerSubPix(
                    left_img_gray,
                    left_corners,
                    (11, 11),
                    (-1, -1),
                    cornerSubPix_criteria
                )
            if right_found:
                right_corners_2 = cv2.cornerSubPix(
                    right_img_gray,
                    right_corners,
                    (11, 11),
                    (-1, -1),
                    cornerSubPix_criteria
                )

            # 这里不用放 drawChessboardCOrners,因为会在两图不同时有
            # 角点的情况下绘图

            # cv2.imshow("left_img_corner::{}".format(count), left_img)
            # cv2.imshow("right_img_corner::{}".format(count), right_img)
            # count_both_found = 0
            if left_found and right_found:
                count_both_found += 1
                self.img_left_corner_points.append(left_corners)
                self.img_right_corner_points.append(right_corners)
                self.obj_points.append(self.objp)  # 定义 obj_points

                # cv2.imshow("left", left_img)
                cv2.drawChessboardCorners(
                    left_img,
                    self.pattern_size,
                    left_corners_2,
                    left_found
                )
                cv2.drawChessboardCorners(
                    right_img,
                    self.pattern_size,
                    right_corners_2,
                    right_found
                )

            # cv2.imshow("left chess {}" .format(count), left_img)
            # cv2.imshow("right chess {}" .format(count), right_img)

            count += 1
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        print("count_both_found::", count_both_found)

    # 先不管 stereo，重新单目
    def stereo_calibrate(self):
        """ 立体相机标度步骤
        1. calibrateCamera() --> left
            in--> obj_points
            in--> img_left_corner_points
            in--> image_size
            out--> mtx_left, dist_left
        2. calibrateCamera() --> right
            in--> obj_points
            in--> img_right_corner_points
            in--> image_size
            out--> mtx_left, dist_left
        3. stereoCalibrate()
            in--> obj_points, i_l_c_p, i_r_c_p
            in--> mtx_left, mtx_right, dist_left, dist_right
            in--> img.size(w, h), None, None, None, None,
            flag, criteria
            out--> rvecs, tvecs"""
        height, width = self.image_size[:2]

        # calibrateCamera() left
        (
            ret_calib_left,
            mtx_left,
            dist_left,
            _,  # rvecs_left
            _  # tvecs_left
        ) = cv2.calibrateCamera(
            self.obj_points,
            self.img_left_corner_points,
            (width, height),
            None,
            None
        )
        print("\nmtx_left=\n", mtx_left)
        print("\ndist_left=\n", dist_left)

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
            self.obj_points,
            self.img_right_corner_points,
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
            self.obj_points,
            self.img_left_corner_points,
            self.img_right_corner_points,
            mtx_left,
            dist_left,
            mtx_right,
            dist_right,
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

        # 赋值给实例变量
        self.mtx_left = mtx_left
        self.dist_left = dist_left

        self.mtx_right = mtx_right
        self.dist_right = dist_right

        self.stereoCalibrate_R = stereoCalibrate_R
        self.stereoCalibrate_T = stereoCalibrate_T

    # 在这个函数 修改 rectify_scale 的值
    def stereo_rectify(self):
        print("\nin stereo_rectify")
        height, width = self.image_size[:2]
        rectify_scale = 0  # 0=full crop, 1=no crop 这个参数太重要了
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
            self.mtx_left,
            self.dist_left,
            self.mtx_right,
            self.dist_right,
            (width, height),  # 这个必须是tuple
            self.stereoCalibrate_R,
            self.stereoCalibrate_T,
            None,  # R1:
            None,  # R2:
            None,  # P1:
            None,  # P2:
            None,  # Q:
            cv2.CALIB_ZERO_DISPARITY,
            alpha=rectify_scale
        )

        # 实例变量赋值，全局化
        self.stereoRectify_R1 = stereoRectify_R1
        self.stereoRectify_R2 = stereoRectify_R2
        self.stereoRectify_P1 = stereoRectify_P1
        self.stereoRectify_P2 = stereoRectify_P2
        self.stereoRectify_Q = stereoRectify_Q
        self.stereoRectify_roi1 = stereoRectify_roi1
        self.stereoRectify_roi2 = stereoRectify_roi2
        print("\nstereoRectify_R1::\n", stereoRectify_R1)
        print("\nstereoRectify_R2::\n", stereoRectify_R2)
        print("\nstereoRectify_P1::\n", stereoRectify_P1)
        print("\nstereoRectify_P2::\n", stereoRectify_P2)
        print("\nstereoRectify_Q::\n", stereoRectify_Q)
        print("\nstereoRectify_roi1::\n", stereoRectify_roi1)
        print("\nstereoRectify_roi2::\n", stereoRectify_roi2)

    def stereo_undistortion(self):
        height, width = self.image_size  # 480, 640
        left_maps_x, left_maps_y = cv2.initUndistortRectifyMap(
            self.mtx_left,
            self.dist_left,
            self.stereoRectify_R1,
            self.stereoRectify_P1,
            (width, height),
            cv2.CV_32FC2
        )
        right_maps_x, right_maps_y = cv2.initUndistortRectifyMap(
            self.mtx_right,
            self.dist_left,
            self.stereoRectify_R2,
            self.stereoRectify_P2,
            (width, height),
            cv2.CV_32FC2
        )

        return left_maps_x, left_maps_y, right_maps_x, right_maps_y

    def stereo_remapping(self):
        (
            left_maps_x, left_maps_y,
            right_maps_x, right_maps_y
        ) = self.stereo_undistortion()

        # 读取图像对
        self.pair_dual_images()

        img_combined_path_list = zip(
            self.img_left_path_list,  # 200 个，有 None
            self.img_right_path_list  # 同上
        )

        # count = 0
        count_remap = 0
        for image_path in img_combined_path_list:
            """ 读取双目图像，找每个图像的角点，并且将都有角点的图像对角点存储 """
            # print("{}\n{}\n" .format(item[0], item[1]))
            print("{}\n{}\n" .format(image_path[0], image_path[1]))

            img_left_path = None
            img_right_path = None
            if image_path[0] and image_path[1] is not None:
                img_left_path = image_path[0]
                img_right_path = image_path[1]
                # print("{}\n{}\n" .format(img_left_path, img_right_path))
            else:
                break

            left_img = cv2.imread(img_left_path)
            right_img = cv2.imread(img_right_path)

            left_img_gray = cv2.cvtColor(
                left_img,
                cv2.COLOR_BGR2GRAY)
            right_img_gray = cv2.cvtColor(
                right_img,
                cv2.COLOR_BGR2GRAY)

            # remap each image pair
            left_img_remap = cv2.remap(
                left_img_gray,
                left_maps_x,
                left_maps_y,
                cv2.INTER_LANCZOS4
                # cv2.INTER_LINEAR
            )
            right_img_remap = cv2.remap(
                right_img_gray,
                right_maps_x,
                right_maps_y,
                cv2.INTER_LANCZOS4
                # cv2.INTER_LINEAR
            )

            # 将 gray 2 bgr
            left_img_remap = cv2.cvtColor(left_img_remap, cv2.COLOR_GRAY2BGR)
            right_img_remap = cv2.cvtColor(right_img_remap, cv2.COLOR_GRAY2BGR)

            # 对 remap 图像进行基本图形绘制
            # pt1, pt2 是可变的
            # stereoRectify_roi1,  # green rectangles roi left
            # stereoRectify_roi2  # green rectangles roi right
            left_img_remap = cv2.rectangle(
                left_img_remap,
                (self.stereoRectify_roi1[0], self.stereoRectify_roi1[1]),
                (self.stereoRectify_roi1[2], self.stereoRectify_roi1[3]),
                (0, 255, 0),
                2
            )
            right_img_remap = cv2.rectangle(
                right_img_remap,
                (self.stereoRectify_roi2[0], self.stereoRectify_roi2[1]),
                (self.stereoRectify_roi2[2], self.stereoRectify_roi2[3]),
                (0, 255, 0),
                2
            )

            # 获得 left，right remap 尺寸, row=height=480, col=width=640
            # print(
            #     "\nright_img_remap shape::{}\ttype::{}"
            #     .format(
            #         right_img_remap.shape, type(right_img_remap)
            #     )
            # )
            print(
                "\nright_img_remap shape::{}\ttype::{}"
                .format(
                    right_img_remap.shape, right_img_remap.dtype
                )
            )
            # output right_img_remap shape::(480, 640)       type::uint8
            # print("\nleft_img_remap shape::\ttype::{}", left_img_remap.shape)
            # 创建拼接图像
            img_remap_merge = np.zeros(
                (
                    left_img_remap.shape[0],
                    left_img_remap.shape[1] + right_img_remap.shape[1],
                    3
                ),
                np.uint8
            )
            img_remap_merge.fill(255)  # 填色 white
            # 使用 boardcast 拷贝图像
            img_remap_merge[:, :640, :] = left_img_remap
            img_remap_merge[:, 640:, :] = right_img_remap
            # 对 img_remap_merge 进行 基本图形绘制
            # 绘制 46 条线，0没有，48不包括
            loop_index = img_remap_merge.shape[0]
            for i in range(int(loop_index / 10)):
                if i == 0:
                    continue

                cv2.line(
                    img_remap_merge,
                    (0, i * 10),  # pt1
                    (img_remap_merge.shape[1], i * 10),  # pt2
                    (0, 0, 255),
                    1
                )
            cv2.imshow("img_remap_merge", img_remap_merge)
            # print("\nimg_remap_merge shape::", img_remap_merge.shape)

            # 还是没有得到结果
            cv2.imshow(
                "left_img{}"
                .format(count_remap), left_img_gray)
            cv2.imshow(
                "right_img{}"
                .format(count_remap), right_img_gray)
            cv2.imshow(
                "left_img_remap_{}"
                .format(count_remap), left_img_remap)
            cv2.imshow(
                "right_img_remap_{}"
                .format(count_remap), right_img_remap)

            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    dcsf = Dual_Camera_Single_Frame()
    dcsf.find_corners()
    dcsf.stereo_calibrate()
    dcsf.stereo_rectify()
    dcsf.stereo_undistortion()
    dcsf.stereo_remapping()
