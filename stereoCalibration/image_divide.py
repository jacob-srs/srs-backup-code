""" 程序说明
@ jacob-srs
@ 20200222
@ 这个程序用来将整体图像左右分离
@ 由于保存的图像的特殊性--左右颠倒，将左半边图像存储到右镜头，右半边左镜头
@ 源图像-cam_org_#, 左半边图像-cam_right_#, 右半边图像-cam_left_#
"""


import os
import cv2
import glob
# import numpy as np


class Image_Dichotomy(object):
    def __init__(self):
        pass

    def __test_read_and_show_image_orign(self):
        """ 私有方法
        读取并显示文件夹图像 """
        ROOT_PATH = (
            r"C:\Users\lijin\OneDrive\srs-backup-code\stereoCalibration"
        )
        IMAGE_ORG_PATH = (
            ROOT_PATH
            + r"\image_origin"
        )
        cam_org_paths = glob.glob(
            IMAGE_ORG_PATH
            + r"\cam_org_*.PNG"
        )
        cam_org_count = 0
        for cam_org_path in cam_org_paths:
            cam_org = cv2.imread(cam_org_path, cv2.IMREAD_UNCHANGED)
            cv2.imshow("cam_org_{0}".format(cam_org_count), cam_org)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cam_org_count += 1
        print("\ntatal image number::{0}".format(cam_org_count))

    def __test_fetch_cam_org_path(self):
        """ 私有方法
        读取cam_org_path并返回 """
        ROOT_PATH = (
            r"C:\Users\lijin\OneDrive\srs-backup-code\stereoCalibration"
        )
        IMAGE_ORG_PATH = (
            ROOT_PATH
            + r"\image_origin"
        )
        cam_org_paths = glob.glob(
            IMAGE_ORG_PATH
            + r"\cam_org_*.PNG"
        )
        return cam_org_paths

    def __test_create_directory(self):
        """私有方法
        在指定目录创建目标文件夹 """
        ROOT_PATH = (
            r"C:\Users\lijin\OneDrive\srs-backup-code\stereoCalibration"
        )
        """ 创建目录
        1. 单层 os.mkdir(path)
        2. 多层 os.makedirs(path) """
        try:
            os.mkdir(ROOT_PATH + r"\image_dichotomy")
        except OSError:
            os.rmdir(ROOT_PATH + r"\image_dichotomy")

    def test_functions(self):
        """ 私有方法调用区 """
        # self.__test_read_and_show_image_orign()
        path_lsit = self.__test_fetch_cam_org_path()
        for path in path_lsit:
            print(path)
        # self.__test_create_directory()

    def cam_org_bisect(self):
        """ 将cam_org图像左右均分，并更正分割后的图像的方向 """
        # 确定图像顺序
        pass


if __name__ == "__main__":
    image_dichotomy = Image_Dichotomy()
    image_dichotomy.test_functions()
