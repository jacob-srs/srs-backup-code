""" 程序说明
@ jacob-srs
@ 20200222
@ 这个程序用来将整体图像左右分离
@ 由于保存的图像的特殊性--左右颠倒，将左半边图像存储到右镜头，右半边左镜头
@ 源图像-cam_org_#, 左半边图像-cam_right_#, 右半边图像-cam_left_#
"""


# pylint: disable=W0105  # 用来抑制 pyline 警告


import os
import sys
import glob
import cv2
import numpy as np


class Image_Dichotomy(object):
    """ 对 mindvision 原始图像进行左右分割 """
    ROOT_PATH = (
            # r"C:\Users\lijin\OneDrive\srs-backup-code\stereoCalibration"
            r"C:\Users\THINKPAD\OneDrive\GitSpace\srs-backup-code-master\srs-backup-code-master\stereoCalibration"
        )

    def __init__(self):
        pass

    @ classmethod
    def __test_read_and_show_image_orign(cls):
        """ 私有方法
        读取并显示文件夹图像 """
        # ROOT_PATH = (
        #     # r"C:\Users\lijin\OneDrive\srs-backup-code\stereoCalibration"
        #     r"C:\Users\THINKPAD\OneDrive\GitSpace\srs-backup-code-master\srs-backup-code-master\stereoCalibration"
        # )
        IMAGE_ORG_PATH = (
            cls.ROOT_PATH
            + r"\image_origin"
        )
        cam_org_paths = glob.glob(
            IMAGE_ORG_PATH
            + r"\cam_org_*.PNG"
        )
        cam_org_count = 0
        for cam_org_path in cam_org_paths:
            cam_org = cv2.imread(cam_org_path, cv2.IMREAD_UNCHANGED)
            if not cam_org.data:  # 判断是否读取图像成功
                print("\n..__test_read_and_show_image_origin()::Error loading image")
                sys.exit()
            cv2.imshow("cam_org_{0}".format(cam_org_count), cam_org)
            print(cam_org_path)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cam_org_count += 1
        print("\ntatal image number::{0}".format(cam_org_count))

    @ classmethod
    def __test_fetch_cam_org_path(cls):
        """ 私有方法
        读取cam_org_path并返回 """
        # ROOT_PATH = (
        #     # r"C:\Users\lijin\OneDrive\srs-backup-code\stereoCalibration"
        #     r"C:\Users\THINKPAD\OneDrive\GitSpace\srs-backup-code-master\srs-backup-code-master\stereoCalibration"
        # )
        IMAGE_ORG_PATH = (
            cls.ROOT_PATH
            + r"\image_origin"
        )
        cam_org_paths = glob.glob(
            IMAGE_ORG_PATH
            + r"\cam_org_*.PNG"
        )
        return cam_org_paths

    @ classmethod
    def __test_create_directory(cls):
        """私有方法
        在指定目录创建目标文件夹 """
        # ROOT_PATH = (
        #     # r"C:\Users\lijin\OneDrive\srs-backup-code\stereoCalibration"
        #     r"C:\Users\THINKPAD\OneDrive\GitSpace\srs-backup-code-master\srs-backup-code-master\stereoCalibration"
        # )
        """ 创建目录
        @ 单层 os.mkdir(path)
        @ 多层 os.makedirs(path) """
        try:
            os.mkdir(cls.ROOT_PATH + r"\image_dichotomy")
        except OSError:
            # os.rmdir(cls.ROOT_PATH + r"\image_dichotomy")  # 删除空目录
            os.removedirs(cls.ROOT_PATH + r"\image_dichotomy\*")

    @ classmethod
    def __test_ascending_path(cls):
        """ 私有方法
        将cam_org路径按升序排列 """
        path_list = cls.__test_fetch_cam_org_path()
        # for path in path_list:
        #     print(path)
        # 定义一个可以对字符串list排序的嵌套函数
        def list_sort_ascending(_target_list):  # 使用一个下划线抑制 pylint:unused argument
            img_index = 0
            temp_list = []
            for path in path_list:
                img_index = path.split("\\")[-1].split(".")[-2].split("_")[-1]
                temp_list.insert(int(img_index), path)  # 在指定索引位置插入元素
            return temp_list
        # print("+-"*63)
        ascending_list = list_sort_ascending(path_list)
        # for path in ascending_list:
        #     print(path)
        return ascending_list

    @ classmethod
    def __test_image_info(cls):
        """ 私有方法
        获得图像信息 """
        # 确定图像顺序
        ascending_path = cls.__test_ascending_path()
        for cam_org_path_sort in ascending_path:
            cam_org_sort = cv2.imread(cam_org_path_sort, cv2.IMREAD_UNCHANGED)
            print("{0}...{1}".format(cam_org_sort.shape, cam_org_sort.dtype))

    def test_functions(self):
        """ 私有方法调用区 """
        # self.__test_read_and_show_image_orign()
        # self.__test_ascending_path()
        # self.__test_create_directory()
        # self.__test_image_info()

    def this_create_directory(self):
        """ 实例方法调用类方法区 """
        self.__test_create_directory()

    def cam_org_bisect(self):
        """ 将cam_org图像左右均分，并更正分割后的图像的方向 """
        # 确定图像顺序
        ascending_path = self.__test_ascending_path()
        """ cam_org_bisect() inner test block """
        # 打印col/2数据
        def print_col_by_2(_col_info):
            bisect_pixel = int(_col_info / 2)
            print(bisect_pixel)
        # 显示图像
        def show_image(
            _direction_1=None, _direction_2=None,
            _image_1=None, _image_2=None
        ):
            condition_1 = np.logical_and(_direction_1 is not None, _image_1 is not None).all()
            condition_2 = np.logical_and(_direction_2 is not None, _image_2 is not None).all()
            condition_show_image_left = np.logical_and(condition_1, not condition_2).all()
            condition_show_image_right = np.logical_and(not condition_1, condition_2).all()
            condition_show_image_pair = np.logical_and(condition_1, condition_2)
            if (condition_show_image_left):
                cv2.imshow("cam_org_{0}".format(_direction_1), _image_1)
            elif (condition_show_image_right):
                cv2.imshow("cam_org_{0}".format(_direction_2), _image_2)
            elif (condition_show_image_pair):
                cv2.imshow("cam_org_{0}".format(_direction_1), _image_1)
                cv2.imshow("cam_org_{0}".format(_direction_2), _image_2)
            else:
                pass
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        """ cam_org_bisect() inner test block """
        self.this_create_directory()
        for cam_org_path_sort in ascending_path:
            img_index = cam_org_path_sort.split("\\")[-1].split(".")[-2].split("_")[-1]
            cam_org_sort = cv2.imread(cam_org_path_sort, cv2.IMREAD_UNCHANGED)
            if not cam_org_sort.data:  # 判断是否读取图像成功
                print("\n..cam_org_bisect()::Error loading image")
                sys.exit()    
            bisect_pixel = int(cam_org_sort.shape[1] / 2)
            cam_org_left = cam_org_sort[:, bisect_pixel:]
            cam_org_right = cam_org_sort[:, :bisect_pixel]
            # show_image(
            #     "left", "right",
            #     cam_org_left, cam_org_right
            # )
            img_sequence_left_file = (
                self.ROOT_PATH
                + r"\image_dichotomy\cam_org_left_{0}.png".format(img_index),
            )
            img_sequence_right_file = (
                self.ROOT_PATH
                + r"\image_dichotomy\cam_org_right_{0}.png".format(img_index),
            )
            if os.path.exists(img_sequence_left_file):
                os.remove(img_sequence_left_file)
            elif os.path.exists(img_sequence_right_file):
                os.remove(img_sequence_right_file)
            # cv2.imwrite(
            #     self.ROOT_PATH
            #     + r"\image_dichotomy\cam_org_left_{0}.png".format(img_index),
            #     cam_org_left
            # )
            # cv2.imwrite(
            #     self.ROOT_PATH
            #     + r"\image_dichotomy\cam_org_right_{0}.png".format(img_index),
            #     cam_org_right
            # )



if __name__ == "__main__":
    image_dichotomy = Image_Dichotomy()
    # image_dichotomy.test_functions()
    image_dichotomy.cam_org_bisect()
