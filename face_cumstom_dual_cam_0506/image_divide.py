import os

import cv2
import numpy

ROOT = r"D:\P_project_SRS\face_cumstom_dual_cam_0506"
# IMG_FOLDER_PATH = r"D:\P_project_SRS\face_cumstom_dual_cam_0506\calib_img"

def divide_image():
    IMG_FOLDER_PATH = os.path.join(ROOT, "calib_img")
    img_path_list = os.listdir(IMG_FOLDER_PATH)

    for img_path in img_path_list:
        # print(img_path)
        img_path_abs = os.path.join(IMG_FOLDER_PATH, img_path)

        image_name = img_path.strip().split('.')[0].split('-')[0]
        
        image = cv2.imread(img_path_abs, cv2.IMREAD_UNCHANGED)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        height, width = image_gray.shape
        middle_width = int(width / 2)
        img_gray_left = image_gray[:, :middle_width]
        img_gray_right = image_gray[:, middle_width:]

        divide_img_dir_name = make_dir("divided_img")

        img_left_path = os.path.join(divide_img_dir_name, "{}_left.jpg".format(image_name))
        img_right_path = os.path.join(divide_img_dir_name, "{}_right.jpg".format(image_name))
        cv2.imwrite(img_left_path, img_gray_left)
        cv2.imwrite(img_right_path, img_gray_right)


def make_dir(dir_name):
    new_dir_name = os.path.join(ROOT, dir_name)
    if not os.path.exists(new_dir_name):
        os.mkdir(new_dir_name)

    return new_dir_name


if __name__ == "__main__":
    divide_image()