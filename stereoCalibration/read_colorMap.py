import cv2
import glob

# path = r"C:\Users\lijin\OneDrive\srs-backup-code\stereoCalibration\cam_depth\cam_depth_0.png"
# img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
# img_colorMap = cv2.applyColorMap(img_gray, cv2.COLORMAP_JET)
# cv2.imshow("img_colorMap", img_colorMap)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def read_sequence_color_map():
    """ 读取连续深度图colorMap """
    # mindvision
    PATH_MIND_VISION = r"C:\Users\lijin\OneDrive\srs-backup-code\stereoCalibration\cam_depth\cam_depth_*.png"
    # test_stereo
    PATH_TEST_STEREO = r"C:\LeeSRSPrgoramFile\V_vsCode\.vscode\20191118_dualCam\QTUI_DOC\DUEL_CAM_DOC\cam_depth\cam_depth_*.png"
    # noise
    PATH_NOISE = r"C:\LeeSRSPrgoramFile\V_vsCode\.vscode\20191217_depthImage_analysis\doc_depth_images\*.png"

    PATH = PATH_NOISE
    PATH = PATH_MIND_VISION
    # PATH = PATH_TEST_STEREO

    image_depth_paths = glob.glob(PATH)
    for image_depth_path in image_depth_paths:
        img_gray = cv2.imread(image_depth_path, cv2.IMREAD_GRAYSCALE)
        # img_colorMap = cv2.applyColorMap(img_gray, cv2.COLORMAP_JET)
        img_colorMap = cv2.applyColorMap(
            cv2.convertScaleAbs(img_gray, alpha=1), cv2.COLORMAP_JET)        
        cv2.imshow("img_colorMap", img_colorMap)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    read_sequence_color_map()