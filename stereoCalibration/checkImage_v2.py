import cv2
import numpy as np
import glob

paths = glob.glob(r"C:\Users\lijin\OneDrive\srs-backup-code\stereoCalibration\backup\cam_depth\*.png")

# org_count = 0
for path in paths:
    # org_path = r"C:\Users\lijin\OneDrive\srs-backup-code\stereoCalibration\backup\image_dichotomy\cam_org_left_{}".format(count)
    # org_img = cv2.imread(r"C:\Users\lijin\OneDrive\srs-backup-code\stereoCalibration\backup\image_dichotomy\cam_org_left_{}".format(org_count), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(img, alpha=1), cv2.COLORMAP_JET)
    norm_img = (img-img.mean()) / (img.max() - img.min())

    print('mean:',img.mean(), ' max:', img.max(), ' min:', img.min())
    while True:
        # cv2.imshow('org', org_img)
        cv2.imshow('gray', norm_img)
        cv2.imshow('color',depth_colormap)
        # org_count += 1
        key = cv2.waitKey(100)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
