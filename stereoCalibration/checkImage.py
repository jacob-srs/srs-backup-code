import cv2
import numpy as np

img = cv2.imread(r'C:\Users\hpdre\Downloads\cam_depth\cam_depth_0.png', cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(img, alpha=1), cv2.COLORMAP_JET)
norm_img = (img-img.mean()) / (img.max() - img.min())

print('mean:',img.mean(), ' max:', img.max(), ' min:', img.min())
while True:
    cv2.imshow('gray', norm_img)
    cv2.imshow('color',depth_colormap)
    key = cv2.waitKey(10)
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        break
