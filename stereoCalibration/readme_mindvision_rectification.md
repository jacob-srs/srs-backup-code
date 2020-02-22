
# mindvision双目相机标定程序
# 版本：v1.0
# 时间：20200222
---
#### footnotes
当前项目包含两个.py文件  
1. image_divide.py  
这个程序用来将mindvision双目相机采集的原始图像分割成左右图像  
由于相机底层程序问题，原始图像左右相机图像颠倒，所以分割后的  
图像要注意方向  

2. opencv_CameraRectification_07_singlePair.py  
这个程序用来读取左右相机图像并计算相机的参数

