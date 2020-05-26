'''
@Author: your name
@Date: 2019-12-14 14:45:25
@LastEditTime: 2019-12-14 14:56:36
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \V_vsCode\.vscode\20191118_dualCam\videoSaveDual_01.py
'''
""" 从视频读取帧保存为图片"""
import cv2 as cv

ROOT_PATH = (
    r".vscode\20191118_dualCam\QTUI_DOC\QTUI_DOC_TEST12_OCR03_FT03"
)
VIDEO_PATH =(
    ROOT_PATH +
    r"\Video_Doc"
)
VIDEO_SAVE_PATH = (
    VIDEO_PATH +
    r"\video_save"
)
VIDEO_READ_PATH = (
    VIDEO_PATH +
    r"\video_frames"
)

Save_Path = (
    r"D:\P_project_SRS\face_cumstom_dual_cam_0506\cam_image"
)


# out = cv.VideoWriter(r'.\output_single.avi', fourcc, 20.0, (640, 480))
# out = cv.VideoWriter(r'C:\LeeSRSPrgoramFile\V_vsCode\.vscode\20191118_双目\output_single.avi', fourcc, 20.0, (640, 480))


# out = cv.VideoWriter(
#     VIDEO_SAVE_PATH +
#     r'\output_single.avi', fourcc, 20.0, (640, 480))

# out = cv.VideoWriter(r'C:\LeeSRSPrgoramFile\V_vsCode\.vscode\20191118_双目\output_single.avi', fourcc, 20.0, (640, 480))

def video_save():
    cap = cv.VideoCapture(0)

    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(
        VIDEO_SAVE_PATH +
        r'\output_single.avi', fourcc, 20.0, (640, 480)
    )
    # out = cv.VideoWriter(r'.\output_single.avi', fourcc, 20.0, (640, 480))
    # out = cv.VideoWriter(r'C:\LeeSRSPrgoramFile\V_vsCode\.vscode\20191118_双目\output_single.avi', fourcc, 20.0, (640, 480))
    
    image_count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = cv.flip(frame, 0)
    
            # write the flipped frame
            out.write(frame)
    
            cv.imshow('frame', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                print("q pressed")
                break

            if cv.waitKey(1) == 32 & 0xFF == ord(' '):
                print("image saved")
                cv.imwrite(Save_Path + "\{0}.jpg".format(image_count), frame)
        else:
            break
    
    # Release everything if job is finished
    cap.release()
    out.release()
    cv.destroyAllWindows()

def video_read():
    cap = cv.VideoCapture("003.mp4")#名为'003.mp4'的文件
    c=0                             #文件名从0开始
    while(1):
        # get a frame
        ret, frame = cap.read()
        # show a frame
        cv2.imshow("capture", frame)
        # cv2.imwrite('image/'+str(c) + '.jpg',frame) #存储为图像
        cv2.imwrite(VIDEO_PATH + str(c) + '.jpg', frame)  #存储q为图像
        c=c+1
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    video_save()