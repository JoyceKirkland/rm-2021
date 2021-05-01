'''
Author: joyce
Date: 2021-04-09 21:32:45
LastEditTime: 2021-05-01 16:08:36
LastEditors: Please set LastEditors
Description:: 
'''
#-------------------------------------#
#   调用摄像头或者视频进行检测
#   调用摄像头直接运行即可
#   调用视频可以将cv2.VideoCapture()指定路径
#   视频的保存并不难，可以百度一下看看
#-------------------------------------#
import time

import cv2
import numpy as np
import tensorflow as tf
import pyrealsense2 as rs
import cv2
import crc8
from PIL import Image


from yolo import YOLO

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

yolo = YOLO()
#-------------------------------------#
#   调用摄像头
#   capture=cv2.VideoCapture("1.mp4")
#-------------------------------------#
pipeline=rs.pipeline()
config=rs.config()
# pipeline_wrapper=rs.pipeline_wrapper(pipeline)
# pipeline_profile=config.resolve(pipeline_wrapper)
# device=pipeline_profile.get_device()
# device_product_line = str(device.get_info(rs.camera_info.product_line))
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# if device_product_line == 'L500':
    # config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
# else:
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

capture=cv2.VideoCapture(6)
fps = 0.0
b=1
flag=0
cv2.namedWindow("1",cv2.WINDOW_FULLSCREEN)
# cv2.namedWindow("2",cv2.WINDOW_NORMAL)
cv2.setWindowProperty("1",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
# cv2.setWindowProperty("2",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
try:
    while(True):
        
        #______________深度相机_____________________________#
        t1=time.time()
        ref,frame=capture.read()
        # flag=0
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame=frames.get_depth_frame()
        depth_w=depth_frame.get_width()
        depth_h=depth_frame.get_height()
        distacne_center=100*depth_frame.get_distance(int(depth_w/2),int(depth_h/2))
        # print('distance:%.3f'%(distacne_center))
        if not color_frame:
            continue
        color_image = np.asanyarray(color_frame.get_data())
        # color_image=cv2.cvtColor(color_image,cv2.COLOR_BGR2RGB)
        color_image = Image.fromarray(np.uint8(color_image))
        color_image= np.array(yolo.detect_image(color_image,distacne_center))
        # print('x:%.2f'%x)
        # print('y:%.2f'%y)
        color_image = cv2.cvtColor(color_image,cv2.COLOR_RGB2BGR)
        color_image=cv2.resize(color_image,(2560,1440),cv2.INTER_AREA)#放大显示，帧率会掉
        frame=cv2.resize(frame,(2560,1440),cv2.INTER_AREA)#放大显示，帧率会掉
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        # print("fps= %.2f"%(fps))
        color_image = cv2.putText(color_image, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        #color_colormap_dim = color_image.shape
        # cv2.imshow('1', color_image)
        # frame_change=color_image
        # cv2.imshow("RealSense",color_image)
        c= cv2.waitKey(30) & 0xff
        # flag=0 
        if b==1:
            # cv2.imshow("video",frame)
            
            cv2.imshow("1",color_image)
            # print(flag)
            
        elif b==2:
            
            cv2.imshow("1",frame)
            # print(flag)
            
        #__按1深度相机，按2为普通小黑____________________#    
        if c==50:#按键数字2
            b=2
            
        elif c==49:#按键数字1
            b=1
        #————————————————————————————————#

        #__按2直接切换______________________#      
        # if c==50:
        #     b=2
        #     flag=flag+1
        #     if(flag%2==0):
        #         b=1
        #————————————————————————————————#

        if c==27:
            capture.release()
            break
        
        #——————————————————————————————————————————————#

    # while(True):
#         t1 = time.time()
#         # 读取某一帧
        # ref,frame=capture.read()
#         # 格式转变，BGRtoRGB
#         frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#         # 转变成Image
#         frame = Image.fromarray(np.uint8(frame))

#         # 进行检测
#         frame = np.array(yolo.detect_image(frame))

#         # RGBtoBGR满足opencv显示格式
#         frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        
#         fps  = ( fps + (1./(time.time()-t1)) ) / 2
#         print("fps= %.2f"%(fps))
        # frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
#         cv2.imshow("video",frame)
#         c= cv2.waitKey(30) & 0xff 
#         if c==27:
#             capture.release()
#             break
finally:
    pipeline.stop()
