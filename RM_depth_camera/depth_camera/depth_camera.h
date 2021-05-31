/*
 * @Author: joyce
 * @Date: 2021-05-27 21:45:24
 * @LastEditTime: 2021-05-30 11:24:16
 * @LastEditors: Please set LastEditors
 * @Description:: 
 */
#include <iostream>
#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
class RM_DepthCapture
{
private:
  cv  ::Mat color_img_;              //彩色图
  cv  ::Mat depth_img_;              //彩色深度图
  cv  ::Mat aligned_to_depth_frame_; //深度图对齐到彩色图
  cv  ::Mat aligned_to_color_frame_; //深度图对齐到彩色图
  rs2 ::colorizer color_map_;        //帮助着色深度图像
  rs2 ::pipeline pipeline_;          //创建数据管道
  rs2 ::config config_;              //配置信息
  float distance_;                   //深度距离信息

public:
  RM_DepthCapture();                                                                   //构造函数
  ~RM_DepthCapture();                                                                  //析构函数
  float getDistance(rs2::frameset _frameset);                                          //获得深度距离信息
  cv ::Mat getColorImage(rs2::pipeline _pipeline, rs2::frameset _frameset);            //获得彩色图
  cv ::Mat getDepthImage(rs2::pipeline _pipeline, rs2::frameset _frameset);            //获得彩色深度图
  cv ::Mat getAlignedToColorFrame(rs2::pipeline _pipeline, rs2::frameset _frameset);   //深度图对齐到彩色图
  cv ::Mat getAlignedToDepthFrame(rs2::pipeline _pipeline, rs2::frameset _frameset);   //彩色图对齐到深度图
};
