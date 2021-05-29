/*
 * @Author: joyce
 * @Date: 2021-05-27 21:45:24
 * @LastEditTime: 2021-05-29 14:30:07
 * @LastEditors: Please set LastEditors
 * @Description:: 
 */
#include <iostream>
#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

class RM_DepthCapture
{
private:
  cv ::Mat color_img_;                                //彩色图
  cv ::Mat depth_img_;                                //彩色深度图
  cv ::Mat aligned_depth_frame_;                      //深度图对齐到彩色图
  rs2 ::colorizer color_map_;                         //帮助着色深度图像
  rs2 ::pipeline pipeline_;                           //创建数据管道
  rs2 ::config config_;
  float distance_;
public:
  RM_DepthCapture();   //构造函数
  ~RM_DepthCapture();  //析构函数
  float getDistance(rs2::frameset _frameset); //获得相机中心点到最近物体的距离
  cv ::Mat getColorImg(rs2::pipeline _pipeline,rs2::frameset _frameset);          //获得彩色图
  cv ::Mat getDepthImg(rs2::pipeline _pipeline,rs2::frameset _frameset);          //获得彩色深度图
  cv ::Mat getAlignedDepthFrame(rs2::pipeline _pipeline,rs2::frameset _frameset); //获得对齐后的彩色图

};
