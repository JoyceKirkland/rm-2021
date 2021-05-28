/*
 * @Author: joyce
 * @Date: 2021-05-27 21:45:36
 * @LastEditTime: 2021-05-28 12:30:01
 * @LastEditors: Please set LastEditors
 * @Description:: 
 */


#include "depth_camera.h"

RM_DepthCapture::RM_DepthCapture()
{
  colorizer        colorizer;//帮助着色深度图像
  pipeline         pipeline;//创建数据管道
  pipeline_profile profile= pipeline.start();//start()函数返回数据管道的profile
}

RM_DepthCapture::~RM_DepthCapture(){}

Mat RM_DepthCapture::getColorImg()//获得彩色图
{
  RM_DepthCapture();
}          


