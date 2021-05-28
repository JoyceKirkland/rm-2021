/*
 * @Author: joyce
 * @Date: 2021-05-27 21:45:24
 * @LastEditTime: 2021-05-28 12:30:15
 * @LastEditors: Please set LastEditors
 * @Description:: 
 */
#include <iostream>
#include <librealsense2/rs.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
using namespace rs2;

class RM_DepthCapture
{
private:
  Mat color_img;              //彩色图
  Mat depth_img;              //彩色深度图
  Mat aligned_depth_frame;    //深度图对齐到彩色图
public:
  RM_DepthCapture();          //构造函数
  ~RM_DepthCapture();         //析构函数
  Mat getColorImg();          //获得彩色图
  Mat getDepthImg();          //获得彩色深度图
  Mat getAlignedDepthFrame(); //获得对齐后的彩色图
  float getDistance();        //获得相机中心点到最近物体的距离
 
};

