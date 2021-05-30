/*
 * @Author: joyce
 * @Date: 2021-05-29 14:22:39
 * @LastEditTime: 2021-05-30 11:32:35
 * @LastEditors: Please set LastEditors
 * @Description:: 
 */

// #include "depth_camera.h"
#include "depth_camera.cpp"

int main()
{
  rs2::pipeline pipeline;                           //创建数据管道
  RM_DepthCapture depth;                            //实例化类
  rs2::pipeline_profile profile = pipeline.start(); //start()函数返回数据管道的profile

  for (;;)
  {
    rs2::frameset frameset = pipeline.wait_for_frames(); //堵塞程序直到新的一帧捕获

    cv::Mat color_img = depth.getColorImage(pipeline, frameset); //获得彩色图
    cv::Mat depth_img = depth.getDepthImage(pipeline, frameset); //获得彩色深度图
    // cv::Mat depth_align_to_color=depth.getAlignedToColorFrame(pipeline,frameset);//深度图对齐到彩色图
    // cv::Mat color_align_to_depth=depth.getAlignedToDepthFrame(pipeline,frameset);//彩色图对齐到深度图

    cv::imshow("color_img", color_img);
    cv::imshow("depth_img", depth_img);
    // cv::imshow("depth_align_to_color",depth_align_to_color);
    // cv::imshow("color_align_to_depth",color_align_to_depth);
    // std::cout<<"distacne:"<<depth.getDistance(frameset)<<std::endl;//获得深度距离信息
    cv::waitKey(1);
  }
}