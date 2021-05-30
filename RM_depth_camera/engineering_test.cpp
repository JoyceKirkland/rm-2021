/*
 * @Author: joyce
 * @Date: 2021-05-30 15:18:31
 * @LastEditTime: 2021-05-30 16:40:23
 * @LastEditors: Please set LastEditors
 * @Description:: 
 */

#include "rm_engineering/rm_engineering.cpp"
// #include <opencv2/dnn.hpp>
int main()
{
  rs2::pipeline pipeline;                           //创建数据管道
  // RM_DepthCapture depth;                            //实例化类
  RM_Engineering engineering;
  rs2::pipeline_profile profile = pipeline.start(); //start()函数返回数据管道的profile

  for (;;)
  {
    rs2::frameset frameset = pipeline.wait_for_frames(); //堵塞程序直到新的一帧捕获

    // engineering.getDistance(frameset);
    // cv::imshow("color_img", color_img);
    // cv::imshow("depth_img", depth_img);
    // cv::imshow("depth_align_to_color",depth_align_to_color);
    // cv::imshow("color_align_to_depth",color_align_to_depth);
    std::cout<<"distacne:"<<engineering.getDistance(frameset)<<std::endl;//获得深度距离信息
    cv::waitKey(1);
  }
}