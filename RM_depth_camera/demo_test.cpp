/*
 * @Author: joyce
 * @Date: 2021-05-29 14:22:39
 * @LastEditTime: 2021-05-29 22:38:20
 * @LastEditors: Please set LastEditors
 * @Description:: 
 */

// #include "depth_camera.h"
#include "depth_camera.cpp"

int main()
{
  rs2::pipeline pipeline;
  RM_DepthCapture depth;
  rs2 ::pipeline_profile profile = pipeline.start(); //start()函数返回数据管道的profile

  for(;;)
  {
    rs2::frameset frameset=pipeline.wait_for_frames();

    // cv::Mat color_img=depth.getColorImg(pipeline,frameset);
    // cv::Mat depth_img=depth.getDepthImg(pipeline,frameset);
    cv::Mat depth_align_to_color=depth.getAlignedDepthFrame(pipeline,frameset);
    // cv::imshow("1",color_img);
    // cv::imshow("2",depth_img);
    cv::imshow("3",depth_align_to_color);
    // std::cout<<"distacne:"<<depth.getDistance(frameset)<<std::endl;
    cv::waitKey(1);
    // std::cout<<1<<std::endl;
  }
}