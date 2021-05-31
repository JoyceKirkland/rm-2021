/*
 * @Author: joyce
 * @Date: 2021-05-30 15:18:31
 * @LastEditTime: 2021-05-31 21:03:51
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
  std::string classesFile = "/home/joyce/github/rm_eng/rm/rm_yolov3-tiny/voc_classes.txt";
  std::string modelConfiguration = "/home/joyce/github/rm_eng/rm/rm_yolov3-tiny/yolov3-tiny.cfg";
  // String modelWeights = "yolov3-tiny_final.weights";
  std::string modelWeights = "/home/joyce/github/rm_eng/rm/rm_yolov3-tiny/yolov3-tiny_final_1178rgb.weights";
  std::vector<std::string> classes;
  
  for (;;)
  {
    rs2::frameset frameset = pipeline.wait_for_frames(); //堵塞程序直到新的一帧捕获

    float distance=engineering.getDistance(frameset);
    cv::Mat color_img=engineering.getColorImage(pipeline,frameset);
    classes=engineering.setClassNames(classesFile);
    cv::dnn::Net net=engineering.setNet(modelConfiguration,modelWeights);
    engineering.getOutputsNames(net);
    std::vector<cv::Mat> outs;
    outs=engineering.getOutputsLayer(net,color_img);
    engineering.postprocess(color_img,outs,distance,classes);
    // engineering.
    cv::imshow("color_img", color_img);
    // cv::imshow("depth_img", depth_img);
    // cv::imshow("depth_align_to_color",depth_align_to_color);
    // cv::imshow("color_align_to_depth",color_align_to_depth);
    std::cout<<"distacne:"<<engineering.getDistance(frameset)<<std::endl;//获得深度距离信息
    cv::waitKey(1);
  }
}