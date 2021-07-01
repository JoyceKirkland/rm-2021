/*
 * @Author: joyce
 * @Date: 2021-05-30 15:18:31
 * @LastEditTime: 2021-07-01 11:53:07
 * @LastEditors: Please set LastEditors
 * @Description:: 
 */

#include "rm_engineering/rm_engineering.cpp"
// #include <opencv2/dnn.hpp>
int main()
{
  rs2::pipeline pipeline; //创建数据管道
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

    float distance = engineering.getDistance(frameset);                      //获得深度距离信息
    cv::Mat color_img = engineering.getColorImage(pipeline, frameset);       //获取彩色图
    classes = engineering.setClassNames(classesFile);                        //加载类名文件
    cv::dnn::Net net = engineering.setNet(modelConfiguration, modelWeights); //加载网络
    engineering.getOutputsNames(net);                                        //获取输出层名称
    std::vector<cv::Mat> outs;
    // outs=engineering.getOutputsLayer(net,color_img);//获取输出层的输出
    outs = engineering.getOutputsLayer(net, color_img, 416, 416);

    // engineering.postprocess(color_img,outs,distance,classes);//矩形框筛选
    engineering.postprocess(color_img, outs, distance, classes, 0.5, 0.4);

    // engineering.
    cv::imshow("color_img", color_img); //显示
    // cv::imshow("depth_img", depth_img);
    // cv::imshow("depth_align_to_color",depth_align_to_color);
    // cv::imshow("color_align_to_depth",color_align_to_depth);
    std::cout << "distacne:" << engineering.getDistance(frameset) << std::endl; //显示深度距离信息
    cv::waitKey(1);
  }
}