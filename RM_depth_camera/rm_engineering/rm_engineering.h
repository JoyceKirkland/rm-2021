/*
 * @Author: joyce
 * @Date: 2021-05-30 11:38:19
 * @LastEditTime: 2021-05-31 11:47:58
 * @LastEditors: Please set LastEditors
 * @Description:: 
 */

#include "depth_camera/depth_camera.cpp"
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>

class RM_Engineering:public RM_DepthCapture
{
public:
  std::vector<cv::Point>center_x_y_;//中心点点集
  float confThreshold_ = 0.5; // 置信阈值
  float nmsThreshold_ = 0.4;  // 非最大抑制阈值
  int inpWidth_ = 416;        // 网络输入图像宽度
  int inpHeight_ = 416;       // 网络输入图像高度
  std::vector<cv::Mat> outs_; // 获取输出层的输出
  std::vector<std::string> classes_;       // 类名标签合集
  std::string modelConfiguration_;
  std::string modelWeights_;
  cv::dnn::Net net_;

public:
  RM_Engineering();
  ~RM_Engineering();
  std::vector<cv::Mat> getOutputsLayer(cv::dnn::Net &_net,cv::Mat &_frame);
  std::vector<cv::String> getOutputsNames(const cv::dnn::Net &_net);
  std::vector<cv::Point> postprocess(cv::Mat &_frame, std::vector<cv::Mat> &_outs, int _min_distance,std::vector<std::string> _classes);
  void drawPred(int _classId, float _conf, int _left, int _top, int _right, int _bottom, cv::Mat &_frame, int _min_distance,  std::vector<std::string> _classes);
  cv::dnn::Net setNet(std::string _modelConfiguration,std::string _modelWeights);
  std::vector<std::string> setClassNames(std::string _classesFile);
  // float getconfThreshold();

};
//置信阈值,非最大抑制阈值,网络输入图像宽度,网络输入图像高度的方法.

