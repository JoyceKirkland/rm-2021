/*
 * @Author: joyce
 * @Date: 2021-05-30 11:38:19
 * @LastEditTime: 2021-05-30 21:47:16
 * @LastEditors: Please set LastEditors
 * @Description:: 
 */

#include "depth_camera/depth_camera.cpp"
#include <opencv2/dnn.hpp>

class RM_Engineering:public RM_DepthCapture
{
public:
  // int center_x_;
  // int center_y_;
  cv::Point center_x_y;
  float confThreshold_ = 0.5; // Confidence threshold
  float nmsThreshold_ = 0.4;  // Non-maximum suppression threshold
  int inpWidth_ = 416;        // Width of network's input image
  int inpHeight_ = 416;       // Height of network's input image


//   float distance;
public:
  RM_Engineering();
  ~RM_Engineering();
  std::vector<cv::Mat> getOutputsLayer(cv::dnn::Net &net);
  std::vector<cv::String> getOutputsNames(const cv::dnn::Net &net,cv::Mat &frame);
  std::vector<cv::Point> postprocess(cv::Mat &_frame, std::vector<cv::Mat> &_outs, int _min_distance);
  void drawPred(int _classId, float _conf, int _left, int _top, int _right, int _bottom, cv::Mat &_frame, int _min_distance);

};


