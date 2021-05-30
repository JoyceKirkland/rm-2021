/*
 * @Author: joyce
 * @Date: 2021-05-30 11:39:43
 * @LastEditTime: 2021-05-30 11:39:43
 * @LastEditors: Please set LastEditors
 * @Description:: 
 */

#include "rm_engineering.h"
// #include "depth_camera/depth_camera.h"

RM_Engineering::RM_Engineering(){}

RM_Engineering::~RM_Engineering(){}

//获取输出层的名称
vector<cv::String> RM_Engineering::getOutputsNames(const cv::dnn::Net &_net)
{
  static std::vector<cv::String> names;
  if (names.empty())
  {
      //Get the indices of the output layers, i.e. the layers with unconnected outputs
      std::vector<int> outLayers = net.getUnconnectedOutLayers();

      //get the names of all the layers in the network
      std::vector<cv::String> layersNames = net.getLayerNames();

      // Get the names of the output layers in names
      names.resize(outLayers.size());
      for (size_t i = 0; i < outLayers.size(); ++i)
          names[i] = layersNames[outLayers[i] - 1];
  }
  return names;
}


std::vector<cv::Mat> RM_Engineering::getOutputsLayer(cv::dnn::Net &_net,cv::Mat &_frame)
{
  cv::Mat blob;
  dnn::blobFromImage(_frame, blob, 1 / 255.0, cv::Size(inpWidth_, inpHeight_), cv::Scalar(0, 0, 0), true, false);
  //设置网络输入
  _net.setInput(blob);
  //获取输出层的输出
  std::vector<cv::Mat> outs;
  _net.forward(outs, getOutputsNames(_net));
  return outs;
}

void RM_Engineering::drawPred(int _classId, float _conf, int _left, int _top, int _right, int _bottom, cv::Mat &_frame, int _min_distance)
{
  //画矩形框
  rectangle(_frame, cv::Point(_left, _top), cv::Point(_right, _bottom), cv::Scalar(0, 0, 255), 2);
  int center_x = _left + (_right - _left) / 2;
  int center_y = _top + (_bottom - top) / 2;
  float max_xy_to_center = 0;

  //获取类标签名称和置信度
  string label = cv::format("%.2f", _conf);
  char box_x[20];
  char box_y[20];
  sprintf(box_x, "x=%d", center_x);
  sprintf(box_y, "y=%d", center_y);
  if (!classes.empty())
  {
      CV_Assert(classId < (int)classes.size());
      label = classes[classId] + ":" + label;
  }
  else
  {
      cout << "classes is empty..." << endl;
  }

  int baseLine;
  Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
  top = max(_top, labelSize.height);
  putText(_frame, label, Point(_left, _top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));
  putText(_frame, box_x, Point(center_x, center_y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));
  putText(_frame, box_y, Point(center_x, center_y + 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));
  line(_frame, Point2f((float)center_x, (float)center_y), Point2f(frame.cols / 2, frame.rows / 2), Scalar(0, 255, 255), 2);


}

std::vector<cv::Point> RM_Engineering::postprocess(cv::Mat &_frame, std::vector<cv::Mat> &_outs, int _min_distance)
{
  std::vector<int> classIds;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  std::vector<float> xy_to_center;
  std::vector<cv::Point>center_x_y;
  float min_xy_to_center = 0;
  int index = 0;

  float temp_xy_to_center = 0;

  for (size_t i = 0; i < _outs.size(); ++i)
  {
    float *data = (float *)_outs[i].data;
    for (int j = 0; j < _outs[i].rows; ++j, data += _outs[i].cols)
    {
      Mat scores = _outs[i].row(j).colRange(5, _outs[i].cols);
      Point classIdPoint;
      double confidence;
      // Get the value and location of the maximum score
      minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
      if (confidence > confThreshold)
      {
          int centerX = (int)(data[0] * _frame.cols);
          int centerY = (int)(data[1] * _frame.rows);
          int width = (int)(data[2] * _frame.cols);
          int height = (int)(data[3] * _frame.rows);
          int left = centerX - width / 2;
          int top = centerY - height / 2;
          classIds.push_back(classIdPoint.x);
          confidences.push_back((float)confidence);
          boxes.push_back(cv::Rect(left, top, width, height));
      }
    }
    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        int center_x1 = (2 * boxes[idx].x + boxes[idx].width) / 2;
        int center_y1 = (2 * boxes[idx].y + boxes[idx].height) / 2;
        // line(frame,Point2f((float)center_x1,(float)center_y1),Point2f(frame.cols/2,frame.rows/2),Scalar(255,255,255),2);
        temp_xy_to_center = sqrt(pow(center_x1 - _frame.cols / 2, 2) + pow(center_y1 - _frame.rows / 2, 2));
        xy_to_center.push_back(temp_xy_to_center);
        center_x_y.push_back(cv::Point(center_x1,center_y1));
        // cout<<"xy_to_center["<<i<<"]:"<<xy_to_center[i]<<endl;
        min_xy_to_center = xy_to_center[0];
    }

    for (size_t i = 0; i < indices.size(); i++)
    {
        // cout<<"xy_to_center["<<i<<"]:"<<xy_to_center[i]<<endl;
        if (xy_to_center[i] < min_xy_to_center)
        {
            min_xy_to_center = xy_to_center[i];
            index = i;
        }
    }
    // cout<<"min_xy_to_center:"<<min_xy_to_center<<endl;

    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[index];
        cv::Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                  box.x + box.width, box.y + box.height, _frame, _min_distance);
        // cout<<"idx["<<i<<"]:"<<idx<<endl;
    }
  }
  return center_x_y;
}