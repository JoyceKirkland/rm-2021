/*
 * @Author: joyce
 * @Date: 2021-05-27 21:45:36
 * @LastEditTime: 2021-05-29 22:39:19
 * @LastEditors: Please set LastEditors
 * @Description:: 
 */


#include "depth_camera.h"

RM_DepthCapture::RM_DepthCapture(){}

RM_DepthCapture::~RM_DepthCapture(){}

cv::Mat RM_DepthCapture::getColorImg(rs2::pipeline _pipeline,rs2::frameset _frameset)//获得彩色图
{ 
  rs2::video_frame video_src = _frameset.get_color_frame();
  const int color_weight = video_src.as<rs2::video_frame>().get_width();
  const int color_height = video_src.as<rs2::video_frame>().get_height();
  color_img_=cv::Mat(cv::Size(color_weight, color_height),
                      CV_8UC3, (void *)video_src.get_data(), cv::Mat::AUTO_STEP);
  cv::cvtColor(color_img_, color_img_,cv::COLOR_BGR2RGB);
  return color_img_;
}

cv::Mat RM_DepthCapture::getDepthImg(rs2::pipeline _pipeline,rs2::frameset _frameset)//获得彩色深度图
{
  rs2::depth_frame depth_frame_temp = _frameset.get_depth_frame();
  rs2::frame depth_frame_colormap = depth_frame_temp.apply_filter(color_map_);
  const int dep_w=depth_frame_colormap.as<rs2::video_frame>().get_width();
  const int dep_h=depth_frame_colormap.as<rs2::video_frame>().get_height();

  depth_img_=cv::Mat(cv::Size(dep_w,dep_h),
                        CV_8UC3,(void*)depth_frame_colormap.get_data(),cv::Mat::AUTO_STEP);
  return depth_img_;
}

float RM_DepthCapture::getDistance(rs2::frameset _frameset)
{
  rs2::depth_frame depth_frame= _frameset.get_depth_frame();
  float width = depth_frame.get_width();
  float height = depth_frame.get_height();
  distance_ = depth_frame.get_distance(width / 2, height / 2) * 100; //cm
  return distance_;
}

cv::Mat RM_DepthCapture::getAlignedDepthFrame(rs2::pipeline _pipeline,rs2::frameset _frameset)//深度图对齐到彩色图
{
  config_.enable_stream(RS2_STREAM_COLOR);
  config_.enable_stream(RS2_STREAM_DEPTH);
  rs2::align align_to_color(RS2_STREAM_COLOR);

  _frameset=align_to_color.process(_frameset);
  auto depth = _frameset.get_depth_frame();
  auto colorized_depth = color_map_.colorize(depth); 
  aligned_depth_frame_=cv::Mat(cv::Size(colorized_depth.as<rs2::video_frame>().get_width(),
			colorized_depth.as<rs2::video_frame>().get_height()), CV_8UC3, (void*)colorized_depth.get_data(), cv::Mat::AUTO_STEP);
  
  return aligned_depth_frame_;
}


