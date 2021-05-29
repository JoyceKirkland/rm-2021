/*
 * @Author: joyce
 * @Date: 2021-05-29 14:22:39
 * @LastEditTime: 2021-05-29 22:04:06
 * @LastEditors: Please set LastEditors
 * @Description:: 
 */

#include "depth_camera.h"
#include <opencv2/opencv.hpp>
// #include "depth_camera.cpp"
using namespace cv;
using namespace rs2;


int main()
{
  rs2::colorizer color_map;
	// rs2::context ctx;
	rs2::config cfg;
	cfg.enable_stream(RS2_STREAM_COLOR);
	cfg.enable_stream(RS2_STREAM_DEPTH);
	rs2::align align_to_color(RS2_STREAM_COLOR);
	rs2::pipeline pipe;
  
	pipe.start(cfg);

	// namedWindow("color_img", WINDOW_AUTOSIZE);

	for(;;) 
  {
		frameset frameset = pipe.wait_for_frames();
    // func();
		//  获取RGB图
		//  对齐图像
		frameset = align_to_color.process(frameset);
		// frame color_frames = frames.get_color_frame();

    auto depth = frameset.get_depth_frame();
    auto colorized_depth = color_map.colorize(depth);
		//  帧转化为Mat 尺寸为RGB帧的尺寸
		Mat algin_img = Mat(Size(colorized_depth.as<video_frame>().get_width(),
			colorized_depth.as<video_frame>().get_height()), CV_8UC3, (void*)colorized_depth.get_data(), Mat::AUTO_STEP);
		imshow("algin_img", algin_img);
    cv::waitKey(1);
		// depth_frame depth_frames = aligned_set.get_depth_frame();
		// 从着色的深度数据中创建OpenCV大小（w，h）的OpenCV矩阵
		// frame depth_frames_show = depth_frames.apply_filter(color_map);
		// Mat depth_img = Mat(Size(depth_frames_show.as<video_frame>().get_width(),
			// depth_frames_show.as<video_frame>().get_height()), CV_8UC3, (void*)depth_frames_show.get_data(), Mat::AUTO_STEP);
		// imshow("depth_img", depth_img);
  }
  // RM_DepthCapture depth;
  // depth.get_Profile();
  // for(;;)
  // {
  //   depth.get_Frameset();
  //   cv::Mat color_img=depth.getColorImg();
  //   cv::imshow("1",color_img);
  //   cv::waitKey(1);
  // }
}
/*
/*
 * @Author: joyce
 * @Date: 2021-05-27 21:45:36
 * @LastEditTime: 2021-05-29 12:15:55
 * @LastEditors: Please set LastEditors
 * @Description:: 
 


#include "depth_camera.h"

RM_DepthCapture::RM_DepthCapture()
{
  // rs2 ::colorizer color_map_;                         //帮助着色深度图像
  // rs2 ::pipeline pipeline_;                           //创建数据管道
  // rs2 ::pipeline_profile profile_ = pipeline_.start(); //start()函数返回数据管道的profile
  // rs2 ::frameset frameset_=pipeline_.wait_for_frames();
  // rs2 ::config config_;

}
rs2 ::pipeline RM_DepthCapture::get_Pipeline()//创建数据管道
{
  rs2 ::pipeline pipeline;
  return pipeline;                           
}

rs2 ::frameset RM_DepthCapture::get_Frameset()
{
  rs2 ::frameset frameset=get_Pipeline().wait_for_frames();
  return frameset;
}

rs2 ::pipeline_profile RM_DepthCapture::get_Profile()
{
  rs2 ::pipeline_profile profile = get_Pipeline().start(); //start()函数返回数据管道的profile
  return profile;
}

RM_DepthCapture::~RM_DepthCapture(){}

cv::Mat RM_DepthCapture::getColorImg()//获得彩色图
{ 
  // rs2::frameset frameset=pipeline.wait_for_frames();
  rs2::video_frame video_src = get_Frameset().get_color_frame();
  const int color_weight = video_src.as<rs2::video_frame>().get_width();
  const int color_height = video_src.as<rs2::video_frame>().get_height();
  color_img_=cv::Mat(cv::Size(color_weight, color_height),
                      CV_8UC3, (void *)video_src.get_data(), cv::Mat::AUTO_STEP);
  return color_img_;
}
/*
// cv::Mat RM_DepthCapture::getDepthImg()//获得彩色深度图
// {
//   rs2::frame depth_frame_temp = get_Frameset().get_depth_frame();
//   rs2::frame depth_frame_colormap = frameset_.get_depth_frame().apply_filter(color_map_);
//   const int dep_w=depth_frame_temp.as<rs2::video_frame>().get_width();
//   const int dep_h=depth_frame_temp.as<rs2::video_frame>().get_height();

//   depth_img_=cv::Mat(cv::Size(dep_w,dep_w),
//                         CV_8UC3,(void*)depth_frame_colormap.get_data(),cv::Mat::AUTO_STEP);
//   return depth_img_;
// }

// cv::Mat RM_DepthCapture::getAlignedDepthFrame()//深度图对齐到彩色图
// {
//   config_.enable_stream(RS2_STREAM_COLOR);
//   config_.enable_stream(RS2_STREAM_DEPTH);
//   rs2::align align_to(RS2_STREAM_COLOR);
//   pipeline_.start(config_);

//   rs2::frameset align_set=align_to.process(frameset_);
//   rs2::frame video_src=align_set.get_color_frame();
//   rs2::depth_frame depth_frames=align_set.get_depth_frame();

//   color_img_=cv::Mat(cv::Size(video_src.as<rs2::video_frame>().get_width(),
// 			video_src.as<rs2::video_frame>().get_height()), CV_8UC3, (void*)video_src.get_data(), cv::Mat::AUTO_STEP);
//   rs2::frame depth_frames_show=depth_frames.apply_filter(color_map_);
  
//   aligned_depth_frame_=cv::Mat(cv::Size(depth_frames_show.as<rs2::video_frame>().get_width(),
// 			depth_frames_show.as<rs2::video_frame>().get_height()), CV_8UC3, (void*)depth_frames_show.get_data(), cv::Mat::AUTO_STEP);
//   return aligned_depth_frame_;
// }

______________________________


#include <iostream>
#include <librealsense2/rs.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class RM_DepthCapture
{
private:
  cv ::Mat color_img_;                                //彩色图
  cv ::Mat depth_img_;                                //彩色深度图
  cv ::Mat aligned_depth_frame_;                      //深度图对齐到彩色图
  // rs2 ::colorizer color_map_;                         //帮助着色深度图像
  // rs2 ::pipeline pipeline_;                           //创建数据管道
  // rs2 ::pipeline_profile profile_ = pipeline_.start(); //start()函数返回数据管道的profile
  // rs2 ::frameset frameset_=pipeline_.wait_for_frames();
  // rs2 ::config config_;

public:
  RM_DepthCapture();   //构造函数
  ~RM_DepthCapture();  //析构函数
  float getDistance(); //获得相机中心点到最近物体的距离
  rs2 ::frameset get_Frameset();
  rs2 ::pipeline get_Pipeline();
  rs2 ::pipeline_profile get_Profile();
  cv ::Mat getColorImg();          //获得彩色图
  // cv ::Mat getDepthImg();          //获得彩色深度图
  // cv ::Mat getAlignedDepthFrame(); //获得对齐后的彩色图
};

*/