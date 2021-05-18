/*
 * @Author: joyce
 * @Date: 2021-01-21 15:17:19
 * @LastEditTime: 2021-05-16 15:12:15
 * @LastEditors: Please set LastEditors
 * @Description:: 
 */
 
#include"rm-eng_version2/configure.h"
#include "realsense.cpp"
// #include "serialport.h"
//笔记本更改
//主机更改test
int main() try
{
    mineral min;  
    SerialPort serialport;
    //double t=(double)cv::getTickCount();
    min.get_frame();
    //t=((double)cv::getTickCount()-t)/cv::getTickFrequency();
    //int fps=double(1.0/t);
    //cout<<"fps:"<<fps<<endl;
    //min.test();
    //cout<<"????"<<endl;
    return 0;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception & e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}