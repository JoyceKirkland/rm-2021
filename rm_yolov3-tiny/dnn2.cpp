// #include <iostream>
// #include <fstream>
// #include <opencv2/opencv.hpp>
// #include <opencv2/dnn.hpp>
// #include <opencv2/dnn/shape_utils.hpp>
// #include <opencv2/imgproc.hpp>
// #include <opencv2/highgui.hpp>
// #include <librealsense2/rs.h>
// #include <librealsense2/rs.hpp>
// #include <librealsense2/h/rs_pipeline.h>
// #include <librealsense2/h/rs_option.h>
// #include <librealsense2/h/rs_frame.h>
// #include<librealsense2/rsutil.h>
// #include <librealsense2/hpp/rs_frame.hpp>
// using namespace std;
// using namespace cv;
// using namespace rs2;
#include "configure.h"
#include "serialport.cpp"
// Remove the bounding boxes with low confidence using non-maxima suppression
// void postprocess(cv::Mat& frame, std::vector<cv::Mat>& outs);
 
// Get the names of the output layers
// std::vector<cv::String> getOutputsNames(const cv::dnn::Net& net);
 
// Draw the predicted bounding box
// void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame);
 
// Initialize the parameters
float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 416;        // Width of network's input image
int inpHeight = 416;       // Height of network's input image
 
vector<std::string> classes;


// Get the names of the output layers
vector<cv::String> getOutputsNames(const cv::dnn::Net& net)
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

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame,int min_distance)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255));
    int center_x=left+(right-left)/2;
    int center_y=top+(bottom-top)/2;
    float max_xy_to_center=0;

    //Get the label for the class name and its confidence
    string label = cv::format("%.2f", conf);
    char box_x[20];
    char box_y[20];
    sprintf(box_x,"x=%d",center_x);
    sprintf(box_y,"y=%d",center_y);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }
    else
    {
        cout<<"classes is empty..."<<endl;
    }
 
    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255));
    putText(frame, box_x, Point(center_x, center_y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255));
    putText(frame, box_y, Point(center_x, center_y+20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255));
    line(frame,Point2f((float)center_x,(float)center_y),Point2f(frame.cols/2,frame.rows/2),Scalar(0,255,255),2);

    // cout<<"(x,y):"<<(int16_t)center_x<<","<<(int16_t)center_y<<endl;
    SerialPort::RMserialWrite((int16_t)center_x,(int16_t)center_y,min_distance);

}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(cv::Mat& frame, std::vector<cv::Mat>& outs,int min_distance)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    vector<float>xy_to_center;
    float min_xy_to_center=0;
    int index=0;

    float temp_xy_to_center=0;
 
    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
 
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
 
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
            }

        }
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        int center_x1=(2*boxes[idx].x+boxes[idx].width)/2;
        int center_y1=(2*boxes[idx].y+boxes[idx].height)/2;
        // line(frame,Point2f((float)center_x1,(float)center_y1),Point2f(frame.cols/2,frame.rows/2),Scalar(255,255,255),2);
        temp_xy_to_center=sqrt(pow(center_x1-frame.cols/2,2)+pow(center_y1-frame.rows/2,2));
        xy_to_center.push_back(temp_xy_to_center);
        cout<<"xy_to_center["<<i<<"]:"<<xy_to_center[i]<<endl;
        min_xy_to_center=xy_to_center[0];

    }
    line(frame,Point2f(frame.cols/2-20,frame.rows/2),Point2f(frame.cols/2+20,frame.rows/2),Scalar(255,255,255),2);
    line(frame,Point2f(frame.cols/2,frame.rows/2-20),Point2f(frame.cols/2,frame.rows/2+20),Scalar(255,255,255),2);

        for(size_t i=0;i<indices.size();i++)
        {
            // cout<<"xy_to_center["<<i<<"]:"<<xy_to_center[i]<<endl;
            if(xy_to_center[i]<min_xy_to_center)
            {
                min_xy_to_center=xy_to_center[i];
                index=i;
            }

        }
        cout<<"min_xy_to_center:"<<min_xy_to_center<<endl;
        cout<<"index:"<<index<<endl;

    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[index];
        cv::Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame,min_distance);
        // cout<<"idx["<<i<<"]:"<<idx<<endl;
    }

}
int find_capture()//choose the useful id of VideoCapture
{
    int cap_index=-1;
    int caps[6]={0,1,3,5,6,7};  
    // VideoCapture cap(NULL);
    Mat find_src;
    for(int i=0;i<6;i++)
    { 
        cap_index=caps[i];
        VideoCapture cap_f(cap_index);
        for(int i=0;i<10;i++)
        {
            cap_f>>find_src;
        }
        if(!find_src.empty())
        {
            cout<<"cap_index:"<<cap_index<<endl;
            // cap_index++;
            break;
        }
    }
    return cap_index;
}

int main(int argc, char** argv)
{
    // Load names of classes
    //________choose the id of VideoCapture_______//
    int cap_index=find_capture();
    
    SerialPort serialport;
	int change=1;
    string classesFile = "voc_classes.txt";
    ifstream classNamesFile(classesFile.c_str());
    if (classNamesFile.is_open())
    {
        string className = "";
        while (std::getline(classNamesFile, className))
            classes.push_back(className);
    }
    else{
        std::cout<<"can not open classNamesFile"<<std::endl;
    } 
    // Give the configuration and weight files for the model
    String modelConfiguration = "yolov3-tiny.cfg";
    String modelWeights = "yolov3-tiny_final.weights";
 
    // Load the network
    dnn::Net net = dnn::readNetFromDarknet(modelConfiguration, modelWeights);
    std::cout<<"Read Darknet..."<<std::endl;
    net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(dnn::DNN_TARGET_CPU);
 
    // Process frames.
    std::cout <<"Processing..."<<std::endl;

    VideoCapture cap(cap_index);
    Mat src;
	rs2::frame color_frame;
    rs2::frame depth_frame;
	colorizer c;   // 帮助着色深度图像
    pipeline pipe;         //创建数据管道
    pipeline_profile profile = pipe.start(); //start()函数返回数据管道的profile
    // float depth_scale = get_depth_scale(profile.get_device());

	namedWindow("1",WINDOW_NORMAL);
    setWindowProperty("1", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);    
    for(;;)
    {
        // get frame from the video
		frameset frameset = pipe.wait_for_frames();  //堵塞程序直到新的一帧捕获

        cap >> src;
		double t1 = (double)cv::getTickCount();
		rs2::video_frame video_src=frameset.get_color_frame();
		rs2::depth_frame depth = frameset.get_depth_frame();

        frameset.get_data();
		float width=depth.get_width();
		float height=depth.get_height();
        const int color_w=video_src.as<video_frame>().get_width();
        const int color_h=video_src.as<video_frame>().get_height();
        Mat color_image(Size(color_w,color_h),
                        CV_8UC3,(void*)video_src.get_data(),Mat::AUTO_STEP);
		cvtColor(color_image,color_image,COLOR_RGB2BGR);
        int min_distance=depth.get_distance(width/2,height/2)*100;//cm
        // cout<<"min_distance: "<<min_distance<<"cm"<<endl;
		 
        // Create a 4D blob from a frame.
        Mat blob;
        dnn::blobFromImage(color_image, blob, 1/255.0, Size(inpWidth, inpHeight), Scalar(0,0,0), true, false);
 
        //Sets the input to the network
        net.setInput(blob);
 
        // Runs the forward pass to get output of the output layers
        vector<Mat> outs;
        net.forward(outs, getOutputsNames(net));
 
        // Remove the bounding boxes with low confidence
        postprocess(color_image, outs,min_distance);
 
        // Put efficiency information. The function getPerfProfile returns the
        // overall time for inference(t) and the timings for each of the layers(in layersTimes)
        vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        string label = format("Inference time for a frame : %.2f ms", t);
        putText(color_image, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
 
        // Write the frame with the detection boxes
        Mat detectedFrame;
        color_image.convertTo(detectedFrame, CV_8U);
        //show detectedFrame
		t1 = ((double)getTickCount() - t1) / getTickFrequency();
		int fps = int(1.0 / t1);//转换为帧率
		cout << "FPS: " << fps<<endl;//输出帧率
        // imshow("detectedFrame",detectedFrame);
        int rm_recive[3];
        
        SerialPort::RMreceiveData(rm_recive);
		int key = waitKey(1);
        if(char(key) == 27)break;
        if(change==1)
        {
            imshow("1",detectedFrame);
        }
        else if(change==2)
        {
            imshow("1",src);
        }

        if(char(key) ==49)
        {
            change=1;

        }else if(char(key) ==50)
        {
            change=2;
        }

        /*——————按键切换屏幕，串口读取——————*/
        // if(rm_recive[1] ==1)
        // {
        //     change=1;

        // }else if(rm_recive[1] ==2)
        // {
        //     change=2;
        // }
		
    }
    cap.release();
    cout<<"Esc..."<<endl;
    return 0;
}