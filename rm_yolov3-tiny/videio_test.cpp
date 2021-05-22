/*
 * @Author: joyce
 * @Date: 2021-05-18 22:05:46
 * @LastEditTime: 2021-05-22 12:04:45
 * @LastEditors: Please set LastEditors
 * @Description:: 
 */
#include "configure.h"

int main()
{
   int cap_index=-1;
    // VideoCapture cap_0(0);
    // VideoCapture cap_1(1);
    // VideoCapture cap_2(2);
    // VideoCapture cap_3(3);
    // VideoCapture cap_4(4);
    // VideoCapture cap_5(5);
    // VideoCapture cap_6(6);
    // VideoCapture cap_7(7);
    // VideoCapture cap_8(8);
    

    
    Mat find_src;
    for(;cap_index<7;cap_index++)
    // do
    {
        VideoCapture cap(cap_index);
        for(int i=0;i<3;i++)
        {
            cap>>find_src;
            if(find_src.empty())
            {
                cap_index++;
                // cout<<"now_cap("<<cap_index<<")"<<endl;
                // break;
            }

        }
        
    }
    //while (cap_index<7);
    // cout<<"now_cap("<<cap_index<<")"<<endl; 
    VideoCapture cap(cap_index);
    Mat frame;
    
    for(;;)
    {
        // cap_index++;
        cap>>frame;

        // if(frame.empty())
        {
            // continue;            
            // cout<<"video error"<<endl;
            // break;
        }
        imshow("frame",frame);
        // waitKey(1);
        int key = waitKey(1);
        if(char(key) == 27)break;
    }
}