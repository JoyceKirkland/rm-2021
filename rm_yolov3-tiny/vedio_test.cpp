/*
 * @Author: joyce
 * @Date: 2021-05-26 21:37:28
 * @LastEditTime: 2021-05-27 16:56:59
 * @LastEditors: Please set LastEditors
 * @Description:: 
 */
#include "configure.h"

int main()
{
    int cap_index=-1;
    // int caps[6]={0,1,3,5,6,7};  
    // VideoCapture cap(NULL);
    vector<int>caps_index;
    Mat find_src;
    for(int i=0;i<8;i++)
    { 
        cap_index=i;
        VideoCapture cap_f(cap_index);
        for(int j=0;j<10;j++)
        {
            cap_f>>find_src;
           
        }
        if(!find_src.empty())
        {
            caps_index.push_back(i);
            cout<<"cap_index:"<<cap_index<<endl;
                // cap_index++;
                // break;
        }
    }
    // cout<<"cap_index:"<<cap_index<<endl;
    // for(int i=0;i<caps.size();i++)
    {
        // VideoCapture cap_1 (caps[i]);

    }
    VideoCapture cap(caps_index[0]);
    VideoCapture cap_1();

    Mat src;
    for(;;)
    {
        cap>>src;
        imshow("1",src);
        int key=waitKey(1);
        if((char)key==27)break;
    }
}