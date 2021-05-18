/*
 * @Author: joyce
 * @Date: 2021-05-18 22:05:46
 * @LastEditTime: 2021-05-18 22:09:25
 * @LastEditors: Please set LastEditors
 * @Description:: 
 */
#include "configure.h"

int main()
{
    VideoCapture cap(6);
    Mat frame;
    for(;;)
    {
        cap>>frame;
        imshow("frame",frame);
        waitKey(1);
        int key = waitKey(1);
        if(char(key) == 27)break;
    }
}