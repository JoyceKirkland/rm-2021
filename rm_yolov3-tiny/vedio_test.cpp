#include "configure.h"

int main()
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
    // cout<<"cap_index:"<<cap_index<<endl;
    VideoCapture cap(cap_index);
    Mat src;
    for(;;)
    {
        cap>>src;
        imshow("1",src);
        int key=waitKey(1);
        if((char)key==27)break;
    }
}