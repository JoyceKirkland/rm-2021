'''
Author: joyce
Date: 2021-04-24 10:46:24
LastEditTime: 2021-04-25 13:23:57
LastEditors: Please set LastEditors
Description:: 
'''

import serial  # 导入模块
import threading
import time
import logging
import sys
import struct
# import crc8

CRC8Tab =[0, 94, 188, 226, 97, 63, 221, 131, 194, 156, 126, 32, 163, 253, 31, 65,
        157, 195, 33, 127, 252, 162, 64, 30, 95, 1, 227, 189, 62, 96, 130, 220,
        35, 125, 159, 193, 66, 28, 254, 160, 225, 191, 93, 3, 128, 222, 60, 98,
        190, 224, 2, 92, 223, 129, 99, 61, 124, 34, 192, 158, 29, 67, 161, 255,
        70, 24, 250, 164, 39, 121, 155, 197, 132, 218, 56, 102, 229, 187, 89, 7,
        219, 133, 103, 57, 186, 228, 6, 88, 25, 71, 165, 251, 120, 38, 196, 154,
        101, 59, 217, 135, 4, 90, 184, 230, 167, 249, 27, 69, 198, 152, 122, 36,
        248, 166, 68, 26, 153, 199, 37, 123, 58, 100, 134, 216, 91, 5, 231, 185,
        140, 210, 48, 110, 237, 179, 81, 15, 78, 16, 242, 172, 47, 113, 147, 205,
        17, 79, 173, 243, 112, 46, 204, 146, 211, 141, 111, 49, 178, 236, 14, 80,
        175, 241, 19, 77, 206, 144, 114, 44, 109, 51, 209, 143, 12, 82, 176, 238,
        50, 108, 142, 208, 83, 13, 239, 177, 240, 174, 76, 18, 145, 207, 45, 115,
        202, 148, 118, 40, 171, 245, 23, 73, 8, 86, 180, 234, 105, 55, 213, 139,
        87, 9, 235, 181, 54, 104, 138, 212, 149, 203, 41, 119, 244, 170, 72, 22,
        233, 183, 85, 11, 136, 214, 52, 106, 43, 117, 151, 201, 74, 20, 246, 168,
        116, 42, 200, 150, 21, 75, 169, 247, 182, 232, 10, 84, 215, 137, 107, 53]

g_CRC_buf=[0,0,0,0,0,0,0]
g_write_buf=[0,0,0,0,0,0,0,0,0]


#CRC校验
def crc_sum(buf, len):
    check = 0
    # (ctypes.c_uint16)len=len

    i = 0
    while(i<len):
        # crc += data[i]
        check=CRC8Tab[buf[i]^(buf[i])]
        i += 1
    return (check) & 0x00FF

def getDataForCRC(rect_x,rect_y,distacne_center):
    global g_CRC_buf
    g_CRC_buf[0]=0x53
    g_CRC_buf[1]=(rect_x>>8)&0xff
    g_CRC_buf[2]=(rect_x)&0xff
    g_CRC_buf[3]=(rect_y>>8)&0xff
    g_CRC_buf[4]=(rect_y)&0xff
    g_CRC_buf[5]=(distacne_center>>8) & 0xff
    g_CRC_buf[6]=(distacne_center) & 0xff

#发送内容
def getDataForSend(rect_x,rect_y,distacne_center,CRC):
    global g_write_buf
    g_write_buf[0]=0x53
    g_write_buf[1]=(rect_x>>8) & 0xff
    g_write_buf[2]=(rect_x)& 0xff
    g_write_buf[3]=(rect_y>>8) & 0xff
    g_write_buf[4]=(rect_y)& 0xff
    g_write_buf[5]=(distacne_center>>8) & 0xff
    g_write_buf[6]=(distacne_center) & 0xf
    g_write_buf[7]=CRC & 0xff
    g_write_buf[8]=0x45

# 打开串口
def DOpenPort():
    try:
        # 打开串口，并得到串口对象
        ser = serial.Serial('dev/ttyUSB0', 115200, timeout=0)
        # 判断是否打开成功
        if(False == ser.is_open):
           ser = -1
    except Exception as e:
        print("---异常---：", e)

    return ser

# 关闭串口
def DColsePort(ser):
    uart.fdstate = -1
    ser.close()
                               
# 写数据
def DWritePort(ser, rect_x,rect_y,distacne_center):
    getDataForCRC(rect_x,rect_y,distacne_center)
    CRC=crc_sum(g_CRC_buf,len(g_CRC_buf))
    getDataForSend(rect_x,rect_y,distacne_center,CRC)
    result = ser.write(g_write_buf)  # 写数据
    return result



def DReadPort(ser):
    # 循环接收数据，此为死循环，可用线程实现
    readstr = ""
    while(ser!=-1):
        if ser.in_waiting:
            readstr = ser.read(ser.in_waiting)
        else:
            print('none')
        # if readbuf[0] == 0x55 and readbuf[1] == 0xaa:
        #     readstr = readbuf
        # else:
        #     readstr = readstr + readbuf
            return readstr



def TestStop(ser):
    DColsePort(uart.fd)  # 关闭串口

