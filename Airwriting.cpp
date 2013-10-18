////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2013, Ronnie Wang（王程诚）
// Permission of use under MIT License
//
// Filename     ：Airwriting.cpp
// Project Code ：
// Abstract     ：This file contains the main loop of the program Airwriting. Major global variables are included, plus the state machine that drives Airwriting.
// Reference    ：
//
// Version      ：1.0
// Author       ：Ronnie Wang（王程诚）
// Accomplished date ： 2013.06
//
// Replaced version  :
// Original Author   :
// Accomplished date : 
//
////////////////////////////////////////////////////////////////////////////
#include "opencv2\core\core.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "Particle.h"
#include <iostream>
#include <vector>
#include <stdio.h>

//DEBUG_GLOBAL用于在全局层面测试每一帧的情况
//#define DEBUG_GLOBAL

using namespace cv;
using namespace std;

//hide the local functions in an anon namespace
namespace
{
	//全局变量

	//粒子滤波类，本程序中的粒子滤波使用100个粒子，且定义状态转移矩阵为
	//  1.4  -0.3  -0.1
	//    1     0     0
	//    0     1     0
	//但实际上你可以看到这是一个9×9的矩阵。这是为了和粒子维数相匹配
	Particle particles(100,(Mat_<double>(9,9)<<
		1.4,-0.3,-0.1,0,0,0,0,0,0,
		1,0,0,0,0,0,0,0,0,
		0,1,0,0,0,0,0,0,0,
		0,0,0,1.4,-0.3,-0.1,0,0,0,
		0,0,0,1,0,0,0,0,0,
		0,0,0,0,1,0,0,0,0,
		0,0,0,0,0,0,1,0,0,
		0,0,0,0,0,0,0,1,0,
		0,0,0,0,0,0,0,0,1));
	Mat curframe;//当前帧
	Mat bkgnd;//空白背景帧
	Mat preframe;//前一帧
	//bool trackObject = false;
	//bool airWriting = false;
	bool still=false;//判断当前帧是否静止
	char state='a';//空中手书状态。'a'是等待状态，'b'是跟踪状态，'c'是书写状态
	double timer;//给状态机的定时器
	vector<vector<Point> > lines;//书写轨迹
	//Point origin;
	Mat SE=getStructuringElement(MORPH_ELLIPSE,Size(5,5));//用于闭运算的结构

    void help(char** av)
    {
        cout << "\nThis program justs gets you started reading images from video\n"
        "Usage:\n./" << av[0] << " <video device number>\n" << "q,Q,esc -- quit\n"
        << endl;
    }

	///
	//所输入的均为RGB图像。
	//用论文中所述的方法计算帧差
	//参数列表
	//preframe  减帧
	//curframe  被减帧
	//返回值 Mat 黑白二值图像
	Mat framediff(const Mat& preframe, const Mat& curframe){
		vector<Mat> curchannel(3), prechannel(3), finger(3);//有个中间参数被命名为finger，但是这与手势没有任何关系
			split(curframe,curchannel);
			split(preframe,prechannel);
			for (int i=0;i!=3;i++){
				absdiff(curchannel[i],prechannel[i],finger[i]);
				threshold(finger[i],finger[i],30,255,THRESH_BINARY);
			}
			Mat finger_step1=finger[0]|finger[1]|finger[2];
			return finger_step1;
	}

	///
	//计时器开始计时
	void Setup(void){
		timer=(double)getTickCount();
	}

	///
	//计时器停止计时，并计算是否达到了指定的延迟时间
	//注意：论文中写的状态转换的延迟时间是两秒，但是实际应用中只要一秒就够了
	bool Timervalid(void){
		double t=(double)getTickCount();
		return ((t-timer)/getTickFrequency())>1;
	}

}

void imgshow(Mat& , const vector<vector<Point> >&, const string&);

int main(int ac, char** av)
{
    if (ac != 2)
    {
        help(av);
        return 1;
    }
    std::string arg = av[1];//程序启动输入参数，可以输入后缀为.avi的文件调试，也可输入0代表摄像头。可以在project中的properties中指定
    VideoCapture capture(arg); //try to open string, this will attempt to open it as a video file
    if (!capture.isOpened()){
		//if this fails, try to open as a video camera, through the use of an integer param
		capture.open(atoi(arg.c_str()));
		//capture.set(CV_CAP_PROP_FRAME_WIDTH,320);
		//capture.set(CV_CAP_PROP_FRAME_HEIGHT,240);
        
	}
    if (!capture.isOpened())
    {
        cerr << "Failed to open a video device or video file!\n" << endl;
        help(av);
        return 1;
    }

	string window_name = "Air Writing Debug Version 2";
    cout << "press space to save a picture. q or esc to quit" << endl;
    namedWindow(window_name, CV_WINDOW_AUTOSIZE);
	//调试用
	//namedWindow("debug window 1", CV_WINDOW_AUTOSIZE);
	//namedWindow("debug window 2", CV_WINDOW_AUTOSIZE);
	//namedWindow("debug window 3", CV_WINDOW_AUTOSIZE);

	//摄像头启动时，背景必须为空白，将此空白背景存储入bkgnd
	capture>>curframe;
	bkgnd=curframe.clone();
	preframe=bkgnd.clone();
	Mat img;
	vector<Point> pts;//one stroke

	//如果要记录手势，可以取消下面一行的注释
	//Dango是什么不要太在意
	//VideoWriter vm("Dango.avi",-1,15,cv::Size(640,480));

    while(1){
		capture >> curframe;
		if(curframe.empty())
			continue;
		curframe.copyTo(img);

		Mat vali=framediff(curframe,preframe);//前一帧减当前帧。
		Mat handimg=framediff(curframe,bkgnd);//背景帧减当前帧。显然变量名起得不好
		if (countNonZero(vali)<800 && countNonZero(handimg)>8000)//当场景停顿且有前景物体出现时
			still=true;
		else{
			Setup();//刷新计时器
			still=false;
		}

		switch(state){
		case 'a'://不跟踪
			if(still){
				state='b';
				morphologyEx(handimg,handimg,MORPH_CLOSE,SE);//闭运算
				particles.InitParticle(handimg);//既然已经获得了黑白手势图像，那就利用其中的轮廓信息初始化100个粒子
			}
			imgshow(img,lines,window_name);
			break;
		case 'b'://跟踪，不书写
			particles.PredictParticle();//粒子预测过程
			if(!particles.MeasureParticle(handimg)){//如果跟踪失败
				state='a';
				imgshow(img,lines,window_name);
				break;
			}
			if(still && Timervalid()){//如果手势定格超过一秒则开始书写
				state='c';
				Setup();
			}
			circle(img,particles.MeasuredFingertip(),8,Scalar(0,0,255),2);
			imgshow(img,lines,window_name);
			break;
		case 'c'://跟踪，书写
			particles.PredictParticle();//粒子预测过程
			if(!particles.MeasureParticle(handimg)){//如果跟踪失败
				state='a';
				lines.push_back(pts);
				pts.clear();
				imgshow(img,lines,window_name);
				break;
			}
			if(still && Timervalid()){//停笔
				state='b';
				Setup();
				lines.push_back(pts);
				pts.clear();
				imgshow(img,lines,window_name);
				break;
			}
			Point temp=particles.MeasuredFingertip();//绘出本帧所预测的点
			pts.push_back(temp);
			circle(img,temp,8,Scalar(0,255,0),2);
			if(pts.size()>=2){
				for(int i=0;i!=pts.size()-1;i++){
					line(img,pts[i],pts[i+1],Scalar(255,0,0),2);
				}
			}
			imgshow(img,lines,window_name);
			break;
		}

		preframe=curframe.clone();
		//如果记录视频，请取消下面这行注释
		//vm<<img;

#ifdef DEBUG_GLOBAL
		char key = (char) waitKey(0); 
#else
		char key= (char) waitKey(30);
#endif
		switch (key)
		{
			case 'c'://按c清除全部轨迹
				lines.clear();
				break;
			case 'q':
			case 'Q':
			case 27: //escape key
				return 0;
				//如果记录视频，请取消下面这行注释
				//vm.release();
			default:
				break;
		}
	}

    return 0;
}

///
//在给定窗口中展示轨迹
//参数列表
//img    当前帧
//lines  历史轨迹
//window 窗口名
void imgshow(Mat& img, const vector<vector<Point> >& lines, const string& window){
	if(!lines.empty()){
		for (int i=0;i!=lines.size();i++){
			for (int j=0;j!=lines[i].size()-1;j++){
				line(img,lines[i][j],lines[i][j+1],Scalar(255,0,0),2);
			}
		}
	}
	imshow(window, img);
}