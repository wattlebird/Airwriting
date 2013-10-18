////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2013, Ronnie Wang（王程诚）
// Permission of use under MIT License
//
// Filename     ：Fintipos.cpp
// Project Code ：
// Abstract     ：The implementation of the function to determine fingertip position
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
#include "Fintipos.h"
#include "opencv2\imgproc\imgproc.hpp"
#include <algorithm>
#include <iostream>
#include "opencv2\highgui\highgui.hpp"
#include <stdexcept>
#include <fstream>

//如果需要查看轮廓上哪几个点被选为描述轮廓的模板点，请取消注释下一行
//#define DEBUG_FIN
//如果需要输出轮廓信息，请取消注释下一行
//#define CSV_OUTPUT

//std::find_if的谓词
bool pangle(const double& angle){
	return (angle>-0.2);
}

//std::find_if的谓词
bool nangle(const double& angle){
	return !pangle(angle);
}

//std::max_element的谓词
bool fine_angle_set_compare(const std::vector<double>& a, const std::vector<double>& b){
	return a.size()<b.size();
}

//函数名：FingertipPos
//隶属类：无
//功能：对当前输入的手势轮廓判定其指尖位置，所输入的手势轮廓只能包含一个指尖
//参数列表：
//handcontour   手势轮廓。显然可见，由一系列点组成。
//返回参数： int 指示手指尖在handcontour中的index
//异常类：std::invalid_argument：当找不到手指特征时抛出
int FingertipPos(const std::vector<cv::Point>& handcontour){
	int len=handcontour.size();
	//std::vector<int> hull_index;
	std::vector<double> angle;
	//cv::convexHull(handcontour,hull_index);

#ifdef CSV_OUTPUT
	std::ofstream outf("out.csv");
	outf<<"index,angle,x,y"<<std::endl;
#endif

	for (int i=0;i!=len;i++){//对轮廓中的每一个点
		if(handcontour[i].y<450){//其y轴坐标在450以内的
			cv::Point2d pt1=handcontour[(i+15)%len]-handcontour[i];
			cv::Point2d pt2=handcontour[(i-15+len)%len]-handcontour[i];
			pt1*=(1/cv::norm(pt1));
			pt2*=(1/cv::norm(pt2));
			angle.push_back(pt1.dot(pt2));//两个单位向量之内积为其夹角余弦
			//std::cout<<angle[i]<<std::endl;
		}else{
			angle.push_back(-1);
		}
#ifdef CSV_OUTPUT
		outf<<i<<','<<angle[i]<<','<<handcontour[i].x<<','
			<<handcontour[i].y<<std::endl;
#endif
	}
	
	typedef std::vector<double> Angles;
	std::vector<Angles> fine_angle_set;//一个符合良好手指形态的特征集
	Angles::const_iterator b=angle.begin(), e=angle.end(), pt_itr;
	while(b!=e){
		b=std::find_if(b,e,pangle);
		Angles::const_iterator next=b;
		if (b!=e)
			next=std::find_if(next,e,nangle);
		fine_angle_set.push_back(Angles(b,next));//fine_angle_set里面收藏的是张角余弦范围在[-0.2 1]的连续的点余弦值
		b=next;
	}
	if (fine_angle_set.empty()){
		throw std::invalid_argument("no fingertip feature");
	}else{
		//在fine_angle_set里面查找最长的手指特征
		std::vector<Angles>::const_iterator itr=std::max_element(fine_angle_set.begin(),fine_angle_set.end(),fine_angle_set_compare);
		//在最长的手指特征中选出张角余弦最大的
		pt_itr=std::max_element((*itr).begin(),(*itr).end());
	}
	//再次在原来的handcontour里面搜索那个被找出来的点（当然这里程序设计得不合理，浪费时间而且还可能会有错误
	int index=std::find(angle.begin(),angle.end(),(*pt_itr))-angle.begin();

#ifdef DEBUG_FIN
	cv::Mat imgshow=cv::Mat::zeros(480,640,CV_8UC1);
	std::vector<std::vector<cv::Point> > debugcontour;
	debugcontour.push_back(handcontour);
	cv::drawContours(imgshow,debugcontour,0,cv::Scalar(128),2);
	for(int i=0;i!=hull_index.size();i++){
		cv::circle(imgshow,handcontour[hull_index[i]],2,cv::Scalar(192));
	}
	cv::circle(imgshow,handcontour[hull_index[index]],5,cv::Scalar(255));
	cv::namedWindow("Debug");
	cv::imshow("Debug",imgshow);
	cv::waitKey(0);
#endif

	return index;
}

//函数名：FingertipPos
//隶属类：无
//功能：对当前输入的手势轮廓判定其指尖位置，所输入的手势轮廓只能包含一个指尖
//     是一个重载函数。
//     与上面那个唯一不同的就是对于y轴坐标在450以下的点是一个arbitrary threshold，因此这里改用计算handimg的y轴一阶矩确定threshold
//     我发现论文中写的是上面那个方法，但是实际使用是用的这个函数
//参数列表：
//handcontour   手势轮廓。显然可见，由一系列点组成。
//handimg       黑白二值手势图像
//返回参数： int 指示手指尖在handcontour中的index
//异常类：std::invalid_argument：当找不到手指特征时抛出
int FingertipPos(const std::vector<cv::Point>& handcontour, const cv::Mat& handimg){
	int len=handcontour.size();
	//std::vector<int> hull_index;
	std::vector<double> angle;
	//cv::convexHull(handcontour,hull_index);

#ifdef CSV_OUTPUT
	std::ofstream outf("out.csv");
	outf<<"index,angle,x,y"<<std::endl;
#endif

	cv::Moments mt=cv::moments(handimg,true);
	int level=mt.m01/mt.m00;//threshold确定使用了y轴一阶矩
	for (int i=0;i!=len;i++){
		if(handcontour[i].y<level){
			cv::Point2d pt1=handcontour[(i+15)%len]-handcontour[i];
			cv::Point2d pt2=handcontour[(i-15+len)%len]-handcontour[i];
			pt1*=(1/cv::norm(pt1));
			pt2*=(1/cv::norm(pt2));
			angle.push_back(pt1.dot(pt2));
			//std::cout<<angle[i]<<std::endl;
		}else{
			angle.push_back(-1);
		}
#ifdef CSV_OUTPUT
		outf<<i<<','<<angle[i]<<','<<handcontour[i].x<<','
			<<handcontour[i].y<<std::endl;
#endif
	}
	
	typedef std::vector<double> Angles;
	std::vector<Angles> fine_angle_set;
	Angles::const_iterator b=angle.begin(), e=angle.end(), pt_itr;
	while(b!=e){
		b=std::find_if(b,e,pangle);
		Angles::const_iterator next=b;
		if (b!=e)
			next=std::find_if(next,e,nangle);
		fine_angle_set.push_back(Angles(b,next));
		b=next;
	}
	if (fine_angle_set.empty()){
		throw std::invalid_argument("no fingertip feature");
	}else{
		//for (int i=0;i!=fine_angle_set.size();i++){
			
		std::vector<Angles>::const_iterator itr=std::max_element(fine_angle_set.begin(),fine_angle_set.end(),fine_angle_set_compare);
		pt_itr=std::max_element((*itr).begin(),(*itr).end());
	}
	int index=std::find(angle.begin(),angle.end(),(*pt_itr))-angle.begin();

#ifdef DEBUG_FIN
	cv::Mat imgshow=cv::Mat::zeros(480,640,CV_8UC1);
	std::vector<std::vector<cv::Point> > debugcontour;
	debugcontour.push_back(handcontour);
	cv::drawContours(imgshow,debugcontour,0,cv::Scalar(128),2);
	for(int i=0;i!=hull_index.size();i++){
		cv::circle(imgshow,handcontour[hull_index[i]],2,cv::Scalar(192));
	}
	cv::circle(imgshow,handcontour[hull_index[index]],5,cv::Scalar(255));
	cv::namedWindow("Debug");
	cv::imshow("Debug",imgshow);
	cv::waitKey(0);
#endif

	return index;
}