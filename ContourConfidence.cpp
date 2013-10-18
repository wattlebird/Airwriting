////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2013, Ronnie Wang（王程诚）
// Permission of use under MIT License
//
// Filename     ：ContourConfidence.cpp
// Project Code ：
// Abstract     ：The implementation of the function to determine contour confidence
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
#include "ContourConfidence.h"
#include <cmath>
#include <stdexcept>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>


//如果想显示本粒子用于计算置信度的采样，请取消下一行注释
//#define DEBUG_CONTOUR

double confcal(const cv::Mat&, const cv::Point&, const cv::Point2d&, const double nlen=4.0);
double confcal_debug(cv::Mat&, const cv::Mat&, const cv::Point&, const cv::Point2d&, const double nlen=4.0);

///
//函数名：ContConf
//隶属类：无
//功能：计算由一系列点代表的轮廓的、在当前图像上的置信度
//参数列表：
//corse_contour   轮廓，由一系列点组成。由于这几个点只是完整的轮廓上间隔地取的点，所以说是“粗糙的”
//img             展示出手轮廓边缘的黑白图片
//输出参数 double 轮廓对应的粒子的置信度
//包含函数：confcal
//		  confcal_debug
double ContConf(const std::vector<cv::Point>& corse_contour, const cv::Mat& img){
	int len=corse_contour.size();
	//这段代码的用意是：计算出每个点在轮廓上的法线单位向量，存储在normal中
	//没有使用贝塞尔曲线之类的
	//原理很简单，一个点的法线向量就在它与两个相邻点夹角的角平分线上。
	//对于初始点，由于再前面没有点，其法线方向就是它与它后面的点连线的法线方向
	//对于末尾点，由于再后面没有点，其法线方向就是它与它前面的点连线的法线方向
	std::vector<cv::Point2d> normal(len);
	std::vector<cv::Point>::const_iterator src_itr=corse_contour.begin();
	std::vector<cv::Point2d>::iterator dest_itr=normal.begin();
	for (int i=0;i!=len;i++){
		cv::Point2d pt1,pt2;//两个单位向量，代表所考察的点与其左右的点的连线的单位向量
		if(i==0)//如果是初始点
			pt1=cv::Point2d(0,0);
		else{
			pt1=src_itr[i]-src_itr[i-1];
			pt1*=(1/cv::norm(pt1));
		}
		if(i==len-1)//如果是末尾点
			pt2=cv::Point2d(0,0);
		else{
			pt2=src_itr[i+1]-src_itr[i];
			pt2*=(1/cv::norm(pt2));
		}
		dest_itr[i]=pt1-pt2;//两个单位向量相减就是法线方向的垂直方向。哦我发现这里可以简化……
		if((dest_itr[i].x)||(dest_itr[i].y))
			dest_itr[i]*=(1/cv::norm(dest_itr[i]));
		else//如果法线向量为零向量
			dest_itr[i]=cv::Point2d(pt1.y,-pt1.x);
		if(i==0 || i==(len-1))
			dest_itr[i]=cv::Point2d(dest_itr[i].y,-dest_itr[i].x);
	}


	
#ifdef DEBUG_CONTOUR
	cv::Mat imgshow=img.clone();
	for(int j=0;j!=corse_contour.size();j++){
		cv::circle(imgshow,corse_contour[j],1,cv::Scalar(0,0,0));
	}
#endif

	double dis=0;
	for (int i=0;i!=len;i++){
		try{
#ifdef DEBUG_CONTOUR
			dis+=confcal_debug(imgshow,img,corse_contour[i],normal[i]);
#else
			dis+=confcal(img, corse_contour[i], normal[i]);
#endif
		}catch(std::exception& e){
			if (e.what()=="points are out of pic's range"){
				dis+=0;
				continue;
			}
		}
	}
#ifdef DEBUG_CONTOUR
	cv::namedWindow("debug no1");
	cv::imshow("debug no1",imgshow);
	cv::waitKey(0);
#endif
	dis/=(2*len*1.2);//置信度乘以一个常数，这个常数是Matlab计算和实践试验出来的
	return std::exp(-dis);
}

///
//函数名：confcal
//隶属类：无
//功能：计算在给定的轮廓图片中，一个点在法线距离上偏离轮廓多少
//参数列表
//img   一个黑白轮廓图片
//pt    一个轮廓上的点
//n     法线向量
//nlen  从这个点开始，在法线距离上搜索多远
double confcal(const cv::Mat& img, const cv::Point& pt, const cv::Point2d& n, const double nlen){
	cv::Point unitnormal;
	unitnormal.x=int(n.x*nlen+0.5);
	unitnormal.y=int(n.y*nlen+0.5);
	cv::Point pt1=pt+unitnormal;//从这个点开始，在法线的一个方向上搜索的尽头
	cv::Point pt2=pt-unitnormal;//从这个点开始，在法线的另一个方向上搜索的终点
	double width=img.cols;
	double height=img.rows;
	if (pt1.x<0 || pt1.x>=width || pt1.y<0 || pt1.y>=height || pt2.x<0 || pt2.x>=width || pt2.y<0 || pt2.y>=height)
		throw std::invalid_argument("points are out of pic's range");

	cv::LineIterator normal_itr1(img,pt,pt1);
	cv::LineIterator normal_itr2(img,pt,pt2);
	int i,j;
	for (i=0;i!=normal_itr1.count;i++,normal_itr1++)
		if ((**normal_itr1)==255)//当遇到像素为255的点的时候，就是到了真正轮廓的边缘
			break;

	for (j=0;j!=normal_itr2.count;j++,normal_itr2++)
		if ((**normal_itr2)==255)
			break;
	
	double dis1=cv::norm(normal_itr1.pos()-pt);
	double dis2=cv::norm(normal_itr2.pos()-pt);

	return dis1<dis2 ? dis1 : dis2;

}

///
//函数名：confcal
//隶属类：无
//功能：计算在给定的轮廓图片中，一个点在法线距离上偏离轮廓多少
//     和confcal基本相同，不过就是输出了可以显示法线的图像，用于调试
//参数列表
//imgshow   带有法线和轮廓的图片，用于输出
//img       一个黑白轮廓图片
//pt        一个轮廓上的点
//n         法线向量
//nlen      从这个点开始，在法线距离上搜索多远
double confcal_debug(cv::Mat& imgshow, const cv::Mat& img, const cv::Point& pt, const cv::Point2d& n, const double nlen){
	cv::Point unitnormal;
	unitnormal.x=int(n.x*nlen+0.5);
	unitnormal.y=int(n.y*nlen+0.5);
	cv::Point pt1=pt+unitnormal;
	cv::Point pt2=pt-unitnormal;
	cv::line(imgshow,pt1,pt2,cv::Scalar(255,255,255),2);
	double width=img.cols;
	double height=img.rows;
	if (pt1.x<0 || pt1.x>=width || pt1.y<0 || pt1.y>=height || pt2.x<0 || pt2.x>=width || pt2.y<0 || pt2.y>=height)
		throw std::invalid_argument("points are out of pic's range");

	cv::LineIterator normal_itr1(img,pt,pt1);
	cv::LineIterator normal_itr2(img,pt,pt2);
	int i,j;
	for (i=0;i!=normal_itr1.count;i++,normal_itr1++)
		if ((**normal_itr1)==255)
			break;

	for (j=0;j!=normal_itr2.count;j++,normal_itr2++)
		if ((**normal_itr2)==255)
			break;
	

	double dis1=cv::norm(normal_itr1.pos()-pt);
	double dis2=cv::norm(normal_itr2.pos()-pt);

	return dis1<dis2 ? dis1 : dis2;

}