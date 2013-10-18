////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2013, Ronnie Wang（王程诚）
// Permission of use under MIT License
//
// Filename     ：Particle.cpp
// Project Code ：
// Abstract     ：The implementation of the Class Particle
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
#include "Particle.h"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\highgui\highgui.hpp"
#include <algorithm>
#include <cmath>
#include "ContourConfidence.h"
#include "Fintipos.h"

//用于调试用于标识指尖轮廓的点
//#define DEBUG_INIT


///
//方法名：InitParticle
//隶属类：Particle
//参数列表
//handimg   带有手势的黑白二值图像
//功能：利用手势信息提取手指轮廓，用手指轮廓初始化粒子particleStates
//     基于已确定的手指尖位置，抽取点作为模板
//     然后是填充状态向量的工作。
void Particle::InitParticle(const cv::Mat& handimg){
	std::vector<std::vector<cv::Point> > contours;//cv::findContours参数
	std::vector<cv::Vec4i> hierarchy;//cv::findContours表示的轮廓之间包含关系表
	std::vector<cv::Point> handcontour;//最终要提取的手势轮廓
#ifdef DEBUG_INIT
	cv::Mat imgshow=handimg.clone();
#endif

	cv::findContours( handimg, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );

	//选取最长轮廓为手势轮廓
	if (hierarchy[0][0]>=0 || hierarchy[0][1]>=0){//同一等级有多个并列关系的轮廓
		std::vector<std::vector<cv::Point> > temp;
		int contour_index=0;
		do{
			temp.push_back(contours[contour_index]);
			contour_index=hierarchy[contour_index][0];
		}while(contour_index>0);
		std::vector<std::vector<cv::Point> >::iterator itr=std::max_element(temp.begin(),temp.end(),contours_compare);
		handcontour=(*itr);
	}else
		handcontour=contours[0];
#ifdef DEBUG_INIT
	std::vector<std::vector<cv::Point> > debugcontour;
	debugcontour.push_back(handcontour);
	cv::drawContours(imgshow,debugcontour,0,cv::Scalar(128),2);
#endif

	//调用函数FingertipPos，确定手指尖位置
	int min_index=FingertipPos(handcontour,handimg);

	//std::vector<cv::Point>::iterator itr=std::min_element(handcontour.begin(),handcontour.end(),fingertip_compare);
	//int min_index=itr-handcontour.begin();

	//确定用于表示轮廓的六个点
	int len=handcontour.size();
	for (int i=-CONTOUR_POINTS/2;i!=CONTOUR_POINTS-CONTOUR_POINTS/2;i++){
		templatePointSetx[i+CONTOUR_POINTS/2]=handcontour[(min_index+4*i+2*len)%len].x-handcontour[(min_index)%len].x;
		templatePointSety[i+CONTOUR_POINTS/2]=handcontour[(min_index+4*i+2*len)%len].y-handcontour[(min_index)%len].y;
		templateControlPoint[i+CONTOUR_POINTS/2]=templatePointSetx[i+CONTOUR_POINTS/2];
		templateControlPoint[i+CONTOUR_POINTS/2+CONTOUR_POINTS]=templatePointSety[i+CONTOUR_POINTS/2];
#ifdef DEBUG_INIT
		cv::circle(imgshow,handcontour[(min_index+10*i+2*len)%len],8,cv::Scalar(192),2);
#endif
	}
#ifdef DEBUG_INIT
	cv::namedWindow("debug window 1", CV_WINDOW_AUTOSIZE);
	cv::imshow("debug window 1",imgshow);
	cv::waitKey(0);
	//cv::imwrite("handcontour.jpg",imgshow);
#endif
	
	//初始化形状矩阵W
	if(!W.empty())
		W.pop_back(2*CONTOUR_POINTS);
	for (int i=0;i!=(2*CONTOUR_POINTS);i++){
		if(i<CONTOUR_POINTS){
			double temp[]={1,0,templatePointSetx[i%CONTOUR_POINTS],0,0,templatePointSety[i%CONTOUR_POINTS]};
			W.push_back(cv::Mat(1,6,CV_64F,temp));
		}
		else{
			double temp[]={0,1,0,templatePointSety[i%CONTOUR_POINTS],templatePointSetx[i%CONTOUR_POINTS],0};
			W.push_back(cv::Mat(1,6,CV_64F,temp));
		}
	}
	//std::cout<<"W="<<W<<std::endl<<std::endl;
	

	cv::Vec<double,9> init_state(handcontour[min_index].x,handcontour[min_index].x,handcontour[min_index].x,
		handcontour[min_index].y,handcontour[min_index].y,handcontour[min_index].y,
		0,
		1,1);
	particleStates=std::vector<cv::Vec<double,9> >(particleNum,init_state);
}

//bool fingertip_compare(const cv::Point pt1,
//	const cv::Point pt2){
//		return pt1.y<pt2.y;
//}


//用于std::max_element的谓词
bool contours_compare(const std::vector<cv::Point> obj1,
	const std::vector<cv::Point> obj2){
		return obj1.size()<obj2.size();
}


///
//方法名：PredictParticle
//隶属方法：Particle
//功能：粒子滤波的预测环节。使用本Particle类定义的状态转移矩阵预测下一帧粒子
//     对每个粒子要扩散几次进行判断
//     使用动态模型预测粒子行为。
void Particle::PredictParticle(){
	//本帧生成的粒子要根据上一帧的粒子置信度决定。上一帧置信度低的粒子相应地生成的粒子也少，反之则多。
	std::vector<int> nextround(particleNum);
	for (int i=0;i!=particleNum;i++)
		nextround[i]=int(particleNum*particleConfidence[i]);
	int totalsum=cv::sum(nextround)[0];
	//如果生成的粒子数因为取整的原因少于全部粒子数，则将最大置信度的粒子再生成。
	if (particleNum-totalsum){
		std::vector<int>::iterator next_itr=std::max_element(nextround.begin(),nextround.end());
		(*next_itr)+=(particleNum-totalsum);
	}

	//对每个粒子进行状态转移，详见论文3.3
	std::vector<cv::Vec<double,9> > temp(particleNum);
	int k=0;
	for (int i=0;i!=particleNum;i++)
		if(nextround[i])
			for (int j=0;j!=nextround[i];j++){
				cv::Vec<double,9> randomVec(randobj.gaussian(5),0,0,
					randobj.gaussian(5),0,0,
					randobj.gaussian(0.1),//弧度！
					randobj.gaussian(0.01),
					randobj.gaussian(0.01)
					);
			
				cv::gemm(dynModel,particleStates[i],1.0,randomVec,1.0,temp[k]);
				k++;
				
			}
	particleStates=temp;

}

///
//方法名：MeasureParticle
//隶属类：Particle
//功能：执行粒子滤波的更新环节
//参数列表：
//img     本回的黑白帧差图像
//返回值：bool 跟踪跟踪成功就返回true
//功能：利用图像的信息计算粒子置信度。
//     计算曲线上的点；
//     计算点上的法线；
//     根据法线上的梯度计算置信度。
//     归一化置信度。
//     返回布尔值，代表粒子跟踪成功（从置信度上看）
//本方法问题颇多，需要推倒重写
bool Particle::MeasureParticle(const cv::Mat& img){
	cv::Canny(img,img,50,150);
	std::vector<cv::Vec<double, 2*CONTOUR_POINTS> > controlPoints(particleNum);
	//cv::namedWindow("debug window 1", CV_WINDOW_AUTOSIZE);

	for (int i=0;i!=particleNum;i++){
		//“S”是运动状态，详见论文3-4式
		cv::Mat S=(cv::Mat_<double>(6,1)<<
			particleStates[i][0],
			particleStates[i][3],
			particleStates[i][7]*(std::cos(particleStates[i][6]))-1,
			particleStates[i][8]*(std::cos(particleStates[i][6]))-1,
			-particleStates[i][8]*(std::sin(particleStates[i][6])),
			particleStates[i][7]*(std::sin(particleStates[i][6])));
		cv::gemm(W,S,1,templateControlPoint,1,controlPoints[i]);
		std::vector<cv::Point> affinedPoints(CONTOUR_POINTS);//根据每个粒子描述的运动状态，我们作出仿射变换后的点，就是经过预测的手指轮廓
		for (int j=0;j!=CONTOUR_POINTS;j++){//完成一组描述轮廓的点，这个点是经过模板仿射变换后得到的
			affinedPoints[j].x=controlPoints[i][j];
			affinedPoints[j].y=controlPoints[i][j+CONTOUR_POINTS];
		}
		if (isValid(particleStates[i],affinedPoints)){//如果本粒子对应的轮廓合法
			particleConfidence[i]=ContConf(affinedPoints,img);//使用ContConf计算在当前场景中粒子所对应轮廓的置信度
		}else{
			particleConfidence[i]=0;//该粒子置信度为0
		}
		//std::cout<<particleConfidence[i]<<std::endl;
		//用于调试
		//cv::Mat imgshow=img.clone();
		//for(int j=0;j!=CONTOUR_POINTS;j++){
		//	cv::circle(imgshow,affinedPoints[j],1,cv::Scalar(0,0,0));
		//}
		//cv::imshow("debug window 1",imgshow);
		//cv::waitKey(0);
	}
	
	int flag=0;
	for(int i=0;i!=particleNum;i++){
		if(particleConfidence[i]<0.3){//置信度过小就把置信度置0
			particleConfidence[i]=0;
			flag++;
		}
	}
	cv::normalize(particleConfidence,particleConfidence,1.0,0.0,cv::NORM_L1);
	if (flag!=particleNum)//如果所有粒子的置信度都为0，即为跟踪失败
		return true;
	else 
		return false;
}


///
//测试当前粒子对应的轮廓点是否合法
//参数列表
//state         粒子（状态）
//actualPoints  当前粒子对应的用于描述轮廓的点
bool isValid(const cv::Vec<double,9>& state, const std::vector<cv::Point>& actualPoints){
	//state[0]，state[3]描述了轮廓点的基准位置，分别对应于x坐标和y坐标
	//显然粒子不能超过摄像区域
	//state[7]、state[8]是尺度变换大小。显然尺度变换大小不能为负值
	if (state[0]>=0 && state[0]<640 && state[3]>=0 && state[3]<480 && state[7]>0 && state[8]>0){
		std::vector<cv::Point>::const_iterator itr=actualPoints.begin();//迭代器是对一个指针的精巧包装
		//对描述轮廓的每一个点，其坐标都不能超过摄像区域
		while(itr!=actualPoints.end() && (*itr).x>=0 && (*itr).x<640 && (*itr).y>=0 && (*itr).y<480)
			itr++;
				if (itr==actualPoints.end())
					return true;
	}
	return false;
}

///
//方法名：MeasuredFingertip
//隶属类：Paticle
//功能：计算出各粒子的期望值，其中心位置指示指尖位置
//参数列表
//void
//返回参数 cv:Point 根据当前粒子置信度和粒子状态计算出来的指尖位置
cv::Point Particle::MeasuredFingertip() const {
	double x=0,y=0;
	for(int i=0;i!=particleNum;i++){
		x+=particleConfidence[i]*particleStates[i][0];
		y+=particleConfidence[i]*particleStates[i][3];
	}
	return cv::Point2i((int)x,(int)y);
}
