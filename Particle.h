////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2013, Ronnie Wang（王程诚）
// Permission of use under MIT License
//
// Filename     ：Particle.h
// Project Code ：
// Abstract     ：The declaration of the Class Particle
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
#ifndef GUARD_PARTICLE_H
#define GUARD_PARTICLE_H

#define CONTOUR_POINTS 6 //用6个点描绘手指轮廓

#include <iostream>
#include <vector>
#include "opencv2\core\core.hpp"

//类名称：Particle
//功能：作为粒子滤波主要功能的实现类，包括初始化、预测、更新
class Particle{
public:
	Particle():particleNum(0){};//默认构造器

	///
	//自定义构造器
	//参数列表：
	//num    本次程序运行中粒子滤波的粒子个数。
	//dym    用于粒子滤波预测阶段的动态模型矩阵，默认为单位矩阵，意即下一帧的预测的粒子与这一帧相同。
	explicit Particle(int num, cv::Mat dym=cv::Mat::eye(cv::Size(9,9),CV_64FC1)){
		particleNum=num;
		particleStates=std::vector<cv::Vec<double,9> >(num);
		particleConfidence=std::vector<double>(num,1./num);//初始化每个粒子置信度都相等
		//particleCumulative=std::vector<double>(num);
		//for (int i=1;i!=num;i++)
		//	particleCumulative[i]=particleCumulative[i-1]+particleConfidence[i];
		dynModel=dym;

		//这两行初始化随机数发生器
		unsigned int temp=cv::getTickCount();
		randobj=cv::RNG::RNG(temp);
	}

	void InitParticle(const cv::Mat&);
	void PredictParticle();
	bool MeasureParticle(const cv::Mat&);
	cv::Point MeasuredFingertip() const;

private:
	cv::Vec<double,CONTOUR_POINTS> templatePointSetx;//x轴方向上的模板点。所谓“模板”，就是在初始化阶段从手势提取出来的手指轮廓上的一系列点。这个“模板”将成为以后识别的基础。
	cv::Vec<double,CONTOUR_POINTS> templatePointSety;//y轴方向上的模板点。
	cv::Vec<double,2*CONTOUR_POINTS> templateControlPoint;//templateControlPoint=[templatePointSetx' templatePointSety']' "'"代表转置。显然可见，templatePointSetx和templatePointSety充其量只是这个属性的临时变量
	cv::Mat W;//形状矩阵

	int particleNum;//粒子数目
	std::vector<cv::Vec<double,9> > particleStates;//粒子的集合。整个滤波过程中，最抽象的概念就是“粒子”。我在论文中反复强调，“粒子”原本应该是下一帧不同的可能的手指轮廓位置和方向，但是被简化描述成一组状态值，就是“粒子”。
	                                               //一个粒子是一个9维向量，其每一个元素为[x_t x_t-1 x_t-2 y_t y_t-1 y_t-2 θ s_x s_y]'
	                                               //“_”是下标，t是时间，θ是本粒子对应的轮廓较模板的旋转角度。s_x是仿射变换在x方向上的伸缩量，s_y是仿射变换在y方向上的伸缩量
	std::vector<double> particleConfidence;//粒子置信度，表征了本帧中实际手指轮廓与预测轮廓相匹配的程度。
	//std::vector<double> particleCumulative;
	cv::Mat dynModel;//用于预测阶段的动态模型
	cv::RNG randobj;//随机数发生器
};

bool contours_compare(const std::vector<cv::Point>,
	const std::vector<cv::Point>);
bool isValid(const cv::Vec<double,9>&, const std::vector<cv::Point>&);

#endif