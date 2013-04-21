#ifndef GUARD_PARTICLE_H
#define GUARD_PARTICLE_H

#include <iostream>
#include <vector>
#include "opencv2\core\core.hpp"

class Particle{
public:
	Particle():particleNum(0){};
	explicit Particle(int num, cv::Mat dym=cv::Mat::eye(cv::Size(5,5),CV_64FC1)){
		particleNum=num;
		particleStates=std::vector<cv::Vec<double,5> >(num);
		particleConfidence=std::vector<double>(num,1./num);
		//particleCumulative=std::vector<double>(num);
		//for (int i=1;i!=num;i++)
		//	particleCumulative[i]=particleCumulative[i-1]+particleConfidence[i];
		dynModel=dym;
		unsigned int temp=cv::getTickCount();
		randobj=cv::RNG::RNG(temp);
	}

	//初始化中，首先进行洞的填充。
	//然后使用某些方法确定手指尖位置。
	//基于已确定的手指尖位置，抽取点作为模板
	//然后是填充状态向量的工作。
	void InitParticle(const cv::Mat&);
	//对每个粒子要扩散几次进行判断
	//使用动态模型预测粒子行为。
	void PredictParticle();
	//利用图像的信息计算粒子置信度。
	//计算曲线上的点；
	//计算点上的法线；
	//根据法线上的梯度计算置信度。
	//归一化置信度。
	void MeasureParticle(const cv::Mat&, bool&);
	cv::Point MeasuredFingertip() const;

private:
	//这个形式应该是
	cv::Vec<double,7> templatePointSetx;
	cv::Vec<double,7> templatePointSety;
	cv::Vec<double,14> templateControlPoint;
	cv::Mat W;

	int particleNum;
	std::vector<cv::Vec<double,5> > particleStates;
	std::vector<double> particleConfidence;
	//std::vector<double> particleCumulative;
	cv::Mat dynModel;
	cv::RNG randobj;
};

bool contours_compare(const std::vector<cv::Point>,
	const std::vector<cv::Point>);
bool fingertip_compare(const cv::Point, const cv::Point);

//用以判断状态是否有效，以及与一个状态值对应的顶点是否有效。如果无效，则置信度归零。
bool isValid(const cv::Vec<double,5>&, const std::vector<cv::Point2i>&);

#endif