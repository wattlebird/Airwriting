////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2013, Ronnie Wang�����̳ϣ�
// Permission of use under MIT License
//
// Filename     ��Particle.h
// Project Code ��
// Abstract     ��The declaration of the Class Particle
// Reference    ��
//
// Version      ��1.0
// Author       ��Ronnie Wang�����̳ϣ�
// Accomplished date �� 2013.06
//
// Replaced version  :
// Original Author   :
// Accomplished date : 
//
////////////////////////////////////////////////////////////////////////////
#ifndef GUARD_PARTICLE_H
#define GUARD_PARTICLE_H

#define CONTOUR_POINTS 6 //��6���������ָ����

#include <iostream>
#include <vector>
#include "opencv2\core\core.hpp"

//�����ƣ�Particle
//���ܣ���Ϊ�����˲���Ҫ���ܵ�ʵ���࣬������ʼ����Ԥ�⡢����
class Particle{
public:
	Particle():particleNum(0){};//Ĭ�Ϲ�����

	///
	//�Զ��幹����
	//�����б�
	//num    ���γ��������������˲������Ӹ�����
	//dym    ���������˲�Ԥ��׶εĶ�̬ģ�;���Ĭ��Ϊ��λ�����⼴��һ֡��Ԥ�����������һ֡��ͬ��
	explicit Particle(int num, cv::Mat dym=cv::Mat::eye(cv::Size(9,9),CV_64FC1)){
		particleNum=num;
		particleStates=std::vector<cv::Vec<double,9> >(num);
		particleConfidence=std::vector<double>(num,1./num);//��ʼ��ÿ���������Ŷȶ����
		//particleCumulative=std::vector<double>(num);
		//for (int i=1;i!=num;i++)
		//	particleCumulative[i]=particleCumulative[i-1]+particleConfidence[i];
		dynModel=dym;

		//�����г�ʼ�������������
		unsigned int temp=cv::getTickCount();
		randobj=cv::RNG::RNG(temp);
	}

	void InitParticle(const cv::Mat&);
	void PredictParticle();
	bool MeasureParticle(const cv::Mat&);
	cv::Point MeasuredFingertip() const;

private:
	cv::Vec<double,CONTOUR_POINTS> templatePointSetx;//x�᷽���ϵ�ģ��㡣��ν��ģ�塱�������ڳ�ʼ���׶δ�������ȡ��������ָ�����ϵ�һϵ�е㡣�����ģ�塱����Ϊ�Ժ�ʶ��Ļ�����
	cv::Vec<double,CONTOUR_POINTS> templatePointSety;//y�᷽���ϵ�ģ��㡣
	cv::Vec<double,2*CONTOUR_POINTS> templateControlPoint;//templateControlPoint=[templatePointSetx' templatePointSety']' "'"����ת�á���Ȼ�ɼ���templatePointSetx��templatePointSety������ֻ��������Ե���ʱ����
	cv::Mat W;//��״����

	int particleNum;//������Ŀ
	std::vector<cv::Vec<double,9> > particleStates;//���ӵļ��ϡ������˲������У������ĸ�����ǡ����ӡ������������з���ǿ���������ӡ�ԭ��Ӧ������һ֡��ͬ�Ŀ��ܵ���ָ����λ�úͷ��򣬵��Ǳ���������һ��״ֵ̬�����ǡ����ӡ���
	                                               //һ��������һ��9ά��������ÿһ��Ԫ��Ϊ[x_t x_t-1 x_t-2 y_t y_t-1 y_t-2 �� s_x s_y]'
	                                               //��_�����±꣬t��ʱ�䣬���Ǳ����Ӷ�Ӧ��������ģ�����ת�Ƕȡ�s_x�Ƿ���任��x�����ϵ���������s_y�Ƿ���任��y�����ϵ�������
	std::vector<double> particleConfidence;//�������Ŷȣ������˱�֡��ʵ����ָ������Ԥ��������ƥ��ĳ̶ȡ�
	//std::vector<double> particleCumulative;
	cv::Mat dynModel;//����Ԥ��׶εĶ�̬ģ��
	cv::RNG randobj;//�����������
};

bool contours_compare(const std::vector<cv::Point>,
	const std::vector<cv::Point>);
bool isValid(const cv::Vec<double,9>&, const std::vector<cv::Point>&);

#endif