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

	//��ʼ���У����Ƚ��ж�����䡣
	//Ȼ��ʹ��ĳЩ����ȷ����ָ��λ�á�
	//������ȷ������ָ��λ�ã���ȡ����Ϊģ��
	//Ȼ�������״̬�����Ĺ�����
	void InitParticle(const cv::Mat&);
	//��ÿ������Ҫ��ɢ���ν����ж�
	//ʹ�ö�̬ģ��Ԥ��������Ϊ��
	void PredictParticle();
	//����ͼ�����Ϣ�����������Ŷȡ�
	//���������ϵĵ㣻
	//������ϵķ��ߣ�
	//���ݷ����ϵ��ݶȼ������Ŷȡ�
	//��һ�����Ŷȡ�
	void MeasureParticle(const cv::Mat&, bool&);
	cv::Point MeasuredFingertip() const;

private:
	//�����ʽӦ����
	cv::Vec<double,7> templatePointSetx;
	cv::Vec<double,7> templatePointSety;
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

//�����ж�״̬�Ƿ���Ч���Լ���һ��״ֵ̬��Ӧ�Ķ����Ƿ���Ч�������Ч�������Ŷȹ��㡣
bool isValid(const cv::Vec<double,5>&, const std::vector<cv::Point2i>&);

#endif