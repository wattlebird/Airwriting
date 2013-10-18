////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2013, Ronnie Wang�����̳ϣ�
// Permission of use under MIT License
//
// Filename     ��Fintipos.cpp
// Project Code ��
// Abstract     ��The implementation of the function to determine fingertip position
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
#include "Fintipos.h"
#include "opencv2\imgproc\imgproc.hpp"
#include <algorithm>
#include <iostream>
#include "opencv2\highgui\highgui.hpp"
#include <stdexcept>
#include <fstream>

//�����Ҫ�鿴�������ļ����㱻ѡΪ����������ģ��㣬��ȡ��ע����һ��
//#define DEBUG_FIN
//�����Ҫ���������Ϣ����ȡ��ע����һ��
//#define CSV_OUTPUT

//std::find_if��ν��
bool pangle(const double& angle){
	return (angle>-0.2);
}

//std::find_if��ν��
bool nangle(const double& angle){
	return !pangle(angle);
}

//std::max_element��ν��
bool fine_angle_set_compare(const std::vector<double>& a, const std::vector<double>& b){
	return a.size()<b.size();
}

//��������FingertipPos
//�����ࣺ��
//���ܣ��Ե�ǰ��������������ж���ָ��λ�ã����������������ֻ�ܰ���һ��ָ��
//�����б�
//handcontour   ������������Ȼ�ɼ�����һϵ�е���ɡ�
//���ز����� int ָʾ��ָ����handcontour�е�index
//�쳣�ࣺstd::invalid_argument�����Ҳ�����ָ����ʱ�׳�
int FingertipPos(const std::vector<cv::Point>& handcontour){
	int len=handcontour.size();
	//std::vector<int> hull_index;
	std::vector<double> angle;
	//cv::convexHull(handcontour,hull_index);

#ifdef CSV_OUTPUT
	std::ofstream outf("out.csv");
	outf<<"index,angle,x,y"<<std::endl;
#endif

	for (int i=0;i!=len;i++){//�������е�ÿһ����
		if(handcontour[i].y<450){//��y��������450���ڵ�
			cv::Point2d pt1=handcontour[(i+15)%len]-handcontour[i];
			cv::Point2d pt2=handcontour[(i-15+len)%len]-handcontour[i];
			pt1*=(1/cv::norm(pt1));
			pt2*=(1/cv::norm(pt2));
			angle.push_back(pt1.dot(pt2));//������λ����֮�ڻ�Ϊ��н�����
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
	std::vector<Angles> fine_angle_set;//һ������������ָ��̬��������
	Angles::const_iterator b=angle.begin(), e=angle.end(), pt_itr;
	while(b!=e){
		b=std::find_if(b,e,pangle);
		Angles::const_iterator next=b;
		if (b!=e)
			next=std::find_if(next,e,nangle);
		fine_angle_set.push_back(Angles(b,next));//fine_angle_set�����ղص����Ž����ҷ�Χ��[-0.2 1]�������ĵ�����ֵ
		b=next;
	}
	if (fine_angle_set.empty()){
		throw std::invalid_argument("no fingertip feature");
	}else{
		//��fine_angle_set������������ָ����
		std::vector<Angles>::const_iterator itr=std::max_element(fine_angle_set.begin(),fine_angle_set.end(),fine_angle_set_compare);
		//�������ָ������ѡ���Ž���������
		pt_itr=std::max_element((*itr).begin(),(*itr).end());
	}
	//�ٴ���ԭ����handcontour���������Ǹ����ҳ����ĵ㣨��Ȼ���������Ƶò������˷�ʱ����һ����ܻ��д���
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

//��������FingertipPos
//�����ࣺ��
//���ܣ��Ե�ǰ��������������ж���ָ��λ�ã����������������ֻ�ܰ���һ��ָ��
//     ��һ�����غ�����
//     �������Ǹ�Ψһ��ͬ�ľ��Ƕ���y��������450���µĵ���һ��arbitrary threshold�����������ü���handimg��y��һ�׾�ȷ��threshold
//     �ҷ���������д���������Ǹ�����������ʵ��ʹ�����õ��������
//�����б�
//handcontour   ������������Ȼ�ɼ�����һϵ�е���ɡ�
//handimg       �ڰ׶�ֵ����ͼ��
//���ز����� int ָʾ��ָ����handcontour�е�index
//�쳣�ࣺstd::invalid_argument�����Ҳ�����ָ����ʱ�׳�
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
	int level=mt.m01/mt.m00;//thresholdȷ��ʹ����y��һ�׾�
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