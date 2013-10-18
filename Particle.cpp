////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2013, Ronnie Wang�����̳ϣ�
// Permission of use under MIT License
//
// Filename     ��Particle.cpp
// Project Code ��
// Abstract     ��The implementation of the Class Particle
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
#include "Particle.h"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\highgui\highgui.hpp"
#include <algorithm>
#include <cmath>
#include "ContourConfidence.h"
#include "Fintipos.h"

//���ڵ������ڱ�ʶָ�������ĵ�
//#define DEBUG_INIT


///
//��������InitParticle
//�����ࣺParticle
//�����б�
//handimg   �������Ƶĺڰ׶�ֵͼ��
//���ܣ�����������Ϣ��ȡ��ָ����������ָ������ʼ������particleStates
//     ������ȷ������ָ��λ�ã���ȡ����Ϊģ��
//     Ȼ�������״̬�����Ĺ�����
void Particle::InitParticle(const cv::Mat& handimg){
	std::vector<std::vector<cv::Point> > contours;//cv::findContours����
	std::vector<cv::Vec4i> hierarchy;//cv::findContours��ʾ������֮�������ϵ��
	std::vector<cv::Point> handcontour;//����Ҫ��ȡ����������
#ifdef DEBUG_INIT
	cv::Mat imgshow=handimg.clone();
#endif

	cv::findContours( handimg, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );

	//ѡȡ�����Ϊ��������
	if (hierarchy[0][0]>=0 || hierarchy[0][1]>=0){//ͬһ�ȼ��ж�����й�ϵ������
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

	//���ú���FingertipPos��ȷ����ָ��λ��
	int min_index=FingertipPos(handcontour,handimg);

	//std::vector<cv::Point>::iterator itr=std::min_element(handcontour.begin(),handcontour.end(),fingertip_compare);
	//int min_index=itr-handcontour.begin();

	//ȷ�����ڱ�ʾ������������
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
	
	//��ʼ����״����W
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


//����std::max_element��ν��
bool contours_compare(const std::vector<cv::Point> obj1,
	const std::vector<cv::Point> obj2){
		return obj1.size()<obj2.size();
}


///
//��������PredictParticle
//����������Particle
//���ܣ������˲���Ԥ�⻷�ڡ�ʹ�ñ�Particle�ඨ���״̬ת�ƾ���Ԥ����һ֡����
//     ��ÿ������Ҫ��ɢ���ν����ж�
//     ʹ�ö�̬ģ��Ԥ��������Ϊ��
void Particle::PredictParticle(){
	//��֡���ɵ�����Ҫ������һ֡���������ŶȾ�������һ֡���Ŷȵ͵�������Ӧ�����ɵ�����Ҳ�٣���֮��ࡣ
	std::vector<int> nextround(particleNum);
	for (int i=0;i!=particleNum;i++)
		nextround[i]=int(particleNum*particleConfidence[i]);
	int totalsum=cv::sum(nextround)[0];
	//������ɵ���������Ϊȡ����ԭ������ȫ������������������Ŷȵ����������ɡ�
	if (particleNum-totalsum){
		std::vector<int>::iterator next_itr=std::max_element(nextround.begin(),nextround.end());
		(*next_itr)+=(particleNum-totalsum);
	}

	//��ÿ�����ӽ���״̬ת�ƣ��������3.3
	std::vector<cv::Vec<double,9> > temp(particleNum);
	int k=0;
	for (int i=0;i!=particleNum;i++)
		if(nextround[i])
			for (int j=0;j!=nextround[i];j++){
				cv::Vec<double,9> randomVec(randobj.gaussian(5),0,0,
					randobj.gaussian(5),0,0,
					randobj.gaussian(0.1),//���ȣ�
					randobj.gaussian(0.01),
					randobj.gaussian(0.01)
					);
			
				cv::gemm(dynModel,particleStates[i],1.0,randomVec,1.0,temp[k]);
				k++;
				
			}
	particleStates=temp;

}

///
//��������MeasureParticle
//�����ࣺParticle
//���ܣ�ִ�������˲��ĸ��»���
//�����б�
//img     ���صĺڰ�֡��ͼ��
//����ֵ��bool ���ٸ��ٳɹ��ͷ���true
//���ܣ�����ͼ�����Ϣ�����������Ŷȡ�
//     ���������ϵĵ㣻
//     ������ϵķ��ߣ�
//     ���ݷ����ϵ��ݶȼ������Ŷȡ�
//     ��һ�����Ŷȡ�
//     ���ز���ֵ���������Ӹ��ٳɹ��������Ŷ��Ͽ���
//�����������Ķ࣬��Ҫ�Ƶ���д
bool Particle::MeasureParticle(const cv::Mat& img){
	cv::Canny(img,img,50,150);
	std::vector<cv::Vec<double, 2*CONTOUR_POINTS> > controlPoints(particleNum);
	//cv::namedWindow("debug window 1", CV_WINDOW_AUTOSIZE);

	for (int i=0;i!=particleNum;i++){
		//��S�����˶�״̬���������3-4ʽ
		cv::Mat S=(cv::Mat_<double>(6,1)<<
			particleStates[i][0],
			particleStates[i][3],
			particleStates[i][7]*(std::cos(particleStates[i][6]))-1,
			particleStates[i][8]*(std::cos(particleStates[i][6]))-1,
			-particleStates[i][8]*(std::sin(particleStates[i][6])),
			particleStates[i][7]*(std::sin(particleStates[i][6])));
		cv::gemm(W,S,1,templateControlPoint,1,controlPoints[i]);
		std::vector<cv::Point> affinedPoints(CONTOUR_POINTS);//����ÿ�������������˶�״̬��������������任��ĵ㣬���Ǿ���Ԥ�����ָ����
		for (int j=0;j!=CONTOUR_POINTS;j++){//���һ�����������ĵ㣬������Ǿ���ģ�����任��õ���
			affinedPoints[j].x=controlPoints[i][j];
			affinedPoints[j].y=controlPoints[i][j+CONTOUR_POINTS];
		}
		if (isValid(particleStates[i],affinedPoints)){//��������Ӷ�Ӧ�������Ϸ�
			particleConfidence[i]=ContConf(affinedPoints,img);//ʹ��ContConf�����ڵ�ǰ��������������Ӧ���������Ŷ�
		}else{
			particleConfidence[i]=0;//���������Ŷ�Ϊ0
		}
		//std::cout<<particleConfidence[i]<<std::endl;
		//���ڵ���
		//cv::Mat imgshow=img.clone();
		//for(int j=0;j!=CONTOUR_POINTS;j++){
		//	cv::circle(imgshow,affinedPoints[j],1,cv::Scalar(0,0,0));
		//}
		//cv::imshow("debug window 1",imgshow);
		//cv::waitKey(0);
	}
	
	int flag=0;
	for(int i=0;i!=particleNum;i++){
		if(particleConfidence[i]<0.3){//���Ŷȹ�С�Ͱ����Ŷ���0
			particleConfidence[i]=0;
			flag++;
		}
	}
	cv::normalize(particleConfidence,particleConfidence,1.0,0.0,cv::NORM_L1);
	if (flag!=particleNum)//����������ӵ����Ŷȶ�Ϊ0����Ϊ����ʧ��
		return true;
	else 
		return false;
}


///
//���Ե�ǰ���Ӷ�Ӧ���������Ƿ�Ϸ�
//�����б�
//state         ���ӣ�״̬��
//actualPoints  ��ǰ���Ӷ�Ӧ���������������ĵ�
bool isValid(const cv::Vec<double,9>& state, const std::vector<cv::Point>& actualPoints){
	//state[0]��state[3]������������Ļ�׼λ�ã��ֱ��Ӧ��x�����y����
	//��Ȼ���Ӳ��ܳ�����������
	//state[7]��state[8]�ǳ߶ȱ任��С����Ȼ�߶ȱ任��С����Ϊ��ֵ
	if (state[0]>=0 && state[0]<640 && state[3]>=0 && state[3]<480 && state[7]>0 && state[8]>0){
		std::vector<cv::Point>::const_iterator itr=actualPoints.begin();//�������Ƕ�һ��ָ��ľ��ɰ�װ
		//������������ÿһ���㣬�����궼���ܳ�����������
		while(itr!=actualPoints.end() && (*itr).x>=0 && (*itr).x<640 && (*itr).y>=0 && (*itr).y<480)
			itr++;
				if (itr==actualPoints.end())
					return true;
	}
	return false;
}

///
//��������MeasuredFingertip
//�����ࣺPaticle
//���ܣ�����������ӵ�����ֵ��������λ��ָʾָ��λ��
//�����б�
//void
//���ز��� cv:Point ���ݵ�ǰ�������ŶȺ�����״̬���������ָ��λ��
cv::Point Particle::MeasuredFingertip() const {
	double x=0,y=0;
	for(int i=0;i!=particleNum;i++){
		x+=particleConfidence[i]*particleStates[i][0];
		y+=particleConfidence[i]*particleStates[i][3];
	}
	return cv::Point2i((int)x,(int)y);
}
