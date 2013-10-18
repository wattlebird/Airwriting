////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2013, Ronnie Wang�����̳ϣ�
// Permission of use under MIT License
//
// Filename     ��ContourConfidence.cpp
// Project Code ��
// Abstract     ��The implementation of the function to determine contour confidence
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
#include "ContourConfidence.h"
#include <cmath>
#include <stdexcept>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>


//�������ʾ���������ڼ������ŶȵĲ�������ȡ����һ��ע��
//#define DEBUG_CONTOUR

double confcal(const cv::Mat&, const cv::Point&, const cv::Point2d&, const double nlen=4.0);
double confcal_debug(cv::Mat&, const cv::Mat&, const cv::Point&, const cv::Point2d&, const double nlen=4.0);

///
//��������ContConf
//�����ࣺ��
//���ܣ�������һϵ�е����������ġ��ڵ�ǰͼ���ϵ����Ŷ�
//�����б�
//corse_contour   ��������һϵ�е���ɡ������⼸����ֻ�������������ϼ����ȡ�ĵ㣬����˵�ǡ��ֲڵġ�
//img             չʾ����������Ե�ĺڰ�ͼƬ
//������� double ������Ӧ�����ӵ����Ŷ�
//����������confcal
//		  confcal_debug
double ContConf(const std::vector<cv::Point>& corse_contour, const cv::Mat& img){
	int len=corse_contour.size();
	//��δ���������ǣ������ÿ�����������ϵķ��ߵ�λ�������洢��normal��
	//û��ʹ�ñ���������֮���
	//ԭ��ܼ򵥣�һ����ķ����������������������ڵ�нǵĽ�ƽ�����ϡ�
	//���ڳ�ʼ�㣬������ǰ��û�е㣬�䷨�߷����������������ĵ����ߵķ��߷���
	//����ĩβ�㣬�����ٺ���û�е㣬�䷨�߷������������ǰ��ĵ����ߵķ��߷���
	std::vector<cv::Point2d> normal(len);
	std::vector<cv::Point>::const_iterator src_itr=corse_contour.begin();
	std::vector<cv::Point2d>::iterator dest_itr=normal.begin();
	for (int i=0;i!=len;i++){
		cv::Point2d pt1,pt2;//������λ����������������ĵ��������ҵĵ�����ߵĵ�λ����
		if(i==0)//����ǳ�ʼ��
			pt1=cv::Point2d(0,0);
		else{
			pt1=src_itr[i]-src_itr[i-1];
			pt1*=(1/cv::norm(pt1));
		}
		if(i==len-1)//�����ĩβ��
			pt2=cv::Point2d(0,0);
		else{
			pt2=src_itr[i+1]-src_itr[i];
			pt2*=(1/cv::norm(pt2));
		}
		dest_itr[i]=pt1-pt2;//������λ����������Ƿ��߷���Ĵ�ֱ����Ŷ�ҷ���������Լ򻯡���
		if((dest_itr[i].x)||(dest_itr[i].y))
			dest_itr[i]*=(1/cv::norm(dest_itr[i]));
		else//�����������Ϊ������
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
	dis/=(2*len*1.2);//���Ŷȳ���һ�����������������Matlab�����ʵ�����������
	return std::exp(-dis);
}

///
//��������confcal
//�����ࣺ��
//���ܣ������ڸ���������ͼƬ�У�һ�����ڷ��߾�����ƫ����������
//�����б�
//img   һ���ڰ�����ͼƬ
//pt    һ�������ϵĵ�
//n     ��������
//nlen  ������㿪ʼ���ڷ��߾�����������Զ
double confcal(const cv::Mat& img, const cv::Point& pt, const cv::Point2d& n, const double nlen){
	cv::Point unitnormal;
	unitnormal.x=int(n.x*nlen+0.5);
	unitnormal.y=int(n.y*nlen+0.5);
	cv::Point pt1=pt+unitnormal;//������㿪ʼ���ڷ��ߵ�һ�������������ľ�ͷ
	cv::Point pt2=pt-unitnormal;//������㿪ʼ���ڷ��ߵ���һ���������������յ�
	double width=img.cols;
	double height=img.rows;
	if (pt1.x<0 || pt1.x>=width || pt1.y<0 || pt1.y>=height || pt2.x<0 || pt2.x>=width || pt2.y<0 || pt2.y>=height)
		throw std::invalid_argument("points are out of pic's range");

	cv::LineIterator normal_itr1(img,pt,pt1);
	cv::LineIterator normal_itr2(img,pt,pt2);
	int i,j;
	for (i=0;i!=normal_itr1.count;i++,normal_itr1++)
		if ((**normal_itr1)==255)//����������Ϊ255�ĵ��ʱ�򣬾��ǵ������������ı�Ե
			break;

	for (j=0;j!=normal_itr2.count;j++,normal_itr2++)
		if ((**normal_itr2)==255)
			break;
	
	double dis1=cv::norm(normal_itr1.pos()-pt);
	double dis2=cv::norm(normal_itr2.pos()-pt);

	return dis1<dis2 ? dis1 : dis2;

}

///
//��������confcal
//�����ࣺ��
//���ܣ������ڸ���������ͼƬ�У�һ�����ڷ��߾�����ƫ����������
//     ��confcal������ͬ��������������˿�����ʾ���ߵ�ͼ�����ڵ���
//�����б�
//imgshow   ���з��ߺ�������ͼƬ���������
//img       һ���ڰ�����ͼƬ
//pt        һ�������ϵĵ�
//n         ��������
//nlen      ������㿪ʼ���ڷ��߾�����������Զ
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