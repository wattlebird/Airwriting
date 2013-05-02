#include "ContourConfidence.h"
#include <cmath>
#include <stdexcept>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>

#define ROUND(i,len) ((i)<0 ? ((i)+len) : (i))
//#define DEBUG_CONTOUR

double confcal(const cv::Mat&, const cv::Point&, const cv::Point2d&, const double nlen=4.0);
double confcal_debug(cv::Mat&, const cv::Mat&, const cv::Point&, const cv::Point2d&, const double nlen=4.0);


double ContConf(const std::vector<cv::Point>& corse_contour, const cv::Mat& img){
	int len=corse_contour.size();
	std::vector<cv::Point2d> normal(len);
	std::vector<cv::Point>::const_iterator src_itr=corse_contour.begin();
	std::vector<cv::Point2d>::iterator dest_itr=normal.begin();
	for (int i=0;i!=len;i++){
		cv::Point2d pt1,pt2;
		if(i==0)
			pt1=cv::Point2d(0,0);
		else{
			pt1=src_itr[ROUND(i%len,len)]-src_itr[ROUND((i-1)%len,len)];
			pt1*=(1/cv::norm(pt1));
		}
		if(i==len-1)
			pt2=cv::Point2d(0,0);
		else{
			pt2=src_itr[ROUND((i+1)%len,len)]-src_itr[ROUND(i%len,len)];
			pt2*=(1/cv::norm(pt2));
		}
		dest_itr[i]=pt1-pt2;//这里可以优化一下，因为有重复计算的结果。
		if((dest_itr[i].x)||(dest_itr[i].y))
			dest_itr[i]*=(1/cv::norm(dest_itr[i]));
		else
			dest_itr[i]=cv::Point2d(pt1.y,-pt1.x);
		if(i==0 || i==(len-1))
			dest_itr[i]=cv::Point2d(dest_itr[i].y,-dest_itr[i].x);
	}

	cv::Mat gray_img;
	cv::cvtColor(img,gray_img,CV_BGR2GRAY);
	cv::blur(gray_img,gray_img,cv::Size(3,3));
	cv::Canny(gray_img,gray_img,50,150);
	
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
			dis+=confcal_debug(imgshow,gray_img,corse_contour[i],normal[i]);
#else
			dis+=confcal(gray_img, corse_contour[i], normal[i]);
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
	dis/=(2*len*1.2);
	return std::exp(-dis);
}



double confcal(const cv::Mat& img, const cv::Point& pt, const cv::Point2d& n, const double nlen){
	cv::Point unitnormal;
	unitnormal.x=int(n.x*nlen+0.5);
	unitnormal.y=int(n.y*nlen+0.5);
	cv::Point pt1=pt+unitnormal;
	cv::Point pt2=pt-unitnormal;
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