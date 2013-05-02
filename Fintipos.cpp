#include "Fintipos.h"
#include "opencv2\imgproc\imgproc.hpp"
#include <algorithm>
#include <iostream>
#include "opencv2\highgui\highgui.hpp"

#define DEBUG_FIN

bool fine_angle(double angle){
	return (angle>-0.2 && angle<0.05);
}

int FingertipPos(const std::vector<cv::Point>& handcontour){
	int len=handcontour.size();
	std::vector<int> hull_index;
	std::vector<double> angle;
	cv::convexHull(handcontour,hull_index);

	for (int i=0;i!=hull_index.size();i++){
		cv::Point2d pt1=handcontour[(hull_index[i]+15)%len]-handcontour[(hull_index[i])];
		cv::Point2d pt2=handcontour[(hull_index[i]-15+len)%len]-handcontour[(hull_index[i])];
		pt1*=(1/cv::norm(pt1));
		pt2*=(1/cv::norm(pt2));
		angle.push_back(pt1.dot(pt2));
		//std::cout<<angle[i]<<std::endl;
	}
	
	std::vector<double>::const_iterator itr=std::find_if(angle.begin(),angle.end(),fine_angle);

#ifdef DEBUG_FIN
	cv::Mat imgshow=cv::Mat::zeros(480,640,CV_8UC1);
	std::vector<std::vector<cv::Point> > debugcontour;
	debugcontour.push_back(handcontour);
	cv::drawContours(imgshow,debugcontour,0,cv::Scalar(128),2);
	for(int i=0;i!=hull_index.size();i++){
		cv::circle(imgshow,handcontour[hull_index[i]],2,cv::Scalar(192));
	}
	cv::circle(imgshow,handcontour[hull_index[itr-angle.begin()]],5,cv::Scalar(255));
	cv::namedWindow("Debug");
	cv::imshow("Debug",imgshow);
	cv::waitKey(0);
#endif

	return hull_index[itr-angle.begin()];
}