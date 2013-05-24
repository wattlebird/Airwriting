#include "Fintipos.h"
#include "opencv2\imgproc\imgproc.hpp"
#include <algorithm>
#include <iostream>
#include "opencv2\highgui\highgui.hpp"
#include <stdexcept>
#include <fstream>

//#define DEBUG_FIN
//#define CSV_OUTPUT

bool pangle(const double& angle){
	return (angle>-0.2);
}

bool nangle(const double& angle){
	return !pangle(angle);
}

bool fine_angle_set_compare(const std::vector<double>& a, const std::vector<double>& b){
	return a.size()<b.size();
}

int FingertipPos(const std::vector<cv::Point>& handcontour){
	int len=handcontour.size();
	//std::vector<int> hull_index;
	std::vector<double> angle;
	//cv::convexHull(handcontour,hull_index);

#ifdef CSV_OUTPUT
	std::ofstream outf("out.csv");
	outf<<"index,angle,x,y"<<std::endl;
#endif

	for (int i=0;i!=len;i++){
		if(handcontour[i].y<450){
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