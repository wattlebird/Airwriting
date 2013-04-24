#include "Particle.h"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\highgui\highgui.hpp"
#include <algorithm>
#include <cmath>
#include "ContourConfidence.h"

void Particle::InitParticle(const cv::Mat& handimg){
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	std::vector<cv::Point> handcontour;
	//cv::Mat imgshow=handimg.clone();

	cv::findContours( handimg, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );

	if (hierarchy[0][0]>=0 || hierarchy[0][1]>=0){
		//如果发现有不属于手的轮廓的噪音怎么办？
		//当然在前期要以大概率去除这部分噪音，否则留在这一步会很耗效率！！！
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
	//以下用于调试，显示轮廓。
	//std::vector<std::vector<cv::Point> > debugcontour;
	//debugcontour.push_back(handcontour);
	//cv::drawContours(imgshow,debugcontour,0,cv::Scalar(128),2);
	//cv::floodFill(handimg,cv::Mat(),handcontour[0],255);//不把可能的噪声涂掉吗？

	//std::vector<int> convexhullpoints_index;
	//cv::convexHull(handcontour,convexhullpoints_index);
	//int min_index=convexhullpoints_index[0];
	//int min_y=handcontour[min_index].y;
	//for (int i=1;i!=convexhullpoints_index.size();i++){
	//	min_y=std::min(min_y,handcontour[convexhullpoints_index[i]].y);
	//	if (min_y==handcontour[convexhullpoints_index[i]].y)
	//		min_index=i;
	//}
	std::vector<cv::Point>::iterator itr=std::min_element(handcontour.begin(),handcontour.end(),fingertip_compare);
	int min_index=itr-handcontour.begin();

	int len=handcontour.size()+1;
	for (int i=-3;i!=4;i++){
		templatePointSetx[i+3]=handcontour[(min_index+5*i+2*len)%len].x-handcontour[(min_index)%len].x;
		templatePointSety[i+3]=handcontour[(min_index+5*i+2*len)%len].y-handcontour[(min_index)%len].y;
		templateControlPoint[i+3]=templatePointSetx[i+3];
		templateControlPoint[i+10]=templatePointSety[i+3];
		//用于调试
		//cv::circle(imgshow,handcontour[(min_index+10*i+2*len)%len],8,cv::Scalar(192),2);
	}
	//cv::namedWindow("debug window 1", CV_WINDOW_AUTOSIZE);
	//cv::imshow("debug window 1",imgshow);
	// cv::waitKey(0);
	
	if(!W.empty())
		W.pop_back(14);
	for (int i=0;i!=14;i++){
		if(i<7){
			double temp[]={1,0,templatePointSetx[i%7],0,0,templatePointSety[i%7]};
			W.push_back(cv::Mat(1,6,CV_64F,temp));
		}
		else{
			double temp[]={0,1,0,templatePointSety[i%7],templatePointSetx[i%7],0};
			W.push_back(cv::Mat(1,6,CV_64F,temp));
		}
	}
	//std::cout<<"W="<<W<<std::endl<<std::endl;
	

	cv::Vec<double,5> init_state(handcontour[min_index].x,
		handcontour[min_index].y,
		0,
		1,
		1);
	particleStates=std::vector<cv::Vec<double,5> >(particleNum,init_state);
}


bool contours_compare(const std::vector<cv::Point> obj1,
	const std::vector<cv::Point> obj2){
		return obj1.size()<obj2.size();
}

bool fingertip_compare(const cv::Point pt1,
	const cv::Point pt2){
		return pt1.y<pt2.y;
}

void Particle::PredictParticle(){
	std::vector<int> nextround(particleNum);
	for (int i=0;i!=particleNum;i++)
		nextround[i]=int(particleNum*particleConfidence[i]);

	for (int i=0;i!=particleNum;i++)
		if(nextround[i])
			for (int j=0;j!=nextround[i];j++){
				cv::Vec<double,5> randomVec(randobj.gaussian(3),
					randobj.gaussian(3),
					randobj.gaussian(0.03),//弧度！
					randobj.gaussian(0.01),
					randobj.gaussian(0.01)
					);
			
				cv::gemm(dynModel,particleStates[i],1.0,randomVec,1.0,particleStates[i]);
				
			}

}

//输入的img应该是RGB图像。
//……实际上输入什么图像应该由更为上层的结构确定
void Particle::MeasureParticle(const cv::Mat& img, bool& trackObject){
	//这里假设输入RGB图像。
	std::vector<cv::Vec<double, 14> > controlPoints(particleNum);
	//cv::namedWindow("debug window 1", CV_WINDOW_AUTOSIZE);

	for (int i=0;i!=particleNum;i++){
		cv::Mat S=(cv::Mat_<double>(6,1)<<
			particleStates[i][0],
			particleStates[i][1],
			particleStates[i][3]*(std::cos(particleStates[i][2]))-1,
			particleStates[i][4]*(std::cos(particleStates[i][2]))-1,
			-particleStates[i][4]*(std::sin(particleStates[i][2])),
			particleStates[i][3]*(std::sin(particleStates[i][2])));
		cv::gemm(W,S,1,templateControlPoint,1,controlPoints[i]);
		std::vector<cv::Point> affinedPoints(7);
		for (int j=0;j!=7;j++){
			affinedPoints[j].x=controlPoints[i][j];
			affinedPoints[j].y=controlPoints[i][j+7];
		}
		particleConfidence[i]=ContConf(affinedPoints,img);
	}

	trackObject=false;
	for (std::vector<double>::const_iterator itr=particleConfidence.begin();
		itr!=particleConfidence.end();itr++){
			if((*itr)>0.0001){
				trackObject=true;
				break;
			}
	}

	cv::normalize(particleConfidence,particleConfidence,1.0,0.0,cv::NORM_L1);
			
}

bool isValid(const cv::Vec<double,5>& state, const std::vector<cv::Point2i>& actualPoints){
	if (state[0]>=0 && state[0]<320 && state[1]>=0 && state[1]<240 && state[3]>0 && state[4]>0){
		std::vector<cv::Point2i>::const_iterator itr=actualPoints.begin();
		while(itr!=actualPoints.end() && (*itr).x>=0 && (*itr).x<320 && (*itr).y>=0 && (*itr).y<240)
			itr++;
		if (itr==actualPoints.end())
			return true;
	}
	return false;
}

		

cv::Point Particle::MeasuredFingertip() const {
	double x=0,y=0;
	for(int i=0;i!=particleNum;i++){
		x+=particleConfidence[i]*particleStates[i][0];
		y+=particleConfidence[i]*particleStates[i][1];
	}
	return cv::Point2i((int)x,(int)y);
}
