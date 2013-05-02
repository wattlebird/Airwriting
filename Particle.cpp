#include "Particle.h"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\highgui\highgui.hpp"
#include <algorithm>
#include <cmath>
#include "ContourConfidence.h"
#include "Fintipos.h"

//#define DEBUG_INIT

void Particle::InitParticle(const cv::Mat& handimg){
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	std::vector<cv::Point> handcontour;
#ifdef DEBUG_INIT
	cv::Mat imgshow=handimg.clone();
#endif

	cv::findContours( handimg, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );

	if (hierarchy[0][0]>=0 || hierarchy[0][1]>=0){
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


	int min_index=FingertipPos(handcontour);

	int len=handcontour.size()+1;
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
#endif
	
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


bool contours_compare(const std::vector<cv::Point> obj1,
	const std::vector<cv::Point> obj2){
		return obj1.size()<obj2.size();
}



void Particle::PredictParticle(){
	std::vector<int> nextround(particleNum);
	for (int i=0;i!=particleNum;i++)
		nextround[i]=int(particleNum*particleConfidence[i]);
	int totalsum=cv::sum(nextround)[0];
	if (particleNum-totalsum){
		std::vector<int>::iterator next_itr=std::max_element(nextround.begin(),nextround.end());
		(*next_itr)+=(particleNum-totalsum);//可能出现负值，坑
	}



	std::vector<cv::Vec<double,9> > temp(particleNum);
	int k=0;
	for (int i=0;i!=particleNum;i++)
		if(nextround[i])
			for (int j=0;j!=nextround[i];j++){
				cv::Vec<double,9> randomVec(randobj.gaussian(6),0,0,
					randobj.gaussian(6),0,0,
					randobj.gaussian(0.1),//弧度！
					randobj.gaussian(0.01),
					randobj.gaussian(0.01)
					);
			
				cv::gemm(dynModel,particleStates[i],1.0,randomVec,1.0,temp[k]);
				k++;
				
			}
	particleStates=temp;

}

//输入的img应该是RGB图像。
//……实际上输入什么图像应该由更为上层的结构确定
void Particle::MeasureParticle(const cv::Mat& img, bool& trackObject){
	//这里假设输入RGB图像。
	std::vector<cv::Vec<double, 2*CONTOUR_POINTS> > controlPoints(particleNum);
	//cv::namedWindow("debug window 1", CV_WINDOW_AUTOSIZE);

	for (int i=0;i!=particleNum;i++){
		cv::Mat S=(cv::Mat_<double>(6,1)<<
			particleStates[i][0],
			particleStates[i][3],
			particleStates[i][7]*(std::cos(particleStates[i][6]))-1,
			particleStates[i][8]*(std::cos(particleStates[i][6]))-1,
			-particleStates[i][8]*(std::sin(particleStates[i][6])),
			particleStates[i][7]*(std::sin(particleStates[i][6])));
		cv::gemm(W,S,1,templateControlPoint,1,controlPoints[i]);
		std::vector<cv::Point> affinedPoints(CONTOUR_POINTS);
		for (int j=0;j!=CONTOUR_POINTS;j++){
			affinedPoints[j].x=controlPoints[i][j];
			affinedPoints[j].y=controlPoints[i][j+CONTOUR_POINTS];
		}
		if (isValid(particleStates[i],affinedPoints)){
			particleConfidence[i]=ContConf(affinedPoints,img);
		}else{
			particleConfidence[i]=0;
		}
		std::cout<<particleConfidence[i]<<std::endl;
		//用于调试
		//cv::Mat imgshow=img.clone();
		//for(int j=0;j!=CONTOUR_POINTS;j++){
		//	cv::circle(imgshow,affinedPoints[j],1,cv::Scalar(0,0,0));
		//}
		//cv::imshow("debug window 1",imgshow);
		//cv::waitKey(0);
	}

	trackObject=false;
	int flag=0;
	for(int i=0;i!=particleNum;i++){
		if(particleConfidence[i]<0.3){
			particleConfidence[i]=0;
			flag++;
		}
	}
	if (flag!=particleNum)
		trackObject=true;
	else 
		std::cout<<"failure because of confidence"<<std::endl;

	cv::normalize(particleConfidence,particleConfidence,1.0,0.0,cv::NORM_L1);
			
}


	
bool isValid(const cv::Vec<double,9>& state, const std::vector<cv::Point>& actualPoints){
	if (state[0]>=0 && state[0]<640 && state[1]>=0 && state[1]<480 && state[3]>0 && state[4]>0){
		std::vector<cv::Point>::const_iterator itr=actualPoints.begin();
		while(itr!=actualPoints.end() && (*itr).x>=0 && (*itr).x<640 && (*itr).y>=0 && (*itr).y<480)
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
		y+=particleConfidence[i]*particleStates[i][3];
	}
	return cv::Point2i((int)x,(int)y);
}
