#include "Particle.h"
#include "opencv2\imgproc\imgproc.hpp"
#include <algorithm>
#include <cmath>

void Particle::InitParticle(const cv::Mat& handimg){
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	std::vector<cv::Point> handcontour;

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
		templatePointSetx[i+3]=handcontour[(min_index+30*i+2*len)%len].x-handcontour[(min_index)%len].x;
		templatePointSety[i+3]=handcontour[(min_index+30*i+2*len)%len].y-handcontour[(min_index)%len].y;
	}
	
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
				cv::Vec<double,5> randomVec(randobj.gaussian(1),
					randobj.gaussian(1),
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
	cv::Mat gray_img;
	cv::cvtColor(img,gray_img,CV_BGR2GRAY);

	//每个状态对应七个控制点(controlPoints)，每个控制点转化为五个顶点四段曲线；
	//一共有particleNum个状态，每个状态要维护五个顶点的位置(actualPoints)，以及x、y方向上的导数(deravationX)；
	//而每个顶点要维护沿着其法线方向上的20个像素点，以计算法线上的梯度。
	std::vector<cv::Vec<double, 14> > controlPoints(particleNum);
	std::vector<std::vector<cv::Point2i> > actualPoints;
	std::vector<cv::Vec<double, 5> > deravationX(particleNum);
	std::vector<cv::Vec<double, 5> > deravationY(particleNum);

	std::vector<cv::Point2i> temprory(5);
	actualPoints=std::vector<std::vector<cv::Point2i> >(particleNum,temprory);
	//一些计算参数
	cv::Mat bspline=(cv::Mat_<double>(4,4)<<-1,3,-3,1,3,-6,3,0,-3,0,3,0,1,4,1,0);
	for (int i=0;i!=particleNum;i++){
		cv::Mat S=(cv::Mat_<double>(6,1)<<
			particleStates[i][0],
			particleStates[i][1],
			particleStates[i][3]*(std::cos(particleStates[i][2]))-1,
			particleStates[i][4]*(std::cos(particleStates[i][2]))-1,
			-particleStates[i][4]*(std::sin(particleStates[i][2])),
			particleStates[i][3]*(std::sin(particleStates[i][2])));
		cv::gemm(W,S,1,cv::Mat(),0,controlPoints[i]);
		for (int j=0;j!=4;j++){
			cv::Vec4d tempM1,tempM2,baset,dbaset;
			cv::Mat controlPointX,controlPointY;
			controlPointX=(cv::Mat_<double>(4,1)<<controlPoints[i][j],
				controlPoints[i][j+1],
				controlPoints[i][j+2],
				controlPoints[i][j+3]);
			controlPointY=(cv::Mat_<double>(4,1)<<controlPoints[i][j+7],
				controlPoints[i][j+8],
				controlPoints[i][j+9],
				controlPoints[i][j+10]);
			cv::gemm(bspline,controlPointX,1.0/6.0,cv::Mat(),0,tempM1);
			cv::gemm(bspline,controlPointY,1.0/6.0,cv::Mat(),0,tempM2);
			baset=cv::Vec4d(0,0,0,1);
			dbaset=cv::Vec4d(0,0,1,0);
			actualPoints[i][j].x=(int)baset.dot(tempM1);
			actualPoints[i][j].y=(int)baset.dot(tempM2);
			deravationX[i][j]=dbaset.dot(tempM1);
			deravationY[i][j]=dbaset.dot(tempM2);
			if(j==3){
				baset=cv::Vec4d(1,1,1,1);
				dbaset=cv::Vec4d(3,2,1,0);
				actualPoints[i][j+1].x=(int)baset.dot(tempM1);
				actualPoints[i][j+1].y=(int)baset.dot(tempM2);
				deravationX[i][j+1]=dbaset.dot(tempM1);
				deravationY[i][j+1]=dbaset.dot(tempM2);
			}
		}
		if (!isValid(particleStates[i],actualPoints[i]))
			particleConfidence[i]=0;
	}
		
	//这一段代码的目的，就是要算出每个状态的顶点的置信度度量。
	for (int i=0;i!=particleNum;i++){
		if (particleConfidence[i]==0){
			continue;
		}
		else{
			std::vector<double> gradient(5);
			std::vector<cv::Point2i> normalPoints(20);
			//std::vector<std::vector<cv::Point2i> > normal(5);
			for (int j=0;j!=5;j++){
				gradient[j]=-deravationX[i][j]/deravationY[i][j];
				for (int k=-10;k!=10;k++){
					normalPoints[k+10].x=k;
					normalPoints[k+10].y=(int)k*gradient[j];
					normalPoints[k+10]+=actualPoints[i][j];
				}
				//在这里，一个状态的一个顶点的法线已经维护完毕。当然我要提醒诸位的是，可能会有某些点为负值的情况。这时候应该怎么办呢？
				//又名：坑
				std::vector<int> d(19);
				for (int k=0;k!=19;k++){
					try{
						d[k]=(int)gray_img.at<uchar>(normalPoints[k])-(int)gray_img.at<uchar>(normalPoints[k+1]);
						if (d[k]<0) d[k]=-d[k];
					}catch(cv::Exception me){
						if (me.err=="dims <= 2 && data && (unsigned)pt.y < (unsigned)size.p[0] && (unsigned)(pt.x*DataType<_Tp>::channels) < (unsigned)(size.p[1]*channels()) && ((((sizeof(size_t)<<28)|0x8442211) >> ((DataType<_Tp>::depth) & ((1 << 3) - 1))*4) & 15) == elemSize1()")
							d[k]=255;
					}
				}
				std::vector<int>::iterator itr_d=std::max_element(d.begin(),d.end());
				gradient[j]+=((itr_d-d.begin())-9)*((itr_d-d.begin())-9);

			}
			particleConfidence[i]=std::exp(-(cv::sum(gradient))[0]/(2*20));//是否是20，还有待查证。
		}
	}

	trackObject=false;
	for (std::vector<double>::const_iterator itr=particleConfidence.begin();
		itr!=particleConfidence.end();itr++){
			if((*itr)>0.01){
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
