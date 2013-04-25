#include "opencv2\core\core.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "Particle.h"
#include <iostream>
#include <vector>
#include <stdio.h>
//#include <ctype.h>

using namespace cv;
using namespace std;

//hide the local functions in an anon namespace
namespace
{
	Particle particles(20,(Mat_<double>(9,9)<<
		1.8,-0.6,-0.2,0,0,0,0,0,0,
		1,0,0,0,0,0,0,0,0,
		0,1,0,0,0,0,0,0,0,
		0,0,0,1.8,-0.6,-0.2,0,0,0,
		0,0,0,1,0,0,0,0,0,
		0,0,0,0,1,0,0,0,0,
		0,0,0,0,0,0,1,0,0,
		0,0,0,0,0,0,0,1,0,
		0,0,0,0,0,0,0,0,1));
	Mat curframe;
	Mat bkgnd;
	Mat preframe;
	bool trackObject = false;
	Point origin;
	Mat SE=getStructuringElement(MORPH_ELLIPSE,Size(5,5));

    void help(char** av)
    {
        cout << "\nThis program justs gets you started reading images from video\n"
        "Usage:\n./" << av[0] << " <video device number>\n" << "q,Q,esc -- quit\n"
        << endl;
    }

	//所输入的均为RGB图像。

	Mat framediff(const Mat& preframe, const Mat& curframe){
		vector<Mat> curchannel(3), prechannel(3), finger(3);
			split(curframe,curchannel);
			split(preframe,prechannel);
			for (int i=0;i!=3;i++){
				absdiff(curchannel[i],prechannel[i],finger[i]);
				threshold(finger[i],finger[i],40,255,THRESH_BINARY);
			}
			Mat finger_step1=finger[0]|finger[1]|finger[2];
			return finger_step1;
	}

}

int main(int ac, char** av)
{
    if (ac != 2)
    {
        help(av);
        return 1;
    }
    std::string arg = av[1];
    VideoCapture capture(arg); //try to open string, this will attempt to open it as a video file
    if (!capture.isOpened()){
		//if this fails, try to open as a video camera, through the use of an integer param
		capture.open(atoi(arg.c_str()));
		//capture.set(CV_CAP_PROP_FRAME_WIDTH,320);
		//capture.set(CV_CAP_PROP_FRAME_HEIGHT,240);
        
	}
    if (!capture.isOpened())
    {
        cerr << "Failed to open a video device or video file!\n" << endl;
        help(av);
        return 1;
    }

	string window_name = "Air Writing Debug Version 2";
    cout << "press space to save a picture. q or esc to quit" << endl;
    namedWindow(window_name, CV_WINDOW_AUTOSIZE);
	//调试用
	//namedWindow("debug window 1", CV_WINDOW_AUTOSIZE);
	//namedWindow("debug window 2", CV_WINDOW_AUTOSIZE);
	//namedWindow("debug window 3", CV_WINDOW_AUTOSIZE);

	capture>>curframe;
	bkgnd=curframe.clone();
	preframe=bkgnd.clone();
	Mat img;

    for (;;)
    {
		capture >> curframe;
		if(curframe.empty())
			continue;
		curframe.copyTo(img);

		//imshow("debug window 1",bkgnd);
		//imshow("debug window 2",curframe);
		//imshow("debug window 3",preframe);
		//waitKey(0);
		//帧差开始变小，说明手开始固定；
		//与背景相减即看看是否有手势。
		//这一切标志着可以开始初始化。
		Mat vali=framediff(curframe,preframe);
		Mat handimg=framediff(curframe,bkgnd);
		//imshow("debug window 1",handimg);
		cout<<"sub between fram "<<countNonZero(vali)<<endl<<"sub of bkgnd "<<countNonZero(handimg)<<endl<<endl;
		//waitKey(0);
		if(!trackObject && countNonZero(vali)<800 && countNonZero(handimg)>16000){
			
			trackObject=true;//这个值什么时候才能被改变为false呢？
			//cout<<"close operation"<<endl;
			morphologyEx(handimg,handimg,MORPH_CLOSE,SE);
			//imshow("debug window 1",handimg);
			//waitKey(0);
			particles.InitParticle(handimg);
			//continue;
		}

		if(trackObject){

			particles.PredictParticle();

			particles.MeasureParticle(curframe, trackObject);
			
			//表现粒子，使用方框或什么东西。。。(在img上画图！
			//当然作出轨迹更好啦。
			//表现均衡后的粒子，请用此式
			circle(img,particles.MeasuredFingertip(),8,Scalar(0,0,255),2);
			//表现每一个粒子，请用以下四行
			//vector<Rect> positions=particles.EveryParticle(selection);
			//for (int i=0;i!=positions.size();i++){
			//	rectangle(img,positions[i],Scalar(0,255,255),2);
			//}
		}

		imshow(window_name, img);
		preframe=curframe.clone();

		char key = (char) waitKey(0); 
		switch (key)
		{
			case 'q':
			case 'Q':
			case 27: //escape key
				return 0;
			default:
				break;
		}
	}

    return 0;
}
