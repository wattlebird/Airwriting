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

	//������ľ�ΪRGBͼ��

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
	//������
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
		//֡�ʼ��С��˵���ֿ�ʼ�̶���
		//�뱳������������Ƿ������ơ�
		//��һ�б�־�ſ��Կ�ʼ��ʼ����
		Mat vali=framediff(curframe,preframe);
		Mat handimg=framediff(curframe,bkgnd);
		//imshow("debug window 1",handimg);
		cout<<"sub between fram "<<countNonZero(vali)<<endl<<"sub of bkgnd "<<countNonZero(handimg)<<endl<<endl;
		//waitKey(0);
		if(!trackObject && countNonZero(vali)<800 && countNonZero(handimg)>16000){
			
			trackObject=true;//���ֵʲôʱ����ܱ��ı�Ϊfalse�أ�
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
			
			//�������ӣ�ʹ�÷����ʲô����������(��img�ϻ�ͼ��
			//��Ȼ�����켣��������
			//���־��������ӣ����ô�ʽ
			circle(img,particles.MeasuredFingertip(),8,Scalar(0,0,255),2);
			//����ÿһ�����ӣ�������������
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
