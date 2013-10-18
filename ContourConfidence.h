////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2013, Ronnie Wang�����̳ϣ�
// Permission of use under MIT License
//
// Filename     ��ContourConfidence.h
// Project Code ��
// Abstract     ��The declaration of the function to determine contour confidence
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
#ifndef CONTOURCONFIDENCE_H
#define CONTOURCONFIDENCE_H

#include <vector>
#include <opencv2\core\core.hpp>


double ContConf(const std::vector<cv::Point>&, const cv::Mat&);


#endif