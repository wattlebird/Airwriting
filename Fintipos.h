////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2013, Ronnie Wang（王程诚）
// Permission of use under MIT License
//
// Filename     ：Fintipos.h
// Project Code ：
// Abstract     ：The declaration of the function to determine fingertip position
// Reference    ：
//
// Version      ：1.0
// Author       ：Ronnie Wang（王程诚）
// Accomplished date ： 2013.06
//
// Replaced version  :
// Original Author   :
// Accomplished date : 
//
////////////////////////////////////////////////////////////////////////////
#ifndef FINTIPOS_H
#define FINTIPOS_H

#include <vector>
#include "opencv2\core\core.hpp"

int FingertipPos(const std::vector<cv::Point>&);
int FingertipPos(const std::vector<cv::Point>&, const cv::Mat&);

#endif