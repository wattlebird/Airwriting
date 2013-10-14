#ifndef FINTIPOS_H
#define FINTIPOS_H

#include <vector>
#include "opencv2\core\core.hpp"

int FingertipPos(const std::vector<cv::Point>&);
int FingertipPos(const std::vector<cv::Point>&, const cv::Mat&);

#endif