#include <vector>
#include <cmath>
#include <stdexcept>
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>

double ContConf(const std::vector<cv::Point>&, const cv::Mat&);
double ContConf(const std::vector<cv::Point>& , const std::vector<int>&, const cv::Mat&);
