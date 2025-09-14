#pragma once

#ifndef TRACKER_H
#define TRACKER_H

#include "stdafx.h"
#include "global_parameters.h"
#include "utils/utility.h"
#include "tracker/mosse.h"
#include "tracker/template_matching.h"
#include "struct.h"

class Tracking
{
private:
    // template matching constant value setting
    double _scale_max, _scale_min;
	std::string _matchingMethod;
	int _min_search_area = 20;
	double _max_move = 10.0;
    unsigned int _mode_tracking = 0; 
	//0: mosse, 1: template matching, 2: mosse+template matching w/ single thread, 3: mosse+template matching w/ multiple threads
   
    std::vector<int> defaultMove{ 0, 0 };
 
public:

    float t_elapsed = 0;
    //constructor
    Tracking(const double& scale_max=3.0, const double& scale_min=1.5, const int& min_search_area=20, const double& max_move=10.0, const unsigned int& mode_tracking=3, const std::string& matchingMethod="sqdiff")
	: _scale_max(scale_max), _scale_min(scale_min), _min_search_area(min_search_area), _max_move(max_move), _mode_tracking(mode_tracking), _matchingMethod(matchingMethod)
    {
        std::cout << "construtor of tracking" << std::endl;
    };

    ~Tracking() {};

    void main(std::queue<bool>& q_startTracker);
    
    /* Template Matching  */
    std::vector<bool> track(cv::Mat1b& img, const int& frameIndex,Trackers& trackers);

	bool track_template_matching(cv::Mat1b img, cv::Ptr<TemplateMatching> template_matching, cv::Rect2d bbox);
	bool track_mosse(cv::Mat1b img, cv::Ptr<cv::mytracker::TrackerMOSSE> tracker, cv::Rect2d bbox);
};

#endif