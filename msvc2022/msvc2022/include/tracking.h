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
    //tracker constraints
    const double gamma = 0.5; //ration of current and past for tracker velocity, the larger, more important current :: gamma*current+(1-gamma)*past
    const int MAX_VEL = 40; // max roi move 
    const int delta_move = 5.0; //moving distance
    //search area setting
    const double MIN_SEARCH = 10;
    const bool bool_dynamicScale = true; //done -> true :: scaleXTM ::smaller is good //dynamic scale according to current motion
    //check tracker status
    const bool bool_kf = true; //done->true
    const bool bool_skip = true; //skip updating for occlusion and switching prevention
    //exchange between YOLO and current
    const bool bool_TBD = false; //done-> false //if true : update position too
    const bool bool_check_psr = true; //done->true //which tracker adopt : detection or tracking
    const bool bool_comparePSR = false; //compare by PSR 
    const double min_keep_psr = 7.0;
    //delete duplicated trackers
    bool bool_iouCheck = true; //done -> true;check current tracker iou -> if overlapped
    const float IoU_overlapped = 0.6; //done->0.6~75
    const bool bool_augment = false; //whether augment ROI when detecting duplicated ROI
    const bool bool_checkVel = false; //whether check velocity when dealing with duplicated ROI
    const float thresh_cos_dup = 0.0; // threshold angle for judging another objects
 
public:
    // vector for saving position
    //left
    std::vector<std::vector<cv::Rect2d>> posSaverTMLeft;
    std::vector<std::vector<int>> classSaverTMLeft;

    //right
    std::vector<std::vector<cv::Rect2d>> posSaverTMRight;
    std::vector<std::vector<int>> classSaverTMRight;
    //detected frame
    //left
    std::vector<int> detectedFrameLeft;
    std::vector<int> detectedFrameClassLeft;
    //right
    std::vector<int> detectedFrameRight;
    std::vector<int> detectedFrameClassRight;

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
    
    //get data from Tracker2tracker
    void getData(cv::Mat1b& frame, bool& boolTrackerTM, std::vector<int>& classIndexTM, std::vector<cv::Rect2d>& bboxesTM, std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& trackers_mosse,
        std::vector<cv::Mat1b>& templatesTM, std::vector<bool>& boolScalesTM, std::vector<std::vector<int>>& previousMove,
        int& numTrackersTM, cv::Mat1b& previousImg, std::vector<int>& num_notMove, std::queue<Tracker2tracker>& q_tracker2tracker, std::queue<std::vector<std::vector<double>>>& q_seq2tracker);
    
    //organize data among Tracker2tracker, Yolo2tracker and Seq2tracker
    void organizeData(cv::Mat1b& frame, std::vector<int>& classIndexTM, std::vector<cv::Rect2d>& bboxesTM,
        std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& trackers_mosse, std::vector<cv::Mat1b>& templatesTM,
        std::vector<bool>& boolScalesTM, std::vector<std::vector<int>>& previousMove, cv::Mat1b& previousImg,
        bool& boolTrackerYolo,
        std::vector<int>& updatedClasses, std::vector<cv::Rect2d>& updatedBboxes,
        std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& updatedTrackers, std::vector<cv::Mat1b>& updatedTemplates,
        std::vector<std::vector<int>>& updatedMove, int& numTrackersTM, std::vector<int>& num_notMove, std::vector<int>& updated_num_notMove,
        std::queue<Yolo2tracker>& q_yolo2tracker);

    //get Yolo data
    void getYoloData(std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& newTrackers, std::vector<cv::Mat1b>& newTemplates, std::vector<cv::Rect2d>& newBboxes, std::vector<int>& newClassIndexes,
        std::queue<Yolo2tracker>& q_yolo2tracker);

    //calculate IoU
    float calculateIoU_Rect2d(const cv::Rect2d& box1, const cv::Rect2d& box2);
    
    //arrange data between Yolo and Tracker
    void combineYoloTMData(cv::Mat1b& frame, std::vector<int>& classIndexesYolo, std::vector<int>& classIndexTM,
        std::vector<cv::Rect2d>& bboxesYolo, std::vector<cv::Rect2d>& bboxesTM,
        std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& trackersYolo, std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& trackers_mosse,
        std::vector<cv::Mat1b>& templatesYolo, std::vector<cv::Mat1b>& templatesTM,
        std::vector<std::vector<int>>& previousMove, cv::Mat1b& previousImg,
        std::vector<int>& updatedClasses, std::vector<cv::Rect2d>& updatedBboxes,
        std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& updatedTrackers, std::vector<cv::Mat1b>& updatedTemplates,
        std::vector<bool>& boolScalesTM, std::vector<std::vector<int>>& updatedMove, const int& numTrackersTM, std::vector<int>& num_notMove, std::vector<int>& updated_num_notMove);

    //compare current tracker with Yolo's by psr
    double check_tracker(cv::Mat1b& previousImg, cv::Rect2d& roi, cv::Ptr<cv::mytracker::TrackerMOSSE>& tracker);
    
    //main tracking process
    void process(std::vector<int>& updatedClasses, std::vector<cv::Rect2d>& updatedBboxes,
        std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& updatedTrackers, std::vector<cv::Mat1b>& updatedTemplates,
        std::vector<bool>& boolScalesTM, std::vector<std::vector<int>>& updatedMove,std::vector<int>& updated_num_notMove,
        cv::Mat1b& img,
        std::vector<int>& updatedClassesTM, std::vector<cv::Rect2d>& updatedBboxesTM,
        std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& updatedTrackers_mosse, std::vector<cv::Mat1b>& updatedTemplatesTM,
        std::vector<bool>& updatedSearchScales, std::vector<std::vector<int>>& updatedMoveTM, std::vector<int>& updated_num_notMove_tm);

    //template matching
    void matchingTemplate(const int classIndexTM, int counterTracker, int counter,
        std::vector<cv::Mat1b>& updatedTemplates, std::vector<bool>& boolScalesTM, std::vector<cv::Rect2d>& updatedBboxes, std::vector<std::vector<int>>& updatedMove, std::vector<int>& updated_num_notMove, cv::Mat1b& img,
        std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& updatedTrackers_mosse, std::vector<cv::Mat1b>& updatedTemplatesTM,
        std::vector<cv::Rect2d>& updatedBboxesTM, std::vector<int>& updatedClassesTM, std::vector<bool>& updatedSearchScales, std::vector<std::vector<int>>& updatedMoveTM, std::vector<int>& updated_num_notMove_tm);

    //MOSSE
    void track_mosse(const int classIndexTM, int counterTracker, int counter,
        std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& updatedTrackers, std::vector<cv::Mat1b>& updatedTemplates, std::vector<bool>& boolScalesTM, std::vector<cv::Rect2d>& updatedBboxes,
        std::vector<std::vector<int>>& updatedMove, std::vector<int>& num_notMove, cv::Mat1b& img,
        std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& updatedTrackers_mosse, std::vector<cv::Mat1b>& updatedTemplatesTM,
        std::vector<cv::Rect2d>& updatedBboxesTM, std::vector<int>& updatedClassesTM, std::vector<bool>& updatedSearchScales, std::vector<std::vector<int>>& updatedMoveTM, std::vector<int>& updated_num_notMove_tm);
};

#endif