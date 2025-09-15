#pragma once

#ifndef ASSOCIATE_H
#define ASSOCIATE_H

#include "../stdafx.h"
#include "../struct.h"
#include "../global_parameters.h"
//for kalmanfilter and association.
#include "../utils/kalmanfilter.h"
#include "../utils/extrapolation.h"
#include "../utils/hungarian.h"
#include "../utils/matching.h"
//for initializing trackers.
#include "../tracker/mosse.h"
#include "../tracker/template_matching.h"

class Associate
{
private:

    //hungarian algorithm
    HungarianAlgorithm HungAlgo;

	//For tracker setting.
	bool _bool_skip = true; //skip updating for occlusion and switching prevention
	double _K_SIGMA = 2.0;//threshold for skipping condition. MU-_K_SIGMA*std region.
	double _N_WARMUP = 10;
	int _MAX_SKIP = 10;
	double _scoreThreshold_template_matching = 0.7;
	double _psrThreshold_mosse = 5.0;
	//for association setting.
    bool _bool_TBD = false; //done-> false //if true : update position too
	unsigned int _mode_duplication = 3;
	//_mode_duplication.
	//0: no duplication, 1: duplication with IoU, 2: duplication with IoU and velocity.
	//3: duplication with IoU and velocity, and augmentation in merging trackers.
    bool bool_check_psr = true; //done->true //which tracker adopt : detection or tracking
    bool bool_comparePSR = false; //compare by PSR 

public:
	//for merging trackers.
	TrackersMOT trackersMOT_association;//for association with YOLO.

    Associate()
	{
		std::cout << "construct Associate" << std::endl;
		_psrThreshold_mosse = GP::psrThreshold_mosse;
		_scoreThreshold_template_matching = GP::score_threshold_template_matching;
		_K_SIGMA = GP::K_SIGMA;
		_N_WARMUP = GP::N_WARMUP;
		_MAX_SKIP = GP::MAX_SKIP;
		_bool_skip = GP::bool_skip;
		_bool_TBD = GP::bool_TBD;
		_mode_duplication = GP::mode_duplication;
	};
    ~Associate();

	/**
    * @brief match trackers and update Kalman filter.
    */
   static void organize(
	Trackers2MOT& trackers2mot,
	TrackersYOLO& trackersYOLO,
	TrackersMOT& trackersMOT
	);

	/**
	* @brief merge trackers2mot to trackers_sequence, and create trackersMOT.
	*/
	static void mergeTracking_MOT(
		Trackers2MOT& trackers2mot,
		TrackersMOT& trackersMOT
	);
	/**
	* @brief match trackers.
	*/
	staticvoid matching(
		std::vector<cv::Rect2d>& newRoi, std::vector<int>& newLabel, double& frameIndex,
		std::vector<std::vector<std::vector<double>>>& seqData, std::vector<std::vector<std::vector<double>>>& kfData,
		std::vector<KalmanFilter2D>& kalmanVector, std::vector<LinearExtrapolation2D>& extrapolation
	);

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
    

};


#endif