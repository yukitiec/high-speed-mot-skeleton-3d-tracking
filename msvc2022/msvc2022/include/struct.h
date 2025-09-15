#pragma once

#ifndef STRUCT_H
#define STRUCT_H

#include "stdafx.h"
#include "utils/RLS.h"
#include "tracker/mosse.h"
#include "tracker/template_matching.h"

//Yolo to Tracker
struct Yolo2MOT {
    std::vector<torch::Tensor> rois; //detected rois.(n,6),(m,6) :: including both left and right objects
    std::vector<int> labels;//detected labels.
    std::vector<double> scores;//detected scores.
    cv::Mat1b frame;
    int frameIndex;
};
//Trackers in YOLO
struct TrackersYOLO {
    int frameIndex;
	cv::Mat1b frame;
    std::vector<int> classIndex;
    std::vector<cv::Rect2d> bbox;
    std::vector<double> scores;
};
//TrackerInfo in tracking.
struct TrackerInfo{
	cv::Ptr<cv::mytracker::TrackerMOSSE> mosse;
	cv::Ptr<TemplateMatching> template_matching;//have a template image internally.
	cv::Rect2d bbox;
	double scale_searcharea; //scale ratio of search area against bbox
    cv::Point2d vel; //previous velocity
	int n_notMove; //number of not move
};
//All trackers.
struct Trackers {
    std::vector<int> classIndex;
	std::vector<int> index_highspeed; //index for high speed tracking
	std::vector<TrackerInfo> trackerInfo;
    cv::Mat1b previousImg; 
};
//Trackers in MOT.
struct Trackers2MOT{
	int frameIndex;
	std::vector<int> success_flags;
	Trackers trackers;
}

//Trackers info in MOT.
struct TrackersMOT {
	int frameIndex;
    std::vector<int> classIndex;
    std::vector<cv::Rect2d> bbox;
	std::vector<KalmanFilter> kalmanFilter;
	std::vector<int> index_highspeed;
};

struct Trackers_sequence{
	std::vector<KalmanFilter> kalmanFilter;
	std::vector<TrackersMOT> trackersMOT;
	std::vector<int> index_highspeed;
	std::vector<int> classIndex;
}


struct InfoTarget {
    double delta_frame;//frame by catching.
    std::vector<double> p_target;//current target position, {x,y,z,rx,ry,rz}
};

struct InfoParams {
    Eigen::Vector2d param_x;//ax*t+bx
    Eigen::Vector2d param_y;//ay*t+by
    Eigen::Vector3d param_z;//az*t*t+bz*t+cz
};

//trajectory prediction
struct rls {
    RLS rlsx;
    RLS rlsy;
    RLS rlsz;

    // Constructor for rls to initialize each RLS member
    rls(const int dimx, const int dimy, const int dimz, const double forget) :
        rlsx(dimx, forget), rlsy(dimy, forget), rlsz(dimz, forget) {}
};

struct Yolo2Buffer_skeleton {
    torch::Tensor preds;
    std::vector<torch::Tensor> detectedBoxesHuman;
    cv::Mat1b frame;
    int frameIndex;
};

struct Yolo2optflow {
    std::vector<std::vector<cv::Rect2i>> roi; //search ROI
    std::vector<std::vector<cv::Mat1b>> img_search; //search background img
    std::vector<int> index_delete;//human to delete
};

struct Optflow2optflow {
    std::vector<std::vector<cv::Rect2i>> roi; //search ROI
    std::vector<std::vector<cv::Mat1b>> img_search; //search img
    std::vector<std::vector<std::vector<float>>> move; //previous target movement
    std::vector<std::vector<cv::Ptr<cv::DISOpticalFlow>>> ptr_dis; //DIS pointer
};

struct Optflow2tri {
    std::vector<std::vector<std::vector<int>>> data;//data. {#(human),#(joint),(frame,left,top,width,height)}
    std::vector<int> index_delete;//indexes to delete
};

struct Info2Gpu {
    torch::Tensor imgTensor;
    cv::Mat1b frame;
};


#endif//structure