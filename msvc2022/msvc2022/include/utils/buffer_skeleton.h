#pragma once

#ifndef BUFFER_SKELETON_H
#define BUFFER_SKELETON_H

#include "stdafx.h"
#include "global_parameters.h"
#include "hungarian.h"
#include "kalmanfilter_skeleton.h"

class Buffer_skeleton
{
private:
    const double time_predict_max_ = 1.0;//max prediction time window : [sec]
    int counter_yolo_left = 0;
    int counter_yolo_right = 0;
    std::vector<std::vector<double>> human_current_left, human_current_right;
    std::vector<KalmanFilter2D_skeleton> kf_left, kf_right;
    std::vector<std::vector<double>> human_previous_left, human_previous_right;
    std::vector<int> idx_match_left, idx_match_right;
    const double dist_thresh_ = 100.0;
    const double area_thresh_ = 3.0;
    const double lambda_dist_ = 100.0;
    const double lambda_area_ = 10.0;
    const double Cost_max_ = 1e4;
    const bool bool_debug = false;
    HungarianAlgorithm HungAlgo;

    const int originalWidth = 512;
    const int originalHeight = 512;
    int frameWidth = 1024;
    int frameHeight = 512;
    //for real-time
    const int yoloWidth = 1024;//640
    const int yoloHeight = 512;//320
    const int boundary_img = 512;//320

    //for ground truth
    //const int yoloWidth = 1280;
    //const int yoloHeight = 640;
    //const int boundary_img = 640;
    const cv::Size YOLOSize{ yoloWidth, yoloHeight };
    const float IoUThreshold = 0.1;
    const float ConfThreshold_human = 0.50;
    const float ConfThreshold_joint = 0.30;//definitely adopt
    const float IoUThresholdIdentity = 0.33; // for maitainig consistency of tracking
    const int num_joints = 6; //number of tracked joints
    const float roi_direction_threshold = 1.5; //max gradient of neighborhood joints
    std::vector<float> default_neighbor{ (float)(std::pow(2,0.5) / 2),(float)(std::pow(2,0.5) / 2) }; //45 degree direction
    const int MIN_SEARCH = 10; //minimum search size
    const float min_ratio = 0.65;//minimum ratio for the max value

    const std::vector<std::vector<int>> lostHuman_center_ = std::vector<std::vector<int>>(num_joints, std::vector<int>(5, -1));//default lost data.{frame,left,top,width,height}
    const std::vector<cv::Rect2i> lostHuman_search_ = std::vector<cv::Rect2i>(num_joints, cv::Rect2i(-1, -1, -1, -1));//default lost data.
    const std::vector<double> lostSearch_ = std::vector<double>(4, -1.0);

    std::vector<int> counter_notUpdate_left_, counter_notUpdate_right_;//counter for not updating.
    std::vector<int> index_delete_send_left_, index_delete_send_right_;
    std::vector<int> index_candidates_left_, index_candidates_right_;

public:
    const bool bool_oneHuman = false;//single person tracking or multiple people tracking.
    std::vector<std::vector<std::vector<std::vector<int>>>> seqHuman_left, seqHuman_right, saveHuman_left, saveHuman_right;//{#(human),#(seq),#(joints),(frame,left,top,width,height)}

    Buffer_skeleton() {
        std::cout << "Construct Buffer class" << std::endl;
    };

    void main();

    void nonMaxSuppressionHuman(torch::Tensor& x, std::vector<torch::Tensor>& detectedBoxesHuman, float confThreshold, float iouThreshold);

    torch::Tensor xywh2xyxy(torch::Tensor x);

    void nms(torch::Tensor& x, std::vector<torch::Tensor>& detectedBoxes, float& iouThreshold, bool& boolLeft, bool& boolRight);

    float calculateIoU(const torch::Tensor& box1, const torch::Tensor& box2);

    void keyPointsExtractor(std::vector<torch::Tensor>& detectedBboxesHuman, std::vector<std::vector<std::vector<int>>>& keyPoints, std::vector<int>& humanPos, int& frameIndex, const int& ConfThreshold);

    void matching(std::vector<std::vector<double>>& candidates, std::vector<std::vector<double>>& newData, std::vector<int>& idx_match,
        std::vector<int>& counter_notUpdate, std::vector<int>& index_delete_send, int& frameIndex, bool bool_left);

    void drawCircle(cv::Mat1b& frame, std::vector<std::vector<std::vector<int>>>& ROI, int& counter);

    void push2Queue(cv::Mat1b& frame, int& frameIndex, std::vector<std::vector<std::vector<int>>>& keyPoints,
        std::vector<cv::Rect2i>& roiLatest, std::vector<int>& humanPos,
        std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver_left, std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver_right);

    void organizeRoi(cv::Mat1b& frame, int& frameIndex, bool& bool_left, std::vector<std::vector<int>>& pos, std::vector<std::vector<float>>& distances,
        std::vector<cv::Rect2i>& joints, std::vector<cv::Mat1b>& imgJoint, std::vector<std::vector<int>>& jointsCenter);

    void setRoi(int& frameIndex, cv::Mat1b& frame, bool& bool_left, std::vector<std::vector<float>>& distances,
        int& index_joint, std::vector<int>& compareJoints, std::vector<std::vector<int>>& pos,
        std::vector<cv::Rect2i>& joints, std::vector<cv::Mat1b>& imgJoint, std::vector<std::vector<int>>& jointsCenter);

    void defineRoi_left(int& frameIndex, cv::Mat1b& frame, int& index_joint, float& vx, float& vy,
        float& half_diagonal, std::vector<std::vector<int>>& pos,
        std::vector<cv::Rect2i>& joints, std::vector<cv::Mat1b>& imgJoint, std::vector<std::vector<int>>& jointsCenter);

    void defineRoi_right(int& frameIndex, cv::Mat1b& frame, int& index_joint, float& vx, float& vy, float& half_diagonal,
        std::vector<std::vector<int>>& pos,
        std::vector<cv::Rect2i>& joints, std::vector<cv::Mat1b>& imgJoint, std::vector<std::vector<int>>& jointsCenter);

    void organize_left(cv::Mat1b& frame, int& frameIndex, std::vector<int>& pos, std::vector<cv::Rect2i>& joints, std::vector<cv::Mat1b>& imgJoint, std::vector<std::vector<int>>& jointsCenter);

    void organize_right(cv::Mat1b& frame, int& frameIndex, std::vector<int>& pos, std::vector<cv::Rect2i>& joints, std::vector<cv::Mat1b>& imgJoint, std::vector<std::vector<int>>& jointsCenter);
};

#endif