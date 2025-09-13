#pragma once

#ifndef SEQUENCE_H
#define SEQUENCE_H

#include "stdafx.h"
#include "kalmanfilter.h"
#include "global_parameters.h"
#include "utility.h"
#include "hungarian.h"
#include "triangulation.h"
#include "prediction.h"
#include "extrapolation.h"
#include "RLS.h"
#include "ivpf2.h"

class Sequence
{
private:
    //for organizing YOLO detections
    const int originalWidth = 512;
    const int orginalHeight = 512;
    const int frameWidth = 1024;
    const int frameHeight = 512;
    const int yoloWidth = 1024;
    const int yoloHeight = 512;

    //prediction
    const int idx_compensation = 1;//0:kalman filter, 1:linear extrapolation.
    //trajectory predictioni
    const int method_prediction = 0;//0:ordinary least square method. 1: recursive least squares method.


    std::vector<std::vector<double>> defaultVector{ {0.0} }; //for initializing kalman filter
    //matching
    const double IoUThresholdIdentity = 0.1; // for maitainig consistency of trackingS
    const double Rmse_identity = 100.0; // minimum rmse criteria
    const double threshold_area_ratio = 4.0;//2.0 & 2.0
    const double Cost_max = 1000.0;
    const double Cost_params_max = 0.0;
    const double lambda_rmse_ = 2.0;

    //the minimum number of updating predictions.
    const int counter_update_params_ = 3;

    //hungarian algorithm
    HungarianAlgorithm HungAlgo;
    Utility utSeq; //check data
    Prediction prediction;

    //observation
    Eigen::Vector2d observation;
    //for kalmanfilter prediction result
    Eigen::Vector<double, 6> kf_predict;

    //triangulation
    std::string rootDir;
    Triangulation tri;
    Matching match; //matching algorithm

    IVPF ivpf_seq;

    std::vector<double> x_candidates;//candidates of x position.

    //move backward to catch the ball.
    const bool bool_backward_ = true;//move backward.
    bool bool_back_ = false;//whether the robot should move backward.
    const double dt_back_ = 0.10;//backward for 0.2 sec.
    const double dt_back_move_ = 0.05;
    double frame_current_, frame_target_current_;
    Utility ut_robot;
    std::array<cv::Mat1b, 2> frames;
    int frameIndex;
    bool boolImgs;
    InfoParams param_candidate;
    std::vector<InfoParams> params_candidates;
    /////////////////////////////////////

    UR_custom ur_main;
    MinimumDist minDist_main;
    IVPF ivpf_main;

    const double z_head_ = 2.0;//unit [m]
    const double z_foot_ = 0.0;//unit [m]

    const double r_catch_candidate_ = 0.10;//catching candidate -> 0.1 m around wrists. 
    const double t_upper_ = 1.0;//0.5 sec
    const double t_lower_ = 0.2;//0.2sec


    const double dt_ = 0.002;
    const double acceleration_ = omega_max_ur_;
    std::vector<double> init_joints{ 0.0,0.0, 0.0, 0.0, 0.0, 0.0 };

    Prediction prediction_;
    const double margin_ = 0.04;//margin to the edge of the working space.
    const double n_candidates = 5.0;//number of candidates for the catching.

    const double x_max_ = ivpf_main.x_work_[1];
    const double x_min_ = ivpf_main.x_work_[0];
    const double y_max_ = ivpf_main.y_work_[1];
    const double y_min_ = ivpf_main.y_work_[0];
    const double z_max_ = ivpf_main.z_work_[1];
    const double z_min_ = ivpf_main.z_work_[0];
    const double h_cup_ = 0.11;//the height of the cup: 0.11 [m]

    //target information.
    const bool bool_dynamic_targetAdjustment = true; //dynamically adjust target determination.
    double lambda_dist, lambda_speed, lambda_human, lambda_dframe, lambda_dist_intra, lambda_dframe_intra;
    double frame_target_save = 0.0;
    bool bool_fix_target = false;

    InfoTarget infoTarget_;
    const double lambda_min_dframe_ = 0.0;
    const double lambda_max_dframe_ = 1.0;
    const double lambda_min_dist_ = 1.0;
    const double lambda_max_dist_ = 1.0;
    const double lambda_min_human_ = 0.0;
    const double lambda_max_human_ = 1.0;
    const double lambda_min_dframe_intra_ = 0.0;
    const double lambda_max_dframe_intra_ = 1.0;
    const double lambda_min_dist_intra_ = 1.0;
    const double lambda_max_dist_intra_ = 1.0;
    const double threshold_dframe_min_ = 0.1 * (double)FPS;//0.2 second is a threshold.
    const double threshold_dframe_max_ = 0.8 * (double)FPS;//0.2 second is a threshold.
    const double threshold_dframe_fix_ = 0.1 * (double)FPS;
    const double r_update_min_ = 0.10;
    const double r_update_max_ = 1.0;
    double r_update_ = 5.0;
    double frame_finish_fix_ = 0.0;

    const double speed_max_ = (5.0 / (double)FPS);//0.0033[m/frame] -> speed_max_* 300 [fps] -> 1.0 [m/sec]
    const double lambda_dist_ = 1.0;//distance is most important.
    const double lambda_speed_ = 1.0;
    const double lambda_dframe_ = 1.0;
    const double dist_thresh_ = 1.0;
    const double dist_target2human_thresh_ = 0.4;
    const double dframe_thresh_ = (double)FPS;//more than 0.5 sec.

    const bool bool_notDetermine_untilCatch = false;//not determine which object to catch until human catches the ball.
    const bool bool_use_actual_data = false;//use acutual tracking data when catching objects.
    const int frame_use_tracking_ = (int)(FPS / 10.0);//0.1 sec before catching.


public:
    //storage
    std::vector<std::vector<std::vector<double>>> seqData_left, seqData_right, kfData_left, kfData_right, saveData_left, saveData_right, saveKFData_left, saveKFData_right; //{num of objects, num of sequence, unit vector}. {frame,label,left,top,width,height}
    std::vector<KalmanFilter2D> kalmanVector_left, kalmanVector_right; //kalman filter instances
    std::vector<LinearExtrapolation2D> extrapolation_left, extrapolation_right;
    //storage for new data
    std::vector<cv::Rect2d> newRoi_left, newRoi_right;
    std::vector<int> newLabels_left, newLabels_right;
    double frameIndex_left, frameIndex_right;
    double depth_target = 0.5;//900 mm

    std::vector<std::vector<double>> wrists_human;//{#(human)*#(lw,rw),(frame,x,y,z)}
    std::vector<std::vector<double>> poses_target;//storage for candidates.{#(objects),(x,y,z,nx,ny,nz)}
    std::vector<double> pose_target;//current robot target.{x,y,z,nx,ny,nz}
    std::vector<double> min_dists_human_objects;//{#(objects),minimum distance between target and human wrists.}
    double frame_human;
    std::vector<int> idx_human_catch, idx_robot_catch;//indexes for candidates of human catching. 
    double frame_target_robot;

    Sequence(const std::string& rootDir)
        : tri(rootDir)
    {
        std::cout << "construct Sequence class" << std::endl;
    };

    /**
    * @brief main function. update every time sequence get new data from Yolo.
    */
    void main();

    /**
    * @brief Get current data before YOLO inference started.
    * First : Compare YOLO detection and TM detection
    * Second : if match : return new templates in the same order with TM
    * Third : if not match : adapt as a new templates and add after TM data
    * Fourth : return all class indexes including -1 (not tracked one) for maintainig data consistency
    */
    void roiSetting(
        std::vector<torch::Tensor>& detectedBoxes, std::vector<int>& labels,
        std::vector<cv::Rect2d>& newRoi_left, std::vector<int>& newClass_left,
        std::vector<cv::Rect2d>& newRoi_right, std::vector<int>& newClass_right
    );

    /**
    * @brief push detect results to a que.
    */
    void push2Queue(
        std::vector<cv::Rect2d>& newRoi, std::vector<int>& newClass, int& frameIndex, Yolo2seq& newdata
    );

    /**
    * @brief match trackers and update Kalman filter.
    */
    void organize(
        Yolo2seq& newData, bool bool_left,
        std::vector<std::vector<std::vector<double>>>& seqData, std::vector<std::vector<std::vector<double>>>& kfData,
        std::vector<KalmanFilter2D>& kalmanVector, std::vector<LinearExtrapolation2D>& extrapolation,
        std::vector<std::vector<std::vector<double>>>& saveData, std::vector<std::vector<std::vector<double>>>& saveKFData,
        std::vector<int>& index_delete, std::queue<std::vector<std::vector<std::vector<double>>>>& q_seq2tri
    );

    /**
    * @brief match trackers.
    */
    void matching(
        std::vector<cv::Rect2d>& newRoi, std::vector<int>& newLabel, double& frameIndex,
        std::vector<std::vector<std::vector<double>>>& seqData, std::vector<std::vector<std::vector<double>>>& kfData,
        std::vector<KalmanFilter2D>& kalmanVector, std::vector<LinearExtrapolation2D>& extrapolation
    );

    double decide_target(std::vector<Seq2robot>& params_trajectory, double& frame_current, std::vector<double>& pose_target, InfoParams& param_target, double& frame_target_current, bool& bool_back);

    //IoU
    double calculateIoU_Rect2d(const cv::Rect2d& box1, const cv::Rect2d& box2);

    //RMSE
    double calculateRMSE_Rect2d(const cv::Rect2d& box1, const cv::Rect2d& box2);

    //ID check
    double compareID(int label1, int label2);

    //check size difference.
    double sizeDiff(cv::Rect2d& roi1, cv::Rect2d& roi2);

    /**
    * @brief compare parameters of predicted trajectory.
    * @param[in] data1, data2 : {frame,label,ax,bx,cx,ay,by,cy,az,bz,cz}
    * return absolute difference of parameters.
    */
    double compareParams(std::vector<double>& data1, std::vector<double>& data2);

    //find matched value
    int findIndex(const std::vector<int>& vec, int value);

    // Function to append vector b to vector a
    void concatenateVectors(std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b);
};

#endif