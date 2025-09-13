#pragma once

#ifndef GLOBAL_PARAMETERS_H
#define GLOBAL_PARAMETERS_H

#include "stdafx.h"
#include "utils/RLS.h"
#include "struct.h"

//MOT
extern const double PI;
extern const double omega_max_ur_;
extern const bool boolGroundTruth;
//video path
extern const std::string filename_left;
extern const std::string filename_right;
// camera : constant setting
extern const int LEFT_CAMERA;
extern const int RIGHT_CAMERA;
extern const int FPS;
// YOLO label
extern const int BALL;
extern const int BOX;
// tracker
extern const double threshold_mosse;//0.57; //PSR threshold

// tracking
extern const int COUNTER_VALID; //frames by official tracker
extern const int COUNTER_LOST; //frames by deleting tracker
extern const float MAX_ROI_RATE; //max change of roi
extern const float MIN_ROI_RATE; //minimum change of roi
extern const double MIN_IOU; //minimum IoU for identity
extern const double MAX_RMSE; //max RMSE fdor identity

/* save file setting */
extern const std::string rootDir_;
extern const std::string file_yolo_bbox_left;
extern const std::string file_yolo_class_left;
extern const std::string file_seq_left;
extern const std::string file_kf_left;
extern const std::string file_yolo_bbox_right;
extern const std::string file_yolo_class_right;
extern const std::string file_seq_right;
extern const std::string file_kf_right;
extern const std::string file_3d;
extern const std::string file_target;
extern const std::string file_params;
extern const std::string file_match;
//Robot control
extern const std::string file_joints_ivpf;
extern const std::string file_jointsAngle_ivpf;
extern const std::string file_minimumDist_ivpf;
extern const std::string file_human_ivpf;
extern const std::string file_determinant;
extern const std::string file_determinant_elbow;
extern const std::string file_determinant_wrist;
extern const std::string file_attraction;
extern const std::string file_repulsion;
extern const std::string file_tangent;
extern const std::string file_rep_global;
extern const std::string file_rep_att;
extern const std::string file_rep_elbow;
extern const std::string file_rep_wrist;
extern const std::string file_lambda;
extern const std::string file_eta_repulsive;
extern const std::string file_eta_tangent;
extern const std::string file_virtual;
extern const std::string file_target_robot;
extern const std::string file_vels;

// queue definitions
extern std::queue<std::array<cv::Mat1b, 2>> queueFrame_mot; // queue for frame
extern std::queue<int> queueFrameIndex_mot;  // queue for frame index
extern std::queue<std::array<cv::Mat1b, 2>> queueFrame_optflow; // queue for frame
extern std::queue<int> queueFrameIndex_optflow;  // queue for frame index
extern std::queue<std::array<cv::Mat1b, 2>> queueFrame_yolopose; // queue for frame
extern std::queue<int> queueFrameIndex_yolopose;  // queue for frame index
extern std::queue<int> queueFrameIndex_robot;  // queue for frame index


extern const double forgetting_factor;
extern const int dim_poly_x;//linear regression.
extern const int dim_poly_y;//quadratic regression.
extern const int dim_poly_z;//linear regression.

// Yolo2seq
extern std::queue<Yolo2seq> q_yolo2seq_left, q_yolo2seq_right;//extern is for declaration for the compiler to look for the definition.
//yolo2buffer
extern std::queue<Yolo2buffer> q_yolo2buffer;
//seq2tri
extern std::queue<std::vector<std::vector<std::vector<double>>>> q_seq2tri_left, q_seq2tri_right;
//seq2robot
extern std::queue<std::vector<Seq2robot>> q_trajectory_params;
extern std::queue<Seq2robot_send> q_seq2robot;
//CPU 2 GPU
extern std::queue<Info2Gpu> q_img2gpu;

extern std::queue<bool> q_startTracking; //start tracking
extern std::queue<bool> q_endTracking; //end tracking

//mutex
extern std::mutex mtx_yolo2seq; // define mutex


//Skeleton
extern const int LEFT;
extern const int RIGHT;
extern const bool save;
extern const bool boolSparse;
extern const bool boolGray;
extern const bool boolBatch; //if yolo inference is run in concatenated img
extern const std::string methodDenseOpticalFlow; //"lucasKanade_dense","rlof"
extern const float qualityCorner;
/* roi setting */
extern const bool bool_dynamic_roi; //adopt dynamic roi
extern const bool bool_rotate_roi;
//if true
extern const float max_half_diagonal;//70
extern const float min_half_diagonal;//15
extern const int roiWidthOF;
extern const int roiHeightOF;
extern const int roiWidthYolo;
extern const int roiHeightYolo;
extern const float MoveThreshold; //cancell background
extern const float epsironMove;//half range of back ground effect:: a-epsironMove<=flow<=a+epsironMove
/* dense optical flow skip rate */
extern const int skipPixel;
extern const float DIF_THRESHOLD; //threshold for adapting yolo detection's roi
extern const float MIN_MOVE; //minimum opticalflow movement
extern const float MAX_MOVE;
/*if exchange template of Yolo */
extern const bool boolChange;

//Kalman filter setting
extern const double INIT_X;
extern const double INIT_Y;
extern const double INIT_Z;
extern const double INIT_VX;
extern const double INIT_VY;
extern const double INIT_VZ;
extern const double INIT_AX;
extern const double INIT_AY;
extern const double INIT_AZ;
extern const double NOISE_POS;
extern const double NOISE_VEL;
extern const double NOISE_ACC;
extern const double NOISE_SENSOR;
extern const double NOISE_POS_3D;
extern const double NOISE_VEL_3D;
extern const double NOISE_ACC_3D;
extern const double NOISE_SENSOR_3D;

extern const int COUNTER_LOST_HUMAN;//humman life span.

/* save date */
extern const std::string file_yolo_left;
extern const std::string file_yolo_right;
extern const std::string file_of_left;
extern const std::string file_of_right;
extern const std::string file_kf_skeleton_left;
extern const std::string file_kf_skeleton_right;
extern const std::string file_measure_skeleton_left;
extern const std::string file_measure_skeleton_right;
extern const std::string file_3d_pose;
extern const std::string file_kf_skeleton_3d;
extern const std::string file_measure_skeleton_3d;

//structure
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

/*3D position*/
extern std::queue<Optflow2tri> q_optflow2tri_left;
extern std::queue<Optflow2tri> q_optflow2tri_right;

/* from joints to robot control */
extern std::queue<std::vector<std::vector<std::vector<double>>>> queueJointsPositions;//{n_human,n_joints,(frame,x,y,z)}
extern std::queue<skeleton2robot> q_skeleton2robot;
/* notify danger */
extern std::queue<bool> queueDanger;


//queue
extern std::queue<Yolo2Buffer_skeleton> q_yolo2buffer_skeleton;
extern std::queue<Yolo2optflow> q_yolo2optflow_left, q_yolo2optflow_right;
extern std::queue<Optflow2optflow> q_optflow2optflow_left, q_optflow2optflow_right;

extern std::queue<bool> q_startOptflow;

extern std::mutex mtx_img, mtxRobot, mtxYolo_left, mtxYolo_right, mtxTri;

extern std::vector<std::vector<double>> wrists;//{#(human)*#(left,right),(frame,x,y,z)}

//UR setting
using namespace ur_rtde;

extern const std::string URIP;
extern std::unique_ptr<RTDEControlInterface> urCtrl;
extern std::unique_ptr<RTDEIOInterface> urDO;
extern std::unique_ptr<RTDEReceiveInterface> urDI;

extern std::vector<double> ee_current_;

#endif