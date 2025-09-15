#pragma once

#ifndef GLOBAL_PARAMETERS_H
#define GLOBAL_PARAMETERS_H

#include "stdafx.h"
#include "utils/RLS.h"
#include "struct.h"

//Constant variables setting.
extern const double PI;
extern const int LEFT_CAMERA;
extern const int RIGHT_CAMERA;
// YOLO label
extern const int BALL;
extern const int BOX;


class GP {
public:
    // Model parameter settings as static member variables for global access via GP::member
    static inline bool boolGroundTruth = false;
    // video path
    static inline std::string path_to_video_left = "left.mp4";
    static inline std::string path_to_video_right = "right.mp4";
    // save directory
    static inline std::string save_directory;
    // camera: constant setting
    static inline int FPS = 300;
    // tracker
    static inline double threshold_mosse = 5.0; // PSR threshold
    static inline bool bool_skip = false;

    // Tracker setting
    static inline int COUNTER_VALID = 5; // frames by official tracker
    static inline int COUNTER_LOST = 50; // frames by deleting tracker
	static inline double process_noise_pos = 1e-4;
	static inline double process_noise_vel = 1e-4;
	static inline double process_noise_acc = 1e-4;
	static inline double measurement_noise = 1e4;
    // Identity setting
    static inline double MIN_IOU = 0.1; // minimum IoU for identity
    static inline double MAX_RMSE = 30; // max RMSE for identity

    // If you have additional parameters, add them as static inline as well.

    static void _loadParameter(const std::string& parameterFile) {
        //load parameter from .txt file and set the parameter as the class member.
        std::ifstream file(parameterFile);
        std::string line;
        while (std::getline(file, line))
        {
            std::istringstream iss(line);
            std::string key, value;
            iss >> key >> value;
            if (key == "PI") PI = std::stod(value);
            else if (key == "FPS") FPS = std::stoi(value);
            else if (key == "psr_threshold_mosse") threshold_mosse = std::stod(value);
            else if (key == "score_threshold_template_matching") score_threshold_template_matching = std::stod(value);
            else if (key == "K_SIGMA") K_SIGMA = std::stod(value);
            else if (key == "N_WARMUP") N_WARMUP = std::stoi(value);
            else if (key == "MAX_SKIP") MAX_SKIP = std::stoi(value);
            else if (key == "bool_skip") bool_skip = std::stod(value);
            else if (key == "COUNTER_VALID") COUNTER_VALID = std::stoi(value);
            else if (key == "COUNTER_LOST") COUNTER_LOST = std::stoi(value);
			else if (key == "process_noise_pos") process_noise_pos = std::stod(value);
			else if (key == "process_noise_vel") process_noise_vel = std::stod(value);
			else if (key == "process_noise_acc") process_noise_acc = std::stod(value);
			else if (key == "measurement_noise") measurement_noise = std::stod(value);
            else if (key == "MIN_IOU") MIN_IOU = std::stod(value);
            else if (key == "MAX_RMSE") MAX_RMSE = std::stod(value);
            else if (key == "boolGroundTruth") boolGroundTruth = std::stod(value);
            else if (key == "path_to_video_left") path_to_video_left = value;
            else if (key == "path_to_video_right") path_to_video_right = value;
            // _mode_duplication.
            // 0: no duplication, 1: duplication with IoU, 2: duplication with IoU and velocity.
            // 3: duplication with IoU and velocity, and augmentation in merging trackers.
            else if (key == "mode_duplication") mode_duplication = std::stoi(value);
            else if (key == "bool_TBD") bool_TBD = std::stod(value);
        }
        file.close();
    }

    // Static initialization function to load parameters from multiple files
    static void initialize(const std::string& parameterFile_main, const std::string& parameterFile_mot, const std::string& parameterFile_skeleton, const std::string& parameterFile_calibration)
    {
        _loadParameter(parameterFile_main);
        _loadParameter(parameterFile_mot);
        _loadParameter(parameterFile_skeleton);
        _loadParameter(parameterFile_calibration);
    }
};



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

//YOLO -> MOT.
extern std::queue<Yolo2MOT> q_yolo2mot;
//MOT -> Tracking.
extern std::queue<Trackers> q_mot2tracking_left, q_mot2tracking_right;
//Tracking -> MOT.
extern std::queue<Trackers2MOT> q_tracking2mot_left, q_tracking2mot_right;
//Finish flag from MOT and YOLO.

extern std::queue<bool> q_finish_mot,q_finish_tracking, q_finish_yolo;

// Yolo2seq
extern std::queue<Yolo2seq> q_yolo2seq_left, q_yolo2seq_right;//extern is for declaration for the compiler to look for the definition.


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


#endif