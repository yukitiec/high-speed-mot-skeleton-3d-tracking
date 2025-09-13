#include "global_parameters.h"

const double PI = 3.14159265358979323846;
const double omega_max_ur_ = 6.28;
const bool boolGroundTruth = false;
//video path
const std::string filename_left = "left.mp4";
const std::string filename_right = "right.mp4";
// camera : constant setting
const int LEFT_CAMERA = 0;
const int RIGHT_CAMERA = 1;
const int FPS = 300;
// YOLO label
const int BALL = 0;
const int BOX = 1;

// tracking
const int COUNTER_VALID = 5; //frames by official tracker
const int COUNTER_LOST = (int)((double)FPS * 1.0); //frames by deleting tracker
const float MAX_ROI_RATE = 2.0; //max change of roi
const float MIN_ROI_RATE = 0.5; //minimum change of roi
const double MIN_IOU = 0.1; //minimum IoU for identity
const double MAX_RMSE = 30; //max RMSE for identity.

//RLS :: paramter setting.
const double forgetting_factor = 0.99;
const int dim_poly_x = 2;//linear regression.
const int dim_poly_y = 2;//linear regression.
const int dim_poly_z = 3;//quadratic regression.

/* save file setting */
const std::string rootDir_ = "C:/Users/kawaw/cpp/collaboration_catching_realworld/csv";
const std::string file_yolo_bbox_left = rootDir_ + "/yolo_bbox_left.csv";
const std::string file_yolo_class_left = rootDir_ + "/yolo_class_left.csv";
const std::string file_seq_left = rootDir_ + "/seqData_left.csv";
const std::string file_kf_left = rootDir_ + "/kfData_left.csv";
const std::string file_yolo_bbox_right = rootDir_ + "/yolo_bbox_right.csv";
const std::string file_yolo_class_right = rootDir_ + "/yolo_class_right.csv";
const std::string file_seq_right = rootDir_ + "/seqData_right.csv";
const std::string file_kf_right = rootDir_ + "/kfData_right.csv";
const std::string file_3d = rootDir_ + "/triangulation_mot.csv";
const std::string file_target = rootDir_ + "/target.csv";
const std::string file_params = rootDir_ + "/params.csv";
const std::string file_match = rootDir_ + "/match.csv";
//robot control
const std::string file_joints_ivpf = rootDir_ + "/joints_ivpf.csv";
const std::string file_jointsAngle_ivpf = rootDir_ + "/joints_angle.csv";
const std::string file_minimumDist_ivpf = rootDir_ + "/minimum_distance_ivpf.csv";
const std::string file_human_ivpf = rootDir_ + "/human_pose.csv";
const std::string file_determinant = rootDir_ + "/determinant.csv";
const std::string file_determinant_elbow = rootDir_ + "/determinant_elbow.csv";
const std::string file_determinant_wrist = rootDir_ + "/determinant_wrist.csv";
const std::string file_attraction = rootDir_ + "/gain_attraction.csv";
const std::string file_repulsion = rootDir_ + "/gain_repulsion.csv";
const std::string file_tangent = rootDir_ + "/gain_tangent.csv";
const std::string file_rep_global = rootDir_ + "/gain_rep_global.csv";
const std::string file_rep_att = rootDir_ + "/gain_rep_attractive.csv";
const std::string file_rep_elbow = rootDir_ + "/gain_elbow.csv";
const std::string file_rep_wrist = rootDir_ + "/gain_wrist.csv";
const std::string file_lambda = rootDir_ + "/lambda.csv";
const std::string file_eta_repulsive = rootDir_ + "/eta_repulsive.csv";
const std::string file_eta_tangent = rootDir_ + "/eta_tangent.csv";
const std::string file_virtual = rootDir_ + "/virtual.csv";
const std::string file_target_robot = rootDir_ + "/target_robot.csv";
const std::string file_vels = rootDir_ + "/velocities.csv";


// queue definitions
std::queue<std::array<cv::Mat1b, 2>> queueFrame_mot, queueFrame_optflow, queueFrame_yolopose; // queue for frame
std::queue<int> queueFrameIndex_mot, queueFrameIndex_optflow, queueFrameIndex_yolopose, queueFrameIndex_robot;  // queue for frame index
//YOLO2Buffer
std::queue<Yolo2buffer> q_yolo2buffer;
// Yolo2seq
std::queue<Yolo2seq> q_yolo2seq_left, q_yolo2seq_right;
//seq2tri
std::queue<std::vector<std::vector<std::vector<double>>>> q_seq2tri_left, q_seq2tri_right;
//seq2robot
std::queue<std::vector<Seq2robot>> q_trajectory_params;
std::queue<Seq2robot_send> q_seq2robot;
//CPU 2 GPU
std::queue<Info2Gpu> q_img2gpu;

//start and end signal
std::queue<bool> q_startTracking; //start tracking
std::queue<bool> q_endTracking; //end tracking

std::mutex mtx_yolo2seq;

//SKELETON
const int LEFT = 0;
const int RIGHT = 1;
const bool save = true;
const bool boolSparse = false;
const bool boolGray = true;
const bool boolBatch = true; //if yolo inference is run in concatenated img
const std::string methodDenseOpticalFlow = "farneback"; //"lucasKanade_dense","rlof
const float qualityCorner = 0.01;
/* roi setting */
const bool bool_dynamic_roi = true; //adopt dynamic roi
const bool bool_rotate_roi = true;
//if true
const float max_half_diagonal = 50.0 * std::pow(2, 0.5);//70
const float min_half_diagonal = 20.0 * std::pow(2, 0.5);//15
//if false : static roi
const int roiWidthOF = 60;
const int roiHeightOF = 60;
const int roiWidthYolo = 60;
const int roiHeightYolo = 60;
const float MoveThreshold = 0.0; //cancell background
const float epsironMove = 0.05;//half range of back ground effect:: a-epsironMove<=flow<=a+epsironMove
/* dense optical flow skip rate */
const int skipPixel = 1;
const float DIF_THRESHOLD = 1.0; //threshold for adapting yolo detection's roi
const float MIN_MOVE = 1.0; //minimum opticalflow movement : square value
const float MAX_MOVE = 30.0;
/*if exchange template of Yolo */
const bool boolChange = true;

//Kalman filter setting
const double INIT_X = 0.0;
const double INIT_Y = 0.0;
const double INIT_Z = 0.0;
const double INIT_VX = 0.0;
const double INIT_VY = 0.0;
const double INIT_VZ = 0.0;
const double INIT_AX = 0.0;
const double INIT_AY = 0.0;
const double INIT_AZ = 0.0;
const double NOISE_POS = 1e-4;
const double NOISE_VEL = 1e-4;
const double NOISE_ACC = 1e-4;
const double NOISE_SENSOR = 1e4;
const double NOISE_POS_3D = 1e-4;
const double NOISE_VEL_3D = 1e-4;
const double NOISE_ACC_3D = 1e-4;
const double NOISE_SENSOR_3D = 1e4;

const int COUNTER_LOST_HUMAN = 100;//humman life span.

/* save date */
const std::string file_yolo_left = rootDir_ + "/yolo_left.csv";
const std::string file_yolo_right = rootDir_ + "/yolo_right.csv";
const std::string file_of_left = rootDir_ + "/opticalflow_left.csv";
const std::string file_of_right = rootDir_ + "/opticalflow_right.csv";
const std::string file_kf_skeleton_left = rootDir_ + "/kf_skeleton_left.csv";
const std::string file_kf_skeleton_right = rootDir_ + "/kf_skeleton_right.csv";
const std::string file_measure_skeleton_left = rootDir_ + "/measure_skeleton_left.csv";
const std::string file_measure_skeleton_right = rootDir_ + "/measure_skeleton_right.csv";
const std::string file_3d_pose = rootDir_ + "/triangulation_skeleton.csv";
const std::string file_kf_skeleton_3d = rootDir_ + "/triangulation_skeleton_kf.csv";
const std::string file_measure_skeleton_3d = rootDir_ + "/triangulation_skeleton_measure.csv";

std::mutex mtx_img, mtxRobot, mtxYolo_left, mtxYolo_right, mtxTri;

/*3D position*/
std::queue<Optflow2tri> q_optflow2tri_left;
std::queue<Optflow2tri> q_optflow2tri_right;

/* from joints to robot control */
std::queue<std::vector<std::vector<std::vector<double>>>> queueJointsPositions;
std::queue<skeleton2robot> q_skeleton2robot;

/* notify danger */
std::queue<bool> queueDanger;

//queue
std::queue<Yolo2Buffer_skeleton> q_yolo2buffer_skeleton;
std::queue<Yolo2optflow> q_yolo2optflow_left, q_yolo2optflow_right;
std::queue<Optflow2optflow> q_optflow2optflow_left, q_optflow2optflow_right;

std::queue<bool> q_startOptflow;

std::vector<std::vector<double>> wrists;

//UR setting
extern const std::string URIP = "169.254.52.209";
std::unique_ptr<RTDEControlInterface> urCtrl = std::make_unique<RTDEControlInterface>(URIP);
//RTDEControlInterface urCtrl(URIP);
std::unique_ptr<RTDEIOInterface> urDO = std::make_unique<RTDEIOInterface>(URIP);
std::unique_ptr<RTDEReceiveInterface> urDI = std::make_unique<RTDEReceiveInterface>(URIP);

std::vector<double> ee_current_;