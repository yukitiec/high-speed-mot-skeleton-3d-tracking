#pragma once

#ifndef MOT_H
#define MOT_H

#include "stdafx.h"
#include "utils/kalmanfilter.h"
#include "global_parameters.h"
#include "utils/utility.h"
#include "utils/hungarian.h"
#include "triangulation/triangulation.h"
#include "utils/prediction.h"
#include "utils/extrapolation.h"
#include "utils/RLS.h"

class MOT
{
private:

    //prediction
    int idx_compensation = 0;//0:kalman filter, 1:linear extrapolation.
    //trajectory predictioni
    int method_prediction = 0;//0:ordinary least square method. 1: recursive least squares method.
	int _n_max_highspeed = 10;//the maximum number of high speed tracking.

    std::vector<std::vector<double>> defaultVector{ {0.0} }; //for initializing kalman filter
    //matching
    const double IoUThresholdIdentity = 0.1; // for maitainig consistency of trackingS
    const double Rmse_identity = 100.0; // minimum rmse criteria
    const double threshold_area_ratio = 4.0;//2.0*2.0
    const double Cost_max = 1000.0;
    const double Cost_params_max = 0.0;
    const double lambda_rmse_ = 2.0;

	//storage.
	Trackers2MOT trackers2mot_left, trackers2mot_right;
	Yolo2MOT yolo2mot;
	TrackersYOLO trackersYOLO_left, trackersYOLO_right;
	TrackersMOT newData_left, newData_right;
	Trackers_sequence trackers_mot_left, trackers_mot_right;
	cv::Mat1b frameYolo;
	int frameIndex;

    //the minimum number of updating predictions.
    const int counter_update_params_ = 3;

    //hungarian algorithm
    HungarianAlgorithm HungAlgo;

    //observation
    Eigen::Vector2d observation;
    //for kalmanfilter prediction result
    Eigen::Vector<double, 6> kf_predict;

    //triangulation
    std::string rootDir;
    Triangulation tri; 
    Matching match; //matching algorithm

	//Storage.
	//define variables.
	std::vector<cv::Rect2d> newRoi_left, newRoi_right; //new Roi from Yolo inference
	std::vector<int> newLabel_left, newLabel_right; //new class labels from Yolo inference
	std::vector<std::vector<int>> seqClasses_left, seqClasses_right; // storage for sequential classes

	//Yolo2sequence
	std::vector<torch::Tensor> rois;
	std::vector<double> _scores;
	std::vector<int> labels;
	std::vector<cv::Rect2d> roi_left, roi_right;
	std::vector<int> class_left, class_right;
	std::vector<double> scores_left, scores_right;

	//for triangulation
	std::vector<std::vector<std::vector<double>>> data_3d, data_3d_save; //{num of objects, sequential, { frameIndex,label, X,Y,Z }}
	std::vector<std::vector<std::vector<int>>> matching_save;//{n_seq,n_objects,{idx_left,idx_right}}
	std::vector<int> frame_matching;

	std::vector<std::vector<double>> initial_add(1, std::vector<double>(5, 0.0));//{frame,label,x,y,z};initialize with 0.0
	std::vector<std::vector<double>> initial_add_params;
	std::vector<std::vector<double>> initial_add_target(1, std::vector<double>(8, 0.0));//{frame,label,x,y,Fz,nx,ny,nz};initialize with 0.0

	int n_features;

	//for prediction
	std::vector<std::vector<std::vector<double>>> targets, params, targets_save, params_save;//{num of objects, sequence, targets:{frame,label,x,y,z}}, params:{{frame,label,a_x,b_x,c_x,a_y,b_y,c_y,a_z,b_z,c_z}}
	std::vector<double> params_latest, params_prev_latest;
	std::vector<rls> instances_rls;//RLS instance (Recursive Least Squares method)

	int label_latest, label_prev;



public:
    //storage
    std::vector<std::vector<std::vector<double>>> seqData_left, seqData_right, kfData_left, kfData_right, 
	saveData_left, saveData_right, saveKFData_left, saveKFData_right; //{num of objects, num of sequence, unit vector}. {frame,label,left,top,width,height}
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

    MOT(int idx_compensation=0, int method_prediction=0, int n_max_highspeed=10, const std::string& rootDir="")
        : idx_compensation(idx_compensation), method_prediction(method_prediction), _n_max_highspeed(n_max_highspeed), tri(rootDir)
    {
		if (method_prediction == 0) {//Least square method
			n_features = 2 + dim_poly_x + dim_poly_y + dim_poly_z;//dim_poly_x,y,z : global parameters for trajectory prediction. see global_parameters.h
			initial_add_params = std::vector<std::vector<double>>(1, std::vector<double>(n_features, 0.0));
	
		}
		else if (method_prediction == 1) {//Recursive least square method
			n_features = 2 + dim_poly_x + dim_poly_y + dim_poly_z;
			initial_add_params = std::vector<std::vector<double>>(1, std::vector<double>(n_features, 0.0));
		}
	
		rls init_rls(dim_poly_x, dim_poly_y, dim_poly_z, forgetting_factor);//initial RLS 
    };

    /**
    * @brief main function. update every time sequence get new data from Yolo.
    */
    void main();

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