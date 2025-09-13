#pragma once

#ifndef PREDICTION_H
#define PREDICTION_H

#include "stdafx.h"
#include "utility.h"
#include "global_parameters.h"
#include "RLS.h"

class Prediction
{
private:
    //number of points for predicting trajectory
    const int N_POINTS_PREDICT = 20.0;
public:
    cv::Vec3d coefX, coefY, coefZ; //predicted trajectory. {a2,a1,a0} (y=a2*t^2+a1*t+a0)
    Eigen::VectorXd coeff_x, coeff_y, coeff_z;
    int size;


    Prediction()
    {
        std::cout << "construct Prediction class" << std::endl;
    }

    void predictTargets(int& index, double& depth_target, std::vector<std::vector<double>>& data, std::vector<std::vector<std::vector<double>>>& targets3D);

    /**
    * @brief Predict trajectory with RLS (Recursive Least Squares) method.
    */
    std::vector<Eigen::VectorXd> predictTargets_rls(int& index, double& depth_target, std::vector<std::vector<double>>& data, std::vector<rls>& instances_rls, std::vector<std::vector<std::vector<double>>>& targets3D);


    /**
    * @brief predict 3D target point
    * @param[in] depth_target target depth
    * @param[in] params predicted trajectory params.{frame_current,label,param_x,param_y,param_z}
    * @param[in] instances_rls RLS instances. (x,y,z)
    * @param[out] targets3D storage for target 3D points.{frame,label,x,y,z,nx,ny,nz}}
    */
    void calculate_target(double& depth_target, Seq2robot& params, std::vector<double>& target);

    /**
    * @brief Function to perform least squares fitting using OpenCV
    * @param[in] data : {n_seq, {frame,label,x,y,z}}
    * @param[in] index : which parameters to use.
    * @return coefficients {a,b,c}. a*t^2+b*t+c
    */
    cv::Vec3d fitQuadratic(const std::vector<std::vector<double>>& data, int index);

    /**
    * @brief calculate target frame.
    * @param[in] coef coefficients (a,b,c). a*t^2+b*t+c.
    * @param[in] depth_target target depth.
    * @return target frame.
    */
    double calculateTargetFrame(const cv::Vec3d& coef, const double& depth_target, double& frame_latest);

    double calculateTargetFrame_rls(const Eigen::VectorXd& coef, const double& depth_target, double& frame_latest);

    cv::Vec3d linearRegression(int& n_points_predict, std::vector<std::vector<double>>& data);

    cv::Vec3d linearRegressionY(int& n_points_predict, std::vector<std::vector<double>>& data);

    void curveFitting(int& n_points_predict, std::vector<std::vector<double>>& data, std::vector<double>& result);
};

#endif
