#pragma once

#ifndef KALMANFILTER_H
#define KALMANFILTER_H

#include "stdafx.h"
#include "global_parameters.h"

class KalmanFilter2D {
private:
    Eigen::Vector<double, 6> state_; // State estimate [x, y, vx, vy, ax, ay]
    Eigen::Matrix<double, 6, 6> P_;     // Estimate error covariance
    Eigen::Matrix<double, 6, 6> Q_;     // Process noise covariance
    Eigen::Matrix<double, 2, 2> R_;     // Measurement noise covariance
    Eigen::Matrix<double, 6, 6> A_;     // State transition matrix
    Eigen::Matrix<double, 2, 6> H_;     // Measurement matrix
    Eigen::Matrix<double, 6, 2> K_;     // Kalman gain
    double dt_ = 3.0 / (double)FPS*1000.0; // frame interval between Yolo inference
public:
    int counter_notUpdate = 0; //count not update time
    int counter_update = 0; // number of update

    //constructor
    KalmanFilter2D(double initial_x, double initial_y, double initial_vx, double initial_vy, double init_ax, double init_ay,
        double process_noise_pos, double process_noise_vel, double process_noise_acc, double measurement_noise) {
        // Initial state: [x, y, vx, vy, ax, ay]
        state_ << initial_x, initial_y, initial_vx, initial_vy, init_ax, init_ay; //column vector

        // Initial estimate error covariance
        P_ = Eigen::MatrixXd::Identity(6, 6);

        // Process noise covariance
        Q_ << process_noise_pos, 0, 0, 0, 0, 0,
            0, process_noise_pos, 0, 0, 0, 0,
            0, 0, process_noise_vel, 0, 0, 0,
            0, 0, 0, process_noise_vel, 0, 0,
            0, 0, 0, 0, process_noise_acc, 0,
            0, 0, 0, 0, 0, process_noise_acc;

        // Measurement noise covariance
        R_ = Eigen::MatrixXd::Identity(2, 2) * measurement_noise;
    }

    // Prediction step
    void predict(Eigen::Vector<double, 6>& prediction, double dframe, std::vector<std::vector<double>>& seqData);

    void predict_only(Eigen::Vector<double, 6>& prediction, double& dframe);

    // Update step
    void update(const Eigen::Vector2d& measurement);

    //get state
    Eigen::Vector<double, 6> getState() const;
};

#endif
