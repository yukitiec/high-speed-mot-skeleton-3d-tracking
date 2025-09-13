#pragma once

#ifndef KALMANFILTER_3D_H
#define KALMANFILTER_3D_H

#include "stdafx.h"
#include "global_parameters.h"

class KalmanFilter3D {
private:

    Eigen::Matrix<double, 9, 9> P_;     // Estimate error covariance
    Eigen::Matrix<double, 9, 9> Q_;     // Process noise covariance
    Eigen::Matrix<double, 3, 3> R_;     // Measurement noise covariance
    Eigen::Matrix<double, 9, 9> A_;     // State transition matrix
    Eigen::Matrix<double, 3, 9> H_;     // Measurement matrix
    Eigen::Matrix<double, 9, 3> K_;     // Kalman gain
    double dt_ = 1.0 / (double)FPS; // frame interval between Yolo inference
public:
    int counter_update = 0; // number of update
    int counter_notUpdate = 0; //count not update time
    double frame_last = -0.1;//last update frame
    Eigen::Vector<double, 9> state_; // State estimate [x, y, vx, vy, ax, ay]

    //constructor
    KalmanFilter3D(double initial_x, double initial_y, double initial_z, double initial_vx, double initial_vy, double initial_vz, double init_ax, double init_ay, double init_az,
        double process_noise_pos, double process_noise_vel, double process_noise_acc, double measurement_noise) {
        // Initial state: [x, y, vx, vy, ax, ay]
        state_ << initial_x, initial_y, initial_z, initial_vx, initial_vy, initial_vz, init_ax, init_ay, init_az; //column vector

        // Initial estimate error covariance
        P_ = Eigen::MatrixXd::Identity(9, 9);

        // Process noise covarianc
        Q_ << process_noise_pos, 0, 0, 0, 0, 0, 0, 0, 0,
            0, process_noise_pos, 0, 0, 0, 0, 0, 0, 0,
            0, 0, process_noise_pos, 0, 0, 0, 0, 0, 0,
            0, 0, 0, process_noise_vel, 0, 0, 0, 0, 0,
            0, 0, 0, 0, process_noise_vel, 0, 0, 0, 0,
            0, 0, 0, 0, 0, process_noise_vel, 0, 0, 0,
            0, 0, 0, 0, 0, 0, process_noise_acc, 0, 0,
            0, 0, 0, 0, 0, 0, 0, process_noise_acc, 0,
            0, 0, 0, 0, 0, 0, 0, 0, process_noise_acc;

        // Measurement noise covariance
        R_ = Eigen::MatrixXd::Identity(3, 3) * measurement_noise;
    }

    // Prediction step
    void predict(double& dframe);

    void predict_only(Eigen::Vector<double, 9>& prediction, double& dframe);

    // Update step
    void update(const Eigen::Vector3d& measurement);

    //get state
    Eigen::Vector<double, 9> getState() const;
};

#endif
