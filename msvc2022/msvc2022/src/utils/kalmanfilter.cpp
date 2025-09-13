#include "kalmanfilter.h"

void KalmanFilter2D::predict(Eigen::Vector<double, 6>& prediction, double dframe, std::vector<std::vector<double>>& seqData) {
    dt_ = dframe / (double)FPS * 1000.0;//millisec order
    // State transition matrix A for constant acceleration model
    A_ << 1, 0, dt_, 0, 0.5 * dt_ * dt_, 0,
        0, 1, 0, dt_, 0, 0.5 * dt_ * dt_,
        0, 0, 1, 0, dt_, 0,
        0, 0, 0, 1, 0, dt_,
        0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1;

    // Predict the next state
    state_ = A_ * state_;
    prediction = state_;

    // Update the estimate error covariance
    P_ = A_ * P_ * A_.transpose() + Q_;
    counter_notUpdate++;
    if (counter_notUpdate == COUNTER_LOST)
    {
        seqData.push_back({ -1,-1,-1,-1,-1,-1 });
        prediction << -1, -1, -1, -1, -1, -1;
        counter_update = 0;
    }
}

void KalmanFilter2D::predict_only(Eigen::Vector<double, 6>& prediction, double& dframe) {
    dt_ = dframe / (double)FPS * 1000.0;//millisec order
    // State transition matrix A for constant acceleration model
    A_ << 1, 0, dt_, 0, 0.5 * dt_ * dt_, 0,
        0, 1, 0, dt_, 0, 0.5 * dt_ * dt_,
        0, 0, 1, 0, dt_, 0,
        0, 0, 0, 1, 0, dt_,
        0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1;

    // Predict the next state
    prediction = A_ * state_;
}

// Update step
void KalmanFilter2D::update(const Eigen::Vector2d& measurement) {
    // Measurement matrix H (we are measuring the position in x and y)
    H_ << 1, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0; //2*6 vector

    // Kalman gain
    K_ = P_ * H_.transpose() * (H_ * P_ * H_.transpose() + R_).inverse(); //6*2 matrix

    // Update the state estimate
    state_ = state_ + K_ * (measurement - H_ * state_); //6*1 + (6*2)*(2*1) (2*6)*(6*1)

    // Update the estimate error covariance
    P_ = (Eigen::MatrixXd::Identity(6, 6) - K_ * H_) * P_;
    counter_notUpdate = 0;
    counter_update++; //increment update counter
}

Eigen::Vector<double, 6> KalmanFilter2D::getState() const {
    return state_;
}