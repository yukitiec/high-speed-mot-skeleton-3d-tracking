#pragma once

#ifndef KALMANFILTER_H
#define KALMANFILTER_H

#include "../stdafx.h"
#include "../global_parameters.h"

class KalmanFilter {
private:
    unsigned int k_;
    int state_size; // [pos, vel, acc] for each dimension
    int measurement_size; // position measurements only
	double measurement_noise;
	double process_noise_pos;
	double process_noise_vel;
	double process_noise_acc;
	unsigned int COUNTER_LOST;
	double FPS;
    
    Eigen::VectorXd state_; // State estimate [x1, y1, ..., xk, yk, vx1, vy1, ..., vxk, vyk, ax1, ay1, ..., axk, ayk]
    Eigen::MatrixXd P_;     // Estimate error covariance
    Eigen::MatrixXd Q_;     // Process noise covariance
    Eigen::MatrixXd R_;     // Measurement noise covariance
    Eigen::MatrixXd A_;     // State transition matrix
    Eigen::MatrixXd H_;     // Measurement matrix
    Eigen::MatrixXd K_;     // Kalman gain
    double dt_; // frame interval between Yolo inference
	double _frame_last = -0.1;//last update frame
    
public:
    int counter_notUpdate = 0; //count not update time
    int counter_update = 0; // number of update

    //constructor
    KalmanFilter(unsigned int k, double frame_current,
        const std::vector<double>& initial_pos, 
        const std::vector<double>& initial_vel,
        const std::vector<double>& initial_acc
	)
        : k_(k), state_size(3 * k), measurement_size(2*k) {
        
        // Validate input dimensions
        if (initial_pos.size() != k || initial_vel.size() != k || initial_acc.size() != k) {
            throw std::invalid_argument("Input vectors must have size k");
        }

		//set the noise parameters.
		// If you defined these as static inline members in GP (e.g., static inline double process_noise_pos;),
		// then this will work fine without passing a GP reference, as long as they are initialized before use.
		measurement_noise = GP::measurement_noise;
		process_noise_pos = GP::process_noise_pos;
		process_noise_vel = GP::process_noise_vel;
		process_noise_acc = GP::process_noise_acc;
		COUNTER_LOST = GP::COUNTER_LOST;
		FPS = GP::FPS;
		dt_ = 3.0 / (double)FPS * 1000.0; // frame interval between Yolo inference
		//reset counter.
		counter_notUpdate = 0;
		counter_update = 0;
		_frame_last = frame_current;

        // Initialize matrices with correct sizes
        state_ = Eigen::VectorXd::Zero(state_size);
        P_ = Eigen::MatrixXd::Identity(state_size, state_size);
        Q_ = Eigen::MatrixXd::Zero(state_size, state_size);
        R_ = Eigen::MatrixXd::Identity(measurement_size, measurement_size) * measurement_noise;
        A_ = Eigen::MatrixXd::Zero(state_size, state_size);
        H_ = Eigen::MatrixXd::Zero(measurement_size, state_size);
        K_ = Eigen::MatrixXd::Zero(state_size, measurement_size);
        
        // Initialize state vector: [pos, vel, acc] for each dimension
        for (int i = 0; i < k; ++i) {
            state_(i) = initial_pos[i];                    // position
            state_(k + i) = initial_vel[i];                // velocity
            state_(2 * k + i) = initial_acc[i];            // acceleration
        }

        // Process noise covariance
        for (int i = 0; i < k; ++i) {
            Q_(i, i) = process_noise_pos;                  // position noise
            Q_(k + i, k + i) = process_noise_vel;          // velocity noise
            Q_(2 * k + i, 2 * k + i) = process_noise_acc;  // acceleration noise
        }
    }

    // Prediction step
    bool predict(double frame_current);

    Eigen::VectorXd predict_only(double frame_current);

    // Update step
    void update(const Eigen::VectorXd& measurement);

    //get state
    Eigen::VectorXd getState() const;
    
    // Helper function to get position from state
    Eigen::VectorXd getPosition() const;
    
    // Helper function to get velocity from state
    Eigen::VectorXd getVelocity() const;
    
    // Helper function to get acceleration from state
    Eigen::VectorXd getAcceleration() const;
    
    // Get dimension
    unsigned int getDimension() const { return k_; }
};

//inline definition enhance the computational performance because each module is relatively small.
// Method implementations
inline bool KalmanFilter::predict(double frame_current) {
    dt_ = (frame_current - _frame_last) / (double)FPS * 1000.0; // millisec order
    
    // State transition matrix A for constant acceleration model
    A_.setZero();
    for (int i = 0; i < k_; ++i) {
        // Position update: x = x + v*dt + 0.5*a*dt^2
        A_(i, i) = 1.0;                    // pos -> pos
        A_(i, k_ + i) = dt_;               // vel -> pos
        A_(i, 2 * k_ + i) = 0.5 * dt_ * dt_; // acc -> pos
        
        // Velocity update: v = v + a*dt
        A_(k_ + i, k_ + i) = 1.0;          // vel -> vel
        A_(k_ + i, 2 * k_ + i) = dt_;      // acc -> vel
        
        // Acceleration update: a = a (constant acceleration model)
        A_(2 * k_ + i, 2 * k_ + i) = 1.0;  // acc -> acc
    }

    // Predict the next state
    state_ = A_ * state_;

    // Update the estimate error covariance
    P_ = A_ * P_ * A_.transpose() + Q_;
    counter_notUpdate++;
    
    if (counter_notUpdate == COUNTER_LOST) {
        counter_update = 0;
		return false;
    }
	//update the last update frame.
	_frame_last = frame_current;
	return true;
}

inline Eigen::VectorXd KalmanFilter::predict_only(double frame_current) {
    dt_ = (frame_current - _frame_last) / (double)FPS * 1000.0; // millisec order
    
    // State transition matrix A for constant acceleration model
    A_.setZero();
    for (int i = 0; i < k_; ++i) {
        // Position update: x = x + v*dt + 0.5*a*dt^2
        A_(i, i) = 1.0;                    // pos -> pos
        A_(i, k_ + i) = dt_;               // vel -> pos
        A_(i, 2 * k_ + i) = 0.5 * dt_ * dt_; // acc -> pos
        
        // Velocity update: v = v + a*dt
        A_(k_ + i, k_ + i) = 1.0;          // vel -> vel
        A_(k_ + i, 2 * k_ + i) = dt_;      // acc -> vel
        
        // Acceleration update: a = a (constant acceleration model)
        A_(2 * k_ + i, 2 * k_ + i) = 1.0;  // acc -> acc
    }

    // Predict the next state
    return A_ * state_;
}

inline void KalmanFilter::update(const Eigen::VectorXd& measurement) {
    // Measurement matrix H (we are measuring the position and velocityin all k dimensions)
    H_.setZero();
    for (int i = 0; i < k_; ++i) {
        H_(i, i) = 1.0; // position measurement
        H_(k_ + i, k_ + i) = 1.0; // velocity measurement
    }

    // Kalman gain
    K_ = P_ * H_.transpose() * (H_ * P_ * H_.transpose() + R_).inverse();

    // Update the state estimate
    state_ = state_ + K_ * (measurement - H_ * state_);

    // Update the estimate error covariance
    P_ = (Eigen::MatrixXd::Identity(state_size, state_size) - K_ * H_) * P_;
    counter_notUpdate = 0;
    counter_update++; // increment update counter
}

inline Eigen::VectorXd KalmanFilter::getState() const {
    return state_;
}

inline Eigen::VectorXd KalmanFilter::getPosition() const {
    return state_.head(k_);
}

inline Eigen::VectorXd KalmanFilter::getVelocity() const {
    return state_.segment(k_, k_);
}

inline Eigen::VectorXd KalmanFilter::getAcceleration() const {
    return state_.tail(k_);
}

#endif
