#pragma once

#include "stdafx.h"

class RLS {
public:
    RLS(int num_params, double lambda);
    void update(const Eigen::VectorXd& x, double y);
    Eigen::VectorXd getTheta() const;

private:
    Eigen::VectorXd theta;//parameters.
    Eigen::MatrixXd P;//covariance matrix.
    double lambda;//forgetting factor.
};