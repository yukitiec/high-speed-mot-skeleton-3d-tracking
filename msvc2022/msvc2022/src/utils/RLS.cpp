#include "../../include/utils/RLS.h"

RLS::RLS(int num_params, double lambda)
	: theta(Eigen::VectorXd::Zero(num_params)),
	  P(Eigen::MatrixXd::Identity(num_params, num_params) * 1000), // initial covariance matrix.
	  lambda(lambda)
{
}

void RLS::update(const Eigen::VectorXd &x, double y)
{
	Eigen::VectorXd Px = P * x;
	double denominator = lambda + x.transpose() * Px;
	Eigen::VectorXd K = Px / denominator;
	theta.noalias() += K * (y - x.transpose() * theta);
	P.noalias() -= K * x.transpose() * P;
	P /= denominator;
}

Eigen::VectorXd RLS::getTheta() const
{
	return theta;
}