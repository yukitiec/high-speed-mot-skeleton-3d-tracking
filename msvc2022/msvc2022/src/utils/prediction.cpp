#include "prediction.h"

void Prediction::predictTargets(
    int& index, double& depth_target,
    std::vector<std::vector<double>>& data, std::vector<std::vector<std::vector<double>>>& targets3D
)
{
    /**
    * @brief predict 3D trajectory
    * @param[in] index object index whose update is over 3
    * @param[out] depth_target target depth
    * @param[in] data sequential data. shape is like {sequence, {frame,label,x,y,z}}
    * @param[out] targets3D storage for target 3D points. shape is like {n_objects, sequence, {frame,label,x,y,z}}
    */

    int n_seq = data.size();
    int n_points_predict = std::min(n_seq, N_POINTS_PREDICT);
    //// trajectoryPrediction 
    coefX = linearRegression(n_points_predict, data);
    coefY = linearRegressionY(n_points_predict, data);
    //curveFitting(n_points_predict, data, coefY);
    double frame_latest, label, t, x, y, z, frameTarget, xTarget, yTarget, zTarget;
    frame_latest = data.back()[0] / (double)FPS;//frame -> [sec]
    label = data.back()[1];

    //coefX = fitQuadratic(data, 1);//x: index<-index+1 in the function due to the existence of label.
    //coefY = fitQuadratic(data, 2);//y
    coefZ = fitQuadratic(data, 3);//z
    frameTarget = calculateTargetFrame(coefX, depth_target, frame_latest);
    if (frameTarget > frame_latest) {
        xTarget = depth_target;
        zTarget = coefZ(0) * frameTarget * frameTarget + coefZ(1) * frameTarget + coefZ(2);
        yTarget = coefY(0) * frameTarget * frameTarget + coefY(1) * frameTarget + coefY(2);
        //not empty
        if (!targets3D.empty())
        {
            if (targets3D[index][0][0] == 0.0)//not saved yet
                targets3D[index][0] = std::vector<double>{ (double)((int)(frameTarget * (double)FPS)), label,xTarget, yTarget,zTarget }; // push_back target position
            else //empty
                targets3D[index].push_back({ (double)((int)(frameTarget * (double)FPS)),label, xTarget, yTarget, zTarget });
        }
    }
}

std::vector<Eigen::VectorXd> Prediction::predictTargets_rls(
    int& index, double& depth_target, std::vector<std::vector<double>>& data,
    std::vector<rls>& instances_rls, std::vector<std::vector<std::vector<double>>>& targets3D
) {
    /**
    * @brief predict 3D trajectory with Recursive Least squares method.
    * @param[in] index object index whose update is over 3
    * @param[in] depth_target target depth
    * @param[in] data sequential data. shape is like {sequence, {frame,label,x,y,z}}
    * @param[in] instances_rls RLS instances. (x,y,z)
    * @param[out] targets3D storage for target 3D points. shape is like {n_objects, sequence, {frame,label,x,y,z}}
    */

    double frame_latest, label, t, x, y, z, frameTarget, xTarget, yTarget, zTarget;

    frame_latest = data.back()[0] / (double)FPS;//[sec] RLS is sensitive to the scale.
    label = data.back()[1];
    t = data.back()[0] / (double)FPS;//convert [frame]->O(10)[sec]
    x = data.back()[2];//O(1)
    y = data.back()[3];//O(1)
    z = data.back()[4];//O(1)

    //push data in the vector.
    Eigen::VectorXd xVec(dim_poly_x);
    Eigen::VectorXd yVec(dim_poly_y);
    Eigen::VectorXd zVec(dim_poly_z);

    //x
    for (int i = 0; i < dim_poly_x; i++)
        xVec(i) = std::pow(t, dim_poly_x - i - 1);//i=0,1. size-i-1: 1,0
    //y
    for (int i = 0; i < dim_poly_y; i++)
        yVec(i) = std::pow(t, dim_poly_y - i - 1);
    //z
    for (int i = 0; i < dim_poly_z; i++)
        zVec(i) = std::pow(t, dim_poly_z - i - 1);

    //update
    //x
    instances_rls[index].rlsx.update(xVec, x);
    //y
    instances_rls[index].rlsy.update(yVec, y);
    //z
    instances_rls[index].rlsz.update(zVec, z);

    //get parameters
    coeff_x = instances_rls[index].rlsx.getTheta();//at+b
    coeff_y = instances_rls[index].rlsy.getTheta();//at+b
    coeff_z = instances_rls[index].rlsz.getTheta();//at^2+bt+c

    std::vector<Eigen::VectorXd> coeffs;
    coeffs.push_back(coeff_x);
    coeffs.push_back(coeff_y);
    coeffs.push_back(coeff_z);

    //calculate Target frame
    if (dim_poly_x == 3) {//quadratic regression
        frameTarget = calculateTargetFrame_rls(coeff_x, depth_target, frame_latest);//[sec]
        xTarget = depth_target;
    }
    else if (dim_poly_x == 2) {//linear regression
        if (std::abs(coeff_x(0)) > 0) {//negative to positive
            frameTarget = (depth_target - coeff_x(1)) / coeff_x(0);//[sec]
            xTarget = depth_target;
        }
        else {
            frameTarget = 0.0;
            xTarget = 0.0;
        }
    }

    if (frameTarget > frame_latest) {
        //z
        if (dim_poly_z == 3)
            zTarget = coeff_z(0) * frameTarget * frameTarget + coeff_z(1) * frameTarget + coeff_z(2);
        else if (dim_poly_z == 2)
            zTarget = coeff_z(0) * frameTarget + coeff_z(1);
        //y
        if (dim_poly_y == 3)
            yTarget = coeff_y(0) * frameTarget * frameTarget + coeff_y(1) * frameTarget + coeff_y(2);
        else if (dim_poly_y == 2)
            yTarget = coeff_y(0) * frameTarget + coeff_y(1);
        //rotation vector
        double nx, ny, nz, n_norm;
        // nx
        if (dim_poly_x == 3)
            nx = 2.0 * coeff_x(0) * frameTarget + coeff_x(1);
        else if (dim_poly_x == 2)
            nx = coeff_x(0);
        // ny
        if (dim_poly_y == 3)
            ny = 2.0 * coeff_y(0) * frameTarget + coeff_y(1);
        else if (dim_poly_y == 2)
            ny = coeff_y(0);
        // nz
        if (dim_poly_z == 3)
            nz = 2.0 * coeff_z(0) * frameTarget + coeff_z(1);
        else if (dim_poly_z == 2)
            nz = coeff_z(0);
        n_norm = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (n_norm > 1.0e-6) {//adopt -v for target pose.
            nx = (-1.0) * nx / n_norm;
            ny = (-1.0) * ny / n_norm;
            nz = (-1.0) * nz / n_norm;
        }
        else {
            nx = 0.0;
            ny = 0.0;
            nz = 0.0;
        }

        //not empty
        if (!targets3D.empty())
        {
            if (targets3D[index][0][0] == 0.0)//not saved yet
                targets3D[index][0] = std::vector<double>{ (double)((int)(frameTarget * (double)FPS)), label,xTarget,yTarget, zTarget,nx,ny,nz }; // push_back target position
            else //empty
                targets3D[index].push_back({ (double)((int)(frameTarget * (double)FPS)),label, xTarget, yTarget, zTarget,nx,ny,nz });
        }
    }
    return coeffs;
}

void Prediction::calculate_target(double& depth_target, Seq2robot& params, std::vector<double>& target) {

    /**
    * @brief predict 3D target point
    * @param[in] depth_target target depth
    * @param[in] params predicted trajectory params.{frame_current,label,param_x,param_y,param_z}
    * @param[in] instances_rls RLS instances. (x,y,z)
    * @param[out] targets3D storage for target 3D points.{frame,label,x,y,z,nx,ny,nz}}
    */

    double frame_latest, label, t, x, y, z, frameTarget, xTarget, yTarget, zTarget;
    Eigen::Vector2d param_x, param_y;
    Eigen::Vector3d param_z;
    //substitute values
    param_x = params.param_x;
    param_y = params.param_y;
    param_z = params.param_z;
    label = params.label;
    double frame_current = params.frame_current / (double)FPS;//frame -> [sec]->[10sec]

    //calculate Target frame
    if (dim_poly_x == 3) {//quadratic regression
        frameTarget = calculateTargetFrame_rls(param_x, depth_target, frame_current);
        xTarget = depth_target;
    }
    else if (dim_poly_x == 2) {//linear regression
        if (std::abs(param_x(0)) > 0.0) {//negative to positive
            frameTarget = (depth_target - param_x(1)) / param_x(0);
            xTarget = depth_target;
        }
        else {
            frameTarget = 0.0;
            xTarget = 0.0;
        }
    }

    if (frameTarget > frame_current) {
        //z
        if (dim_poly_z == 3)
            zTarget = param_z(0) * frameTarget * frameTarget + param_z(1) * frameTarget + param_z(2);
        else if (dim_poly_z == 2)
            zTarget = param_z(0) * frameTarget + param_z(1);
        //y
        if (dim_poly_y == 3)
            yTarget = param_y(0) * frameTarget * frameTarget + param_y(1) * frameTarget + param_y(2);
        else if (dim_poly_y == 2)
            yTarget = param_y(0) * frameTarget + param_y(1);
        //rotation vector
        double nx, ny, nz, n_norm;
        // nx
        if (dim_poly_x == 3)
            nx = 2.0 * param_x(0) * frameTarget + param_x(1);
        else if (dim_poly_x == 2)
            nx = param_x(0);
        // ny
        if (dim_poly_y == 3)
            ny = 2.0 * param_y(0) * frameTarget + param_y(1);
        else if (dim_poly_y == 2)
            ny = param_y(0);
        // nz
        if (dim_poly_z == 3)
            nz = 2.0 * param_z(0) * frameTarget + param_z(1);
        else if (dim_poly_z == 2)
            nz = param_z(0);
        n_norm = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (n_norm > 0.0) {//adopt -v for target pose.
            nx = (-1.0) * nx / n_norm;
            ny = (-1.0) * ny / n_norm;
            nz = (-1.0) * nz / n_norm;
        }
        else {
            nx = 0.0;
            ny = 0.0;
            nz = 0.0;
        }
        target = std::vector<double>{ (double)((int)(frameTarget * (double)FPS)), label,xTarget,yTarget, zTarget,nx,ny,nz };
    }
    else
        target.clear();
}



cv::Vec3d Prediction::fitQuadratic(const std::vector<std::vector<double>>& data, int index) {
    int n_seq = data.size();//number of sequence.
    int n = std::min(n_seq, N_POINTS_PREDICT);
    cv::Mat A(n, 3, CV_64F);
    cv::Mat b(n, 1, CV_64F);
    double t, pos;
    int counter = 0;
    for (int i = 1; i < n + 1; i++) {//push data in descending way
        t = data[n_seq - i][0] / (double)FPS;//[frame]->[sec]
        pos = data[n_seq - i][index + 1];//index=1:x, index=2:y, index=3:z.
        A.at<double>(counter, 0) = t * t;
        A.at<double>(counter, 1) = t;
        A.at<double>(counter, 2) = 1.0;
        b.at<double>(counter, 0) = pos;
        counter++;
    }
    cv::Mat coeffs;
    cv::solve(A, b, coeffs, cv::DECOMP_SVD); // Use SVD decomposition for solving
    if (std::abs(coeffs.at<double>(0)) < 1.0e-15 || std::isnan(coeffs.at<double>(0)))
        coeffs.at<double>(0) = 0.0;
    if (std::abs(coeffs.at<double>(1)) < 1.0e-15 || std::isnan(coeffs.at<double>(1)))
        coeffs.at<double>(1) = 0.0;
    if (std::abs(coeffs.at<double>(2)) < 1.0e-15 || std::isnan(coeffs.at<double>(2)))
        coeffs.at<double>(2) = 0.0;
    return cv::Vec3d(coeffs.at<double>(0), coeffs.at<double>(1), coeffs.at<double>(2));//(a,b,c). a*t^2+b*t+c.
}

double Prediction::calculateTargetFrame(const cv::Vec3d& coef, const double& depth_target, double& frame_latest) {
    double frameTarget;//[sec]
    if (std::abs(coef(0)) < 1.0e-6 && std::abs(coef(1)) > 0.0) {//linear
        frameTarget = (double)((depth_target - coef(2)) / coef(1));//[sec]
        if (frameTarget > frame_latest)//sec
            return frameTarget;
        else
            return 0.0;//error.
    }
    else if (std::abs(coef(0)) >= 1.0e-6) {//quadratic
        double ele_root = coef(1) * coef(1) - 4 * coef(0) * (coef(2) - depth_target);//b^2-4ac
        double sol1, sol2, sol_min, sol_max;
        if (ele_root > 0) {
            sol1 = (-coef(1) + ele_root) / (2.0 * coef(0));
            sol2 = (-coef(1) - ele_root) / (2.0 * coef(0));
            sol_min = std::min(sol1, sol2);
            sol_max = std::max(sol1, sol2);
            if (frame_latest < sol_min)
                return sol2;
            else if (frame_latest < sol_max)
                return sol1;
            else
                return 0.0;
        }
        else//no solution.
            return 0.0;
    }
    else//not valid
        return 0.0;
}

double Prediction::calculateTargetFrame_rls(const Eigen::VectorXd& coef, const double& depth_target, double& frame_latest) {
    double frameTarget;
    if (std::abs(coef(0)) < 1.0e-6 && std::abs(coef(1)) > 0.0) {//linear
        frameTarget = (double)((depth_target - coef(2)) / coef(1));
        if (frameTarget > frame_latest)
            return frameTarget;
        else
            return 0.0;//error.
    }
    else if (std::abs(coef(0)) >= 1.0e-6) {//quadratic
        double ele_root = coef(1) * coef(1) - 4 * coef(0) * (coef(2) - depth_target);//b^2-4ac
        double sol1, sol2, sol_min, sol_max;
        if (ele_root > 0) {
            sol1 = (-coef(1) + ele_root) / (2.0 * coef(0));
            sol2 = (-coef(1) - ele_root) / (2.0 * coef(0));
            sol_min = std::min(sol1, sol2);
            sol_max = std::max(sol1, sol2);
            if (frame_latest < sol_min)
                return sol2;
            else if (frame_latest < sol_max)
                return sol1;
            else
                return 0.0;
        }
        else//no solution.
            return 0.0;
    }
    else//not valid
        return 0.0;
}

cv::Vec3d Prediction::linearRegression(int& n_points_predict, std::vector<std::vector<double>>& data)
{
    /**
    * @brief linear regression
    * y = ax + b
    * a = (sigma(xy)-n*mean_x*mean_y)/(sigma(x^2)-n*mean_x^2)
    * b = mean_y - a*mean_x
    *
    * @param[in] n_points_predict number of points to predict trajectory.
    * @param[in] data sequential data. shape is like {sequence, {frame,x,y,z}}
    * @param[in] result_x(std::vector<double>&) : vector for saving result x
    */

    double sumt = 0, sumx = 0, sumtx = 0, sumtt = 0; // for calculating coefficients
    double mean_t, mean_x;
    int length = data.size(); // length of data

    for (int i = 1; i < n_points_predict + 1; i++)
    {
        sumt += data[length - i][0] / (double)FPS;//sec
        sumx += data[length - i][2];
        sumtx += (data[length - i][0] / (double)FPS) * data[length - i][2];
        sumtt += (data[length - i][0] / (double)FPS) * (data[length - i][0] / (double)FPS);
    }
    double slope_x, intercept_x;
    if (std::abs((double)n_points_predict * sumtt - sumt * sumt) > 0)
    {
        slope_x = ((double)n_points_predict * sumtx - sumt * sumx) / std::max(1e-10,((double)n_points_predict * sumtt - sumt * sumt));
        intercept_x = (sumx - slope_x * sumt) / (double)n_points_predict;
    }
    else
    {
        slope_x = 0;
        intercept_x = 0;
    }
    return cv::Vec3d(0.0, slope_x, intercept_x);
    //std::cout << "\n\nX :: The best fit value of curve is : x = " << slope_x << " t + " << intercept_x << ".\n\n"<< std::endl;
}

cv::Vec3d Prediction::linearRegressionY(int& n_points_predict, std::vector<std::vector<double>>& data)
{
    /**
     * @brief linear regression
     * y = ax + b
     * a = (sigma(xy)-n*mean_x*mean_y)/(sigma(x^2)-n*mean_x^2)
     * b = mean_y - a*mean_x
     *
     * @param[in] n_points_predict number of points to predict trajectory.
     * @param[in] data sequential data {sequence, {frame, x,y,z}}
     * @param[in] result_x(std::vector<double>&) : vector for saving result x
     */

    double sumt = 0, sumz = 0, sumtt = 0, sumtz = 0; // for calculating coefficients
    double mean_t, mean_z;
    int length = data.size(); // length of data

    for (int i = 1; i < n_points_predict + 1; i++)
    {
        sumt += data[length - i][0] / (double)FPS;//sec
        sumz += data[length - i][3];
        sumtt += (data[length - i][0] / (double)FPS) * (data[length - i][0] / (double)FPS);
        sumtz += (data[length - i][0] / (double)FPS) * data[length - i][3];
    }
    mean_t = static_cast<double>(sumt) / static_cast<double>(n_points_predict);
    mean_z = static_cast<double>(sumz) / static_cast<double>(n_points_predict);
    double slope_z, intercept_z;
    if (std::abs((double)n_points_predict * sumtt - mean_t * mean_t) > 0.0)
    {
        slope_z = ((double)n_points_predict * sumtz - sumt * sumz) / std::max(1e-10, ((double)n_points_predict * sumtt - sumt * sumt));
        intercept_z = (sumz - slope_z * sumt) / (double)n_points_predict;
    }
    else
    {
        slope_z = 0;
        intercept_z = 0;
    }
    return cv::Vec3d(0.0, slope_z, intercept_z);
}

void Prediction::curveFitting(int& n_points_predict, std::vector<std::vector<double>>& data, std::vector<double>& result)
{
    /**
     * @brief curve fitting with parabora
     * y = a*x^2+b*x+c
     *
     * @param[in] n_points_predict number of points to predict trajectory.
     * @param[in] data sequential data {sequence, (frameIndex,x,y,z)}
     * @param[out] result(std::vector<double>&) : vector for saving result
     */

     // argments analysis
    int length = data.size(); // length of data
    // Initialize sums
    double sumX = 0, sumY = 0, sumX2 = 0, sumX3 = 0, sumX4 = 0;
    double sumXY = 0, sumX2Y = 0;

    for (int i = 1; i < n_points_predict + 1; ++i) {
        double x = data[length - i][0];
        double y = data[length - i][2];
        double x2 = x * x;
        double x3 = x2 * x;
        double x4 = x3 * x;

        sumX += x;
        sumY += y;
        sumX2 += x2;
        sumX3 += x3;
        sumX4 += x4;
        sumXY += x * y;
        sumX2Y += x2 * y;
    }

    // Construct matrices
    Eigen::Matrix3d A;
    Eigen::Vector3d B;

    A << n_points_predict, sumX, sumX2,
        sumX, sumX2, sumX3,
        sumX2, sumX3, sumX4;

    B << sumY,
        sumXY,
        sumX2Y;

    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> dec(A);
    Eigen::Vector3d coeffs = dec.solve(B);
    //Eigen::Vector3d coeffs = A.colPivHouseholderQr().solve(B);
    double a, b, c;
    //std::cout << "dec.rank()=" << dec.rank() << ", dec.info()=" << (dec.info()==Eigen::Success) << std::endl;
    if (dec.rank() < 3 || dec.info() != Eigen::Success) {//check if the matrix is of full rank and calculation is successful
        c = 0;
        b = 0;
        a = 0;
    }
    else {
        c = coeffs[0];
        b = coeffs[1];
        a = coeffs[2];
        if (std::abs(c) < 1e-6) c = 0.0;
        if (std::abs(b) < 1e-6) b = 0.0;
        if (std::abs(a) < 1e-6) a = 0.0;
    }

    result = std::vector<double>{ a, b, c };

    //std::cout << "y = " << a << "x^2 + " << b << "x + " << c << std::endl;
}