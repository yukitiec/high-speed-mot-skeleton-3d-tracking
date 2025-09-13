#pragma once

#ifndef MATCHING_SKELETON_H
#define MATCHING_SKELETON_H

#include "stdafx.h"
#include "hungarian.h"


class Matching_skeleton
{
private:
    const bool debug = false;
    const int dif_threshold = 15; //difference between 2 cams
    const float MAX_ROI_DIF = 2.0; //max roi difference
    const float MIN_ROI_DIF = 0.5;//minimum roi difference
    const bool bool_hungarian = true;
    //hungarian algorithm
    const double epsilon = 1e-5;
    //x
    const double lambda_x = 1.0;
    //coefficients in x
    const double slope_x = 2.0;
    const double mu_x = 20.0;
    const double lambda_y = 1.0;
    const double lambda_size = 1.0;
    const double threshold_ydiff = 100; //max difference between each camera in y axis
    const double penalty_newMatch = 1.1;//penalty for new Matching_skeleton.
    const double Cost_max = 50.0;
    const double Cost_identity = 50.0;//diff_y, diff_size -> consecutive values. diff_x,cost_frame,cost_label->0 or Cost_max
public:
    double Delta_oy; //delta in y coordinate between 2 cams
    int frame_left, frame_right, frame_latest;
    std::vector<int> idx_match_prev;//previous matching result.

    HungarianAlgorithm HungAlgo;

    Matching_skeleton()
    {
        std::cout << "construct Matching class" << std::endl;
    }

    void main(std::vector<std::vector<std::vector<std::vector<int>>>>& seqData_left, std::vector<std::vector<std::vector<std::vector<int>>>>& seqData_right,
        const double& oY_left, const double& oY_right, std::vector<std::vector<int>>& matching);

    void matchingHung(
        std::vector<std::vector<std::vector<std::vector<int>>>>& seqData_left, std::vector<std::vector<std::vector<std::vector<int>>>>& seqData_right,
        std::vector<std::vector<int>>& matching
    );

    void calculate_centers(std::vector<std::vector<std::vector<std::vector<int>>>>& seqData,std::vector<std::vector<double>>& candidates, bool bool_left);

    /**
    * @brief calculate differences in x, y and size
    * @param[in] left, right {frame,label,left,top,width,height}
    * @return cv::Point3d x=delta_x, y=delta_y, z=delta_size.
    */
    cv::Point2d compareGeometricFeatures(std::vector<double>& left, std::vector<double>& right);

    double compareID(int label1, int label2, bool bool_frame);
};

#endif 