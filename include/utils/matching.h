#pragma once

#ifndef MATCHING_H
#define MATCHING_H

#include "stdafx.h"
#include "hungarian.h"


class Matching
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
    const double lambda_y = 5.0;
    const double lambda_size = 1.0;
    const double threshold_xdiff_ = 207.0;
    const double threshold_ydiff_ = 100; //max difference between each camera in y axis
    const double threshold_size_ = 9.0;
    //coefficients in x
    const double slope_x = 2.0;
    const double mu_x = 20.0;
    const double penalty_newMatch = 1.1;//penalty for new matching.
    const double Cost_max = 100.0;
    const double Cost_identity = 50.0;//diff_y, diff_size -> consecutive values. diff_x,cost_frame,cost_label->0 or Cost_max
public:
    double Delta_oy; //delta in y coordinate between 2 cams
    int frame_left, frame_right, frame_latest;
    std::vector<int> idx_match_prev;//previous matching result.

    HungarianAlgorithm HungAlgo;

    Matching()
    {
        std::cout << "construct Matching class" << std::endl;
    }

    void main(std::vector<std::vector<std::vector<double>>>& seqData_left, std::vector<std::vector<std::vector<double>>>& seqData_right,
        const double& oY_left, const double& oY_right, std::vector<std::vector<int>>& matching);

    int arrangeData(std::vector<std::vector<std::vector<double>>>& seqData, std::vector<std::vector<double>>& data_ball, std::vector<std::vector<double>>& data_box,
        std::vector<int>& index_ball, std::vector<int>& index_box);

    void sortData(std::vector<std::vector<double>>& data, std::vector<int>& classes);

    void matchingObj(std::vector<std::vector<double>>& ball_left, std::vector<std::vector<double>>& ball_right, std::vector<int>& ball_index_left, std::vector<int>& ball_index_right,
        const float& oY_left, const float& oY_right, std::vector<std::vector<int>>& matching);

    void matchingHung(
        std::vector<std::vector<std::vector<double>>>& seqData_left, std::vector<std::vector<std::vector<double>>>& seqData_right,
        std::vector<std::vector<int>>& matching
    );

    /**
    * @brief calculate differences in x, y and size
    * @param[in] left, right {frame,label,left,top,width,height}
    * @return cv::Point3d x=delta_x, y=delta_y, z=delta_size.
    */
    cv::Point3d compareGeometricFeatures(std::vector<double>& left, std::vector<double>& right);

    //ID check
    double compareID(int label1, int label2, bool bool_frame);
};

#endif 