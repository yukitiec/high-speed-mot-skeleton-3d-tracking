#pragma once

#ifndef TRIANGULATION_H
#define TRIANGULATION_H

#include "stdafx.h"
#include "matching.h"
#include "utility.h"
#include "global_parameters.h"


class Triangulation
{
private:

    const double BASELINE = 208.0;
    const int numJoint = 6; //number of joints
    //which method to triangulate 3D points
    const int method_triangulate = 1; //0 : DLT, 1: stereo method
    const int numObjects = 100;
    const double threshold_difference_perFrame = 300;//max speed per frame is 300 mm/frame
    const int counter_valid = 3; //num of minimum counter for valid data

    void load_intrinsicData(std::string& file_path, bool bool_left) {
        /**
        * @brief load camera intrinsic and distortion coefficients.
        * @param[in] file file path
        * @param[in] bool_left whether the data is left or right.
        */

        // Open the file
        std::ifstream file(file_path);

        if (!file.is_open()) {
            std::cerr << "Error opening file: " << file_path << std::endl;
        }

        std::string line;
        std::string section;  // To identify whether reading intrinsic or distortion parameters
        int intrinsic_row = 0;  // To keep track of current row index for intrinsic matrix

        while (getline(file, line)) {
            // Skip empty lines
            if (line.empty())
                continue;

            // Check if the line indicates the start of a new section
            if (line.find("intrinsic:") != std::string::npos) {
                section = "intrinsic";
                continue;
            }
            else if (line.find("distortion:") != std::string::npos) {
                section = "distortion";
                continue;
            }

            // Process the line based on the current section
            if (section == "intrinsic") {
                // Read intrinsic parameters
                std::stringstream ss(line);
                if (bool_left) {//left
                    for (int col = 0; col < 3; ++col) {
                        ss >> cameraMatrix_left.at<double>(intrinsic_row, col);
                    }
                }
                else {//right
                    for (int col = 0; col < 3; ++col) {
                        ss >> cameraMatrix_right.at<double>(intrinsic_row, col);
                    }
                }
                intrinsic_row++;
            }
            else if (section == "distortion") {

                // Read distortion parameters
                std::stringstream ss(line);
                if (bool_left) {//left
                    for (int i = 0; i < 5; ++i) {
                        ss >> distCoeffs_left.at<double>(i);
                    }
                }
                else {//right
                    for (int i = 0; i < 5; ++i) {
                        ss >> distCoeffs_right.at<double>(i);
                    }
                }
            }
        }
        // Close the file
        file.close();
    };

    void load_extrinsicData(std::string& file_path, bool bool_left) {
        /**
        * @brief load camera intrinsic and distortion coefficients.
        * @param[in] file file path
        * @param[in] bool_left whether the data is left or right.
        */

        // Open the file
        std::ifstream file(file_path);

        if (!file.is_open()) {
            std::cerr << "Error opening file: " << file_path << std::endl;
        }

        std::string line;
        std::string section;  // To identify whether reading intrinsic or distortion parameters
        int R_row = 0;  // To keep track of current row index for intrinsic matrix
        int T_row = 0;

        while (getline(file, line)) {
            // Skip empty lines
            if (line.empty())
                continue;

            // Check if the line indicates the start of a new section
            if (line.find("R:") != std::string::npos) {
                section = "R";
                continue;
            }
            else if (line.find("T:") != std::string::npos) {
                section = "T";
                continue;
            }

            // Process the line based on the current section
            if (section == "R") {
                // Read intrinsic parameters
                std::stringstream ss(line);
                if (bool_left) {//left
                    for (int col = 0; col < 3; ++col) {
                        ss >> R_left.at<double>(R_row, col);
                    }
                }
                else {//right
                    for (int col = 0; col < 3; ++col) {
                        ss >> R_right.at<double>(R_row, col);
                    }
                }
                R_row++;
            }
            else if (section == "T") {
                // Read distortion parameters
                std::stringstream ss(line);
                if (bool_left) {//left
                    ss >> T_left.at<double>(T_row, 0);
                }
                else {//right
                    ss >> T_right.at<double>(T_row, 0);
                }
                T_row++;
            }
        }
        // Close the file
        file.close();
    };

    cv::Mat_<double> loadMatrixFromCSV(const std::string& filename) {
        std::ifstream file(filename);
        cv::Mat_<double> mat(4, 4); // Create a 4x4 matrix of type double

        if (file.is_open()) {
            std::string line;
            for (int i = 0; i < 4 && std::getline(file, line); ++i) {
                std::stringstream ss(line);
                std::string value;
                for (int j = 0; j < 4 && std::getline(ss, value, ','); ++j) {
                    mat(i, j) = std::stod(value); // Convert string to double and store in the matrix
                }
            }
            file.close();
        }
        else {
            std::cerr << "Error opening file: " << filename << std::endl;
        }

        return mat;
    };

public:
    cv::Mat cameraMatrix_left = (cv::Mat_<double>(3, 3) << 754.66874569, 0, 255.393104, // fx: focal length in x, cx: principal point x
        0, 754.64708568, 335.6848201,                           // fy: focal length in y, cy: principal point y
        0, 0, 1                                // 1: scaling factor
        );
    cv::Mat cameraMatrix_right = (cv::Mat_<double>(3, 3) << 802.62616415, 0, 286.48516862, // fx: focal length in x, cx: principal point x
        0, 802.15806832, 293.54957668,                           // fy: focal length in y, cy: principal point y
        0, 0, 1                                // 1: scaling factor
        );
    cv::Mat distCoeffs_left = (cv::Mat_<double>(1, 5) << -0.00661832, -0.19633213, 0.00759942, -0.01391234, 0.73355661);
    cv::Mat distCoeffs_right = (cv::Mat_<double>(1, 5) << 0.00586444, -0.18180071, 0.00489287, -0.00392576, 1.20394993);
    cv::Mat R_left = (cv::Mat_<double>(3, 3) << 1.0, 0.0, 0.0, // fx: focal length in x, cx: principal point x
        0.0, 1.0, 0.0,                           // fy: focal length in y, cy: principal point y
        0.0, 0.0, 1.0                                // 1: scaling factor
        );
    cv::Mat R_right = (cv::Mat_<double>(3, 3) << 1.0, 0.0, 0.0, // fx: focal length in x, cx: principal point x
        0.0, 1.0, 0.0,                           // fy: focal length in y, cy: principal point y
        0.0, 0.0, 1.0                                // 1: scaling factor
        );
    cv::Mat T_left = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 0.0);
    cv::Mat T_right = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 0.0);
    cv::Mat projectMatrix_left = (cv::Mat_<double>(3, 4) << 375.5, 0, 249.76, 0, // fx: focal length in x, cx: principal point x
        0, 375.5, 231.0285, 0,                           // fy: focal length in y, cy: principal point y
        0, 0, 1, 0                                // 1: scaling factor
        );
    cv::Mat projectMatrix_right = (cv::Mat_<double>(3, 4) << 375.5, 0, 249.76, -280, // fx: focal length in x, cx: principal point x
        0, 375.5, 231.028, 0,                           // fy: focal length in y, cy: principal point y
        0, 0, 1, 0                               // 1: scaling factor
        );

    cv::Mat transform_cam2base = (cv::Mat_<double>(4, 4) << 1.0, 0.0, 0.0, 0.0, // fx: focal length in x, cx: principal point x
        0.0, 1.0, 0.0, 0.0,                           // fy: focal length in y, cy: principal point y
        0.0, 0.0, 1.0, 0.0,                              // 1: scaling factor
        0.0, 0.0, 0.0, 1.0
        );

    const double fX = (cameraMatrix_left.at<double>(0, 0) + cameraMatrix_right.at<double>(0, 0)) / 2;
    const double fY = (cameraMatrix_left.at<double>(1, 1) + cameraMatrix_right.at<double>(1, 1)) / 2;
    const double fSkew = (cameraMatrix_left.at<double>(0, 1) + cameraMatrix_right.at<double>(0, 1)) / 2;
    const double oX_left = cameraMatrix_left.at<double>(0, 2);
    const double oX_right = cameraMatrix_right.at<double>(0, 2);
    const double oY_left = cameraMatrix_left.at<double>(1, 2);
    const double oY_right = cameraMatrix_right.at<double>(1, 2);

    int num_obj_left = 0;
    int num_obj_right = 0;

    std::string file_intrinsic_left;
    std::string file_intrinsic_right;
    std::string file_extrinsic_left;
    std::string file_extrinsic_right;
    std::string file_cam2base;
    std::vector<std::vector<std::vector<std::vector<int>>>> seqData_left, seqData_right; //{n_human,n_seq, n_joints, (frame, left,top,width,height)}

    Triangulation(const std::string& rootDir)
    {
        file_intrinsic_left = rootDir + "/camera0_intrinsics.dat";
        file_intrinsic_right = rootDir + "/camera1_intrinsics.dat";
        file_extrinsic_left = rootDir + "/camera0_rot_trans.dat";
        file_extrinsic_right = rootDir + "/camera1_rot_trans.dat";
        //load data
        load_intrinsicData(file_intrinsic_left, true);//left
        load_intrinsicData(file_intrinsic_right, false);//right
        load_extrinsicData(file_extrinsic_left, true);//left
        load_extrinsicData(file_extrinsic_right, false);//right

        //calculate projection matrix
        //left
        cv::Mat Rt_left = cv::Mat::eye(3, 4, CV_64F);
        R_left.copyTo(Rt_left(cv::Rect(0, 0, 3, 3)));
        T_left.copyTo(Rt_left(cv::Rect(3, 0, 1, 3)));
        projectMatrix_left = cameraMatrix_left * Rt_left;

        //right
        cv::Mat Rt_right = cv::Mat::eye(3, 4, CV_64F);
        R_right.copyTo(Rt_right(cv::Rect(0, 0, 3, 3)));
        T_right.copyTo(Rt_right(cv::Rect(3, 0, 1, 3)));
        projectMatrix_right = cameraMatrix_right * Rt_right;

        //load transform_cam2base from .csv file.
        file_cam2base = rootDir + "/transform_camera2base.csv";
        transform_cam2base = loadMatrixFromCSV(file_cam2base);

        // Display the read parameters (optional)
        std::cout << "Intrinsic parameters (left): " << cameraMatrix_left << std::endl;
        std::cout << "Intrinsic parameters (right):" << cameraMatrix_right << std::endl;
        std::cout << "Distortion parameters (left):" << distCoeffs_left << std::endl;
        std::cout << "Distortion parameters (right):" << distCoeffs_right << std::endl;
        std::cout << "Rotation matrix (left): " << R_left << std::endl;
        std::cout << "Rotation matrix (right):" << R_right << std::endl;
        std::cout << "Translation vector (left):" << T_left << std::endl;
        std::cout << "Translation vector (right):" << T_right << std::endl;
        std::cout << "Projection matrix (left):" << projectMatrix_left << std::endl;
        std::cout << "Projection matrix (right):" << projectMatrix_right << std::endl;
        std::cout << "Transform matrix from camera to base frame:" << transform_cam2base << std::endl;
    };

    bool compareVectors(const std::vector<int>& a, const std::vector<int>& b);

    void sortData(std::vector<std::vector<int>>& data);

    void triangulation(std::vector<std::vector<std::vector<double>>>& data_left, std::vector<std::vector<std::vector<double>>>& data_right, std::vector<std::vector<int>>& matchingIndexes, std::vector<std::vector<std::vector<double>>>& data_3d);

    void cal3D(std::vector<cv::Point2d>& pts_left, std::vector<cv::Point2d>& pts_right, int method_triangulate, std::vector<cv::Point3d>& results);

    void dlt(std::vector<cv::Point2d>& points_left, std::vector<cv::Point2d>& points_right, std::vector<cv::Point3d>& results);

    void stereo3D(std::vector<cv::Point2d>& left, std::vector<cv::Point2d>& right, std::vector<cv::Point3d>& results);

    /**
    * @brief make a directory.
    */
    void makeDir(std::filesystem::path& dirPath);
};

#endif