#pragma once

#ifndef UTILITY_H
#define UTILITY_H

#include "stdafx.h"
#include "global_parameters.h"

class Utility
{
public:
    Utility()
    {
        std::cout << "construct Utility" << std::endl;
    }

    bool getImagesFromQueueMot(std::array<cv::Mat1b, 2>& imgs, int& frameIndex);
    bool getImagesFromQueueYoloPose(std::array<cv::Mat1b, 2>& imgs, int& frameIndex);
    bool getImagesFromQueueOptflow(std::array<cv::Mat1b, 2>& imgs, int& frameIndex);
    bool getFrameFromQueueRobot(int& frameIndex);

    void checkStorage(std::vector<std::vector<cv::Rect2d>>& posSaverYolo, std::vector<int>& detectedFrame, std::string fileName);

    void checkClassStorage(std::vector<std::vector<int>>& classSaverYolo, std::vector<int>& detectedFrame, std::string fileName);

    void checkStorageTM(std::vector<std::vector<cv::Rect2d>>& posSaverYolo, std::vector<int>& detectedFrame, std::string fileName);

    void checkClassStorageTM(std::vector<std::vector<int>>& classSaverYolo, std::vector<int>& detectedFrame, std::string fileName);

    void checkSeqData(std::vector<std::vector<std::vector<double>>>& dataLeft, std::string fileName);

    void checkKfData(std::vector<std::vector<std::vector<double>>>& dataLeft, std::string fileName);

    void save_params(std::vector<std::vector<std::vector<double>>>& posSaver, const std::string file);

    void saveMatching(std::vector<std::vector<std::vector<int>>>& dataLeft, std::string fileName);

    void save3d_mot(std::vector<std::vector<std::vector<double>>>& posSaver, const std::string file);

    bool getImagesFromQueueTM(std::array<cv::Mat1b, 2>& imgs, int& frameIndex);

    /* read imgs */
    void pushFrame(std::array<cv::Mat1b, 2>& src, const int frameIndex);

    void saveYolo(std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver, const std::string& file);

    void saveYoloMulti(std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver, const std::string& file);

    void save(std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver, const std::string& file);

    void save3d(std::vector<std::vector<std::vector<std::vector<double>>>>& posSaver, const std::string& file);

    void saveMat(std::string fileName, std::vector<std::vector<cv::Mat>>& joints);
    void saveSeqMat(std::string fileName, std::vector<std::vector<cv::Mat>>& joints);
    void saveDeterminant(std::string fileName, std::vector<double>& d);
    void saveData(std::string fileName, std::vector<std::vector<std::vector<double>>>& joints);
    void saveData2(std::string fileName, std::vector<std::vector<double>>& joints);
    void saveData3(std::string fileName, std::vector<std::vector<std::vector<std::vector<double>>>>& joints);
};

#endif

