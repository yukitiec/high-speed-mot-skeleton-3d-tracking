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

	//Make queues for each module to avoid data race.
    static bool getImagesFromQueueMot(std::array<cv::Mat1b, 2>& imgs, int& frameIndex);
    static bool getImagesFromQueueYoloPose(std::array<cv::Mat1b, 2>& imgs, int& frameIndex);
    static bool getImagesFromQueueOptflow(std::array<cv::Mat1b, 2>& imgs, int& frameIndex);
    static bool getFrameFromQueueRobot(int& frameIndex);

    static void checkStorage(std::vector<std::vector<cv::Rect2d>>& posSaverYolo, std::vector<int>& detectedFrame, std::string fileName);

    static void checkClassStorage(std::vector<std::vector<int>>& classSaverYolo, std::vector<int>& detectedFrame, std::string fileName);

    static void checkStorageTM(std::vector<std::vector<cv::Rect2d>>& posSaverYolo, std::vector<int>& detectedFrame, std::string fileName);

    static void checkClassStorageTM(std::vector<std::vector<int>>& classSaverYolo, std::vector<int>& detectedFrame, std::string fileName);

    static void checkSeqData(std::vector<std::vector<std::vector<double>>>& dataLeft, std::string fileName);

    static void checkKfData(std::vector<std::vector<std::vector<double>>>& dataLeft, std::string fileName);

    static void save_params(std::vector<std::vector<std::vector<double>>>& posSaver, const std::string file);

    static void saveMatching(std::vector<std::vector<std::vector<int>>>& dataLeft, std::string fileName);

    static void save3d_mot(std::vector<std::vector<std::vector<double>>>& posSaver, const std::string file);

    bool getImagesFromQueueTM(std::array<cv::Mat1b, 2>& imgs, int& frameIndex);

    /* read imgs */
    static void pushFrame(std::array<cv::Mat1b, 2>& src, const int frameIndex);

    static void saveYolo(std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver, const std::string& file);

    static void saveYoloMulti(std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver, const std::string& file);

    static void save(std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver, const std::string& file);

    static void save3d(std::vector<std::vector<std::vector<std::vector<double>>>>& posSaver, const std::string& file);

    static void saveMat(std::string fileName, std::vector<std::vector<cv::Mat>>& joints);
    static void saveSeqMat(std::string fileName, std::vector<std::vector<cv::Mat>>& joints);
    static void saveDeterminant(std::string fileName, std::vector<double>& d);
    static void saveData(std::string fileName, std::vector<std::vector<std::vector<double>>>& joints);
    static void saveData2(std::string fileName, std::vector<std::vector<double>>& joints);
    static void saveData3(std::string fileName, std::vector<std::vector<std::vector<std::vector<double>>>>& joints);
};

#endif

