#pragma once

#ifndef YOLO_BATCH_H
#define YOLO_BATCH_H

#include "../stdafx.h"
#include "../global_parameters.h"

/*  YOLO class definition  */
class YOLODetect_batch
{
private:
    torch::jit::script::Module mdl;
    torch::DeviceType devicetype;
    torch::Device* device;

    std::string yolofilePath;//RTX A5000. yolov10n_320_640_last.torchscript
    int originalWidth, orginalHeight;
    int frameWidth;
    int frameHeight;
    int yoloWidth;
    int yoloHeight;
    cv::Size YOLOSize;
    double IoUThreshold,ConfThreshold,IoUThresholdIdentity; // for maitainig consistency of tracking

    /* initialize function */
    void initializeDevice()
    {
        // set device
        if (torch::cuda::is_available())
        {
            // device = new torch::Device(devicetype, 0);
            device = new torch::Device(torch::kCUDA, 0);
            std::cout << "YOLO detection :: set cuda : " << *device << std::endl;
        }
        else
        {
            device = new torch::Device(torch::kCPU);
            std::cout << "set CPU" << std::endl;
        }
    }

    void loadModel()
    {
        // read param
        mdl = torch::jit::load(yolofilePath, *device);
        mdl.to(*device);
        mdl.eval();
        std::cout << "load model" << std::endl;
    }

public:
    // constructor for YOLODetect
    YOLODetect_batch(int imgWidth=512, int imgHeight=512, int yoloWidth=1024, int yoloHeight=512,
		double IoUThreshold=0.4, double ConfThreshold=0.50, double IoUThresholdIdentity=0.1,
		std::string yolofilePath="yolov10n_512_1024.torchscript")
	: originalWidth(imgWidth), orginalHeight(imgHeight), yoloWidth(yoloWidth), yoloHeight(yoloHeight),
	yolofilePath(yolofilePath), IoUThreshold(IoUThreshold), ConfThreshold(ConfThreshold), IoUThresholdIdentity(IoUThresholdIdentity)
    {
        initializeDevice();
        loadModel();

		//size setting.
		frameWidth = 2*imgWidth;//concatenate two images horizontally
    	frameHeight = imgHeight;
		YOLOSize = cv::Size(yoloWidth, yoloHeight);

        std::cout << "YOLO construtor has finished!" << std::endl;
    };
    ~YOLODetect_batch() { delete device; }; // Deconstructor

    /**
    * @briefinference by YOLO
    *  Args:
    *      frame : img
    *      posSaver : storage for saving detected position
    *      queueYoloTemplate : queue for pushing detected img
    *      queueYoloBbox : queue for pushing detected roi, or if available, get candidate position,
    *      queueClassIndex : queue for pushing detected
    */
    void detect(
        cv::Mat1b& frame, const int frameIndex
    );

    //preprocess img
    void preprocessImg(cv::Mat1b& frame, torch::Tensor& imgTensor);

    /**
    * @brief Get current data before YOLO inference started.
    * First : Compare YOLO detection and TM detection
    * Second : if match : return new templates in the same order with TM
    * Third : if not match : adapt as a new templates and add after TM data
    * Fourth : return all class indexes including -1 (not tracked one) for maintainig data consistency
    */
    void roiSetting(
        std::vector<torch::Tensor>& detectedBoxes, std::vector<int>& labels,
        std::vector<cv::Rect2d>& newRoi_left, std::vector<int>& newClass_left,
        std::vector<cv::Rect2d>& newRoi_right, std::vector<int>& newClass_right
    );

    /**
    * @brief push detect results to a que.
    */
    void convert2Yolo2seq(
        std::vector<cv::Rect2d>& newRoi, std::vector<int>& newClass,
        std::vector<std::vector<cv::Rect2d>>& posSaver, std::vector<std::vector<int>>& classSaver,
        const int& frameIndex, std::vector<int>& detectedFrame, std::vector<int>& detectedFrameClass,
        Yolo2seq& newdata
    );
};

#endif 