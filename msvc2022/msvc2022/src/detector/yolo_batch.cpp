#include "../../include/detector/yolo_batch.h"

void YOLODetect_batch::detect(cv::Mat1b& frame, const int frameIndex)
{
    /* inference by YOLO
    *  Args:
    *      frame : img
    *      posSaver : storage for saving detected position
    *      queueYoloTemplate : queue for pushing detected img
    *      queueYoloBbox : queue for pushing detected roi, or if available, get candidate position,
    *      queueClassIndex : queue for pushing detected
    */

    //finish signal
    bool bool_finish = false;

    //PREPROCESS IMAGE
    torch::Tensor imgTensor;
    preprocessImg(frame, imgTensor);

    //GET LATEST DATA and YOLO DETECTION
    std::vector<cv::Rect2d> bboxesCandidateTMLeft, bboxesCandidateTMRight; // for limiting detection area
    std::vector<int> classIndexesTMLeft, classIndexesTMRight;
    torch::Tensor preds;

    /* wrap to disable grad calculation */
    {
        torch::NoGradGuard no_grad;
        preds = mdl.forward({ imgTensor }).toTensor(); // preds shape : [1,300,6]
    }

    //POST PROCESS

    //STEP1 :: divide detections into balls ans boxes
    std::vector<torch::Tensor> rois; //detected rois.(n,6),(m,6) :: including both left and right objects
    std::vector<int> labels;//detected labels.
    //auto start = std::chrono::high_resolution_clock::now();
    torch::Tensor preds_good = preds.select(2, 4) > ConfThreshold; // Extract the high score detections. :: xc is "True" or "False"
    torch::Tensor x0 = preds.index_select(1, torch::nonzero(preds_good[0]).select(1, 0)); // box, x0.shape : (1,n,6) : n: number of candidates
    x0 = x0.squeeze(0); //(1,n,6) -> (n,6) (left,top,right,bottom)
    int size = x0.size(0);//num of detections
    torch::Tensor bbox, pred;
    double left, top, right, bottom;
    int label;

    if (size == 1) {
        pred = x0[0].cpu();
        bbox = pred.slice(0, 0, 4);//x.slice(dim,start,end);
        bbox[0] = std::max(bbox[0].item<double>(), 0.0);//left
        bbox[1] = std::max(bbox[1].item<double>(), 0.0);//top
        bbox[2] = std::min(bbox[2].item<double>(), (double)yoloWidth);//right
        bbox[3] = std::min(bbox[3].item<double>(), (double)yoloHeight);//bottom
        label = pred[5].item<int>();//label
        rois.push_back(bbox);
        labels.push_back(label);
    }
    else if (size >= 2) {
        for (int i = 0; i < size; i++) {
            pred = x0[i].cpu();
            bbox = pred.slice(0, 0, 4);//x.slice(dim,start,end);
            bbox[0] = std::max(bbox[0].item<double>(), 0.0);//left
            bbox[1] = std::max(bbox[1].item<double>(), 0.0);//top
            bbox[2] = std::min(bbox[2].item<double>(), (double)yoloWidth);//right
            bbox[3] = std::min(bbox[3].item<double>(), (double)yoloHeight);//bottom
            label = pred[5].item<int>();//label
            rois.push_back(bbox);
            labels.push_back(label);
        }
    }

    //send detection data to buffer.
    Yolo2buffer yolo2buffer;
    yolo2buffer.rois = rois;
    yolo2buffer.labels = labels;
    yolo2buffer.frame = frame;
    yolo2buffer.frameIndex = frameIndex;
    q_yolo2buffer.push(yolo2buffer);
}

void YOLODetect_batch::preprocessImg(cv::Mat1b& frame, torch::Tensor& imgTensor)
{
    // run
    //std::cout << frame.size() << std::endl;
    cv::Mat yoloimg; // define yolo img type
    cv::resize(frame, yoloimg, YOLOSize);
    cv::cvtColor(yoloimg, yoloimg, cv::COLOR_GRAY2RGB);
    imgTensor = torch::from_blob(yoloimg.data, { yoloimg.rows, yoloimg.cols, 3 }, torch::kByte); // vector to tensor
    imgTensor = imgTensor.permute({ 2, 0, 1 });                                                  // Convert shape from (H,W,C) -> (C,H,W)
    imgTensor = imgTensor.toType(torch::kFloat);                                               // convert to float type
    imgTensor = imgTensor.div(255);                                                            // normalization
    imgTensor = imgTensor.unsqueeze(0);                                                        // expand dims for Convolutional layer (height,width,1)
    imgTensor = imgTensor.to(*device);                                                         // transport data to GPU
}

void YOLODetect_batch::roiSetting(
    std::vector<torch::Tensor>& detectedBoxes, std::vector<int>& labels,
    std::vector<cv::Rect2d>& newRoi_left, std::vector<int>& newClass_left,
    std::vector<cv::Rect2d>& newRoi_right, std::vector<int>& newClass_right
)
{
    // std::cout << "bboxesYolo size=" << detectedBoxes.size() << std::endl;
    /* detected by Yolo */
    if (!detectedBoxes.empty())
    {
        //std::cout << "No TM tracker exist " << std::endl;
        int numBboxes = detectedBoxes.size(); // num of detection
        int left, top, right, bottom;         // score0 : ball , score1 : box
        cv::Rect2d roi;

        /* convert torch::Tensor to cv::Rect2d */
        std::vector<cv::Rect2d> bboxesYolo_left, bboxesYolo_right;
        bboxesYolo_left.reserve(25);
        bboxesYolo_right.reserve(25);
        for (int i = 0; i < numBboxes; ++i)
        {
            float expandrate[2] = { static_cast<float>(frameWidth) / static_cast<float>(yoloWidth), static_cast<float>(frameHeight) / static_cast<float>(yoloHeight) }; // resize bbox to fit original img size
            // std::cout << "expandRate :" << expandrate[0] << "," << expandrate[1] << std::endl;
            left = static_cast<int>(detectedBoxes[i][0].item().toFloat() * expandrate[0]);
            top = static_cast<int>(detectedBoxes[i][1].item().toFloat() * expandrate[1]);
            right = static_cast<int>(detectedBoxes[i][2].item().toFloat() * expandrate[0]);
            bottom = static_cast<int>(detectedBoxes[i][3].item().toFloat() * expandrate[1]);
            //left
            if (right <= originalWidth)
            {
                newRoi_left.emplace_back(left, top, (right - left), (bottom - top));
                newClass_left.push_back(labels[i]);
            }
            //right
            else if (left >= originalWidth)
            {
                newRoi_right.emplace_back(left - originalWidth, top, (right - left), (bottom - top));
                newClass_right.push_back(labels[i]);
            }
        }
    }
    /* No object detected in Yolo -> return -1 class label */
    else
    {
        /* nothing to do */
    }
}

void YOLODetect_batch::convert2Yolo2seq(
    std::vector<cv::Rect2d>& newRoi, std::vector<int>& newClass,
    std::vector<std::vector<cv::Rect2d>>& posSaver, std::vector<std::vector<int>>& classSaver,
    const int& frameIndex, std::vector<int>& detectedFrame, std::vector<int>& detectedFrameClass,
    Yolo2seq& newdata
)
{
    /*
     * push detection data to queueLeft
     */

     /* update data */
     /* detection is successful */
    if (!newRoi.empty())
    {
        // std::cout << "detection succeeded" << std::endl;
        // save detected data
        posSaver.push_back(newRoi);
        classSaver.push_back(newClass);
        detectedFrame.push_back(frameIndex);
        detectedFrameClass.push_back(frameIndex);

        /* finish initialization */
        newdata.bbox = newRoi;
        newdata.classIndex = newClass;
        newdata.frame = frameIndex;
    }
    /* no object detected -> return class label -1 if TM tracker exists */
    else
    {
        //nothing to do.
    }
}