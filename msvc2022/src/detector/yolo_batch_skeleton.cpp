#include "stdafx.h"
#include "yolo_batch_skeleton.h"

void YOLOPoseBatch::detect(cv::Mat1b& frame, int& frameIndex, int& counter,
    std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver_left, std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver_right)
{
    /* inference by YOLO
     *  Args:
     *      frame : img
     *      posSaver : storage for saving detected position
     *      queueYoloTemplate : queue for pushing detected img
     *      queueYoloBbox : queue for pushing detected roi, or if available, get candidate position,
     *      queueClassIndex : queue for pushing detected
     */

     /* preprocess img */
    torch::Tensor imgTensor;
    preprocessImg(frame, imgTensor);

    /* inference */
    torch::Tensor preds;

    std::vector<cv::Rect2i> roiLatest;
    {
        torch::NoGradGuard no_grad;
        preds = mdl.forward({ imgTensor }).toTensor(); // preds shape : [1,6,2100]
    }
    // std::cout << "finish inference" << std::endl;
    preds = preds.permute({ 0, 2, 1 }); // change order : (1,56,2100) -> (1,2100,56)

    //torch::Tensor xc = preds.select(2, 4) > ConfThreshold_human;                       // get dimenseion 2, and 5th element of prediction
    //torch::Tensor x = preds.index_select(1, torch::nonzero(xc[0]).select(1, 0)); // x0.shape : (1,n,6) : n: number of candidates
    //x = x.index_select(1, x.select(2, 4).argsort(1, true).squeeze());                 // sorted in descending order
    //x = x.squeeze(0);                                                                  //(1,n,56) -> (n,56) : whole human{(xCenter,yCenter,width,height,confidence)}+17 joints*(xCenter,yCenter,confidence)
    //x=x.cpu();//send data from GPU to CPU
    std::vector<torch::Tensor> detectedBoxesHuman; //(n,56)
    //Yolo2Buffer_skeleton data;
    //data.preds = x;
    ////data.detectedBoxesHuman = detectedBoxesHuman;
    //data.frame = frame;
    //data.frameIndex = frameIndex;
    //q_yolo2buffer_skeleton.push(data);

    /*detect human */
    nonMaxSuppressionHuman(preds, detectedBoxesHuman, ConfThreshold_human, IoUThreshold);

    if (!detectedBoxesHuman.empty()) {
        Yolo2Buffer_skeleton data;
        data.detectedBoxesHuman = detectedBoxesHuman;
        data.frame = frame;
        data.frameIndex = frameIndex;
        q_yolo2buffer_skeleton.push(data);
    }

    ////std::cout << "detectedBboxesHuman size=" << detectedBoxesHuman.size() << std::endl;
    ///* get keypoints from detectedBboxesHuman -> shoulder,elbow,wrist */
    //std::vector<std::vector<std::vector<int>>> keyPoints; // vector for storing keypoints
    //std::vector<int> humanPos; //whether human is in left or right
    ///* if human detected, extract keypoints */
    //if (!detectedBoxesHuman.empty())
    //{
    //    //std::cout << "Human detected!" << std::endl;
    //    keyPointsExtractor(detectedBoxesHuman, keyPoints, humanPos, ConfThreshold_joint);
    //    //std::cout << "finish keypointsextractor" << std::endl;
    //    /*push updated data to queue*/
    //    push2Queue(frame, frameIndex, keyPoints, roiLatest, humanPos, posSaver_left, posSaver_right);
    //    //std::cout << "frame size:" << frame.cols << "," << frame.rows << std::endl;
    //    /* draw keypoints in the frame */
    //    //drawCircle(frame, keyPoints, counter);
    //}
}

void YOLOPoseBatch::preprocessImg(cv::Mat1b& frame, torch::Tensor& imgTensor)
{
    // run
    cv::Mat yoloimg; // define yolo img type
    //cv::imwrite("input.jpg", frame);
    cv::cvtColor(frame, yoloimg, cv::COLOR_GRAY2RGB);
    cv::resize(yoloimg, yoloimg, YOLOSize);
    //cv::imwrite("yoloimg.jpg", yoloimg);
    //std::cout << "yoloImg.height" << yoloimg.rows << ", yoloimg.width" << yoloimg.cols << std::endl;
    imgTensor = torch::from_blob(yoloimg.data, { yoloimg.rows, yoloimg.cols, 3 }, torch::kByte); // vector to tensor
    imgTensor = imgTensor.permute({ 2, 0, 1 });                                                  // Convert shape from (H,W,C) -> (C,H,W)
    imgTensor = imgTensor.toType(torch::kFloat);                                               // convert to float type
    imgTensor = imgTensor.div(255);                                                            // normalization
    imgTensor = imgTensor.unsqueeze(0);                                                        //(1,3,320,320)
    imgTensor = imgTensor.to(*device);                                                         // transport data to GPU
}

void YOLOPoseBatch::nonMaxSuppressionHuman(torch::Tensor& prediction, std::vector<torch::Tensor>& detectedBoxesHuman, float confThreshold, float iouThreshold)
{
    /* non max suppression : remove overlapped bbox
     * Args:
     *   prediction : (1,2100,,6)
     * Return:
     *   detectedbox0,detectedboxs1 : (n,6), (m,6), number of candidate
     */

    torch::Tensor xc = prediction.select(2, 4) > confThreshold;                       // get dimenseion 2, and 5th element of prediction
    torch::Tensor x = prediction.index_select(1, torch::nonzero(xc[0]).select(1, 0)); // x0.shape : (1,n,6) : n: number of candidates
    x = x.index_select(1, x.select(2, 4).argsort(1, true).squeeze());                 // sorted in descending order
    x = x.squeeze(0);                                                                  //(1,n,56) -> (n,56) : whole human{(xCenter,yCenter,width,height,confidence)}+17 joints*(xCenter,yCenter,confidence)
    //std::cout << "x=" << x << std::endl;
    bool boolLeft = false;
    bool boolRight = false;
    if (x.size(0) >= 1)
    {
        if (x.size(0) == 1)//number of detected human is 1.
        {
            // std::cout << "top defined" << std::endl;
            detectedBoxesHuman.push_back(x[0].cpu());
        }
        else//number of detected humans is more than 2 or 2.
        {
            //first person
            if (x[0][0].item<int>() <= boundary_img)//xCenter is under boundary_img
            {
                detectedBoxesHuman.push_back(x[0].cpu());
                boolLeft = true;
            }
            else if (x[0][0].item<int>() > boundary_img)//xCenter is over boundary_img
            {
                detectedBoxesHuman.push_back(x[0].cpu());
                boolRight = true;
            }

            // for every candidates
            // if adopt many humans, validate here
            if (x.size(0) >= 2)
            {
                nms(x, detectedBoxesHuman, iouThreshold, boolLeft, boolRight); // exclude overlapped bbox : 20 milliseconds
            }
        }
    }
}

torch::Tensor YOLOPoseBatch::xywh2xyxy(torch::Tensor x)
{
    torch::Tensor y = x.clone();
    y[0] = x[0] - x[2] / 2.0; // left
    y[1] = x[1] - x[3] / 2.0; // top
    y[2] = x[0] + x[2] / 2.0; // right
    y[3] = x[1] + x[3] / 2.0; // bottom
    return y;
}

void YOLOPoseBatch::nms(torch::Tensor& x, std::vector<torch::Tensor>& detectedBoxes, float& iouThreshold, bool& boolLeft, bool& boolRight)
{
    /* calculate IoU for excluding overlapped bboxes
     *
     * bbox1,bbox2 : [left,top,right,bottom,score0,score1]
     *
     */

    int numBoxes = x.size(0);
    torch::Tensor box, box_candidate;
    int counter = 0;
    // there are some overlap between two bbox
    for (int i = 1; i < numBoxes; i++)
    {
        if (bool_oneHuman) {
            if (boolLeft && boolRight) break;

            //detect only 1 human in each image
            if ((x[i][0].item<int>() <= boundary_img && !boolLeft) || (x[i][0].item<int>() > boundary_img && !boolRight))
            {
                box = xywh2xyxy(x[i].slice(0, 0, 4)); //(xCenter,yCenter,width,height) -> (left,top,right,bottom)

                bool addBox = true; // if save bbox as a new detection

                for (torch::Tensor& savedBox : detectedBoxes)
                {
                    box_candidate = xywh2xyxy(savedBox.slice(0, 0, 4));
                    float iou = calculateIoU(box, box_candidate); // calculate IoU
                    /* same bbox : already found -> not add */
                    if (iou > iouThreshold)
                    {
                        addBox = false;
                        break; // next iteration
                    }
                }
                /* new tracker */
                if (addBox)
                {
                    detectedBoxes.push_back(x[i].cpu());
                    if (x[i][0].item<int>() <= boundary_img && !boolLeft) boolLeft = true;
                    if (x[i][0].item<int>() > boundary_img && !boolRight) boolRight = true;
                }
            }
        }
        else {//muliple human tracking
            box = xywh2xyxy(x[i].slice(0, 0, 4)); //(xCenter,yCenter,width,height) -> (left,top,right,bottom)

            bool addBox = true; // if save bbox as a new detection

            for (torch::Tensor& savedBox : detectedBoxes)
            {
                box_candidate = xywh2xyxy(savedBox.slice(0, 0, 4));
                float iou = calculateIoU(box, box_candidate); //calculate IoU
                /* same bbox : already found -> not add */
                if (iou > iouThreshold)
                {
                    addBox = false;
                    break; // next iteration
                }
            }
            /* new tracker */
            if (addBox)
            {
                detectedBoxes.push_back(x[i].cpu());
            }
        }
    }
}

float YOLOPoseBatch::calculateIoU(const torch::Tensor& box1, const torch::Tensor& box2)
{
    float left = std::max(box1[0].item<float>(), box2[0].item<float>());
    float top = std::max(box1[1].item<float>(), box2[1].item<float>());
    float right = std::min(box1[2].item<float>(), box2[2].item<float>());
    float bottom = std::min(box1[3].item<float>(), box2[3].item<float>());

    if (left < right && top < bottom)
    {
        float intersection = (right - left) * (bottom - top);
        float area1 = ((box1[2] - box1[0]) * (box1[3] - box1[1])).item<float>();
        float area2 = ((box2[2] - box2[0]) * (box2[3] - box2[1])).item<float>();
        float unionArea = area1 + area2 - intersection;

        return intersection / unionArea;
    }

    return 0.0f; // No overlap
}

void YOLOPoseBatch::keyPointsExtractor(std::vector<torch::Tensor>& detectedBboxesHuman, std::vector<std::vector<std::vector<int>>>& keyPoints, std::vector<int>& humanPos, const int& ConfThreshold)
{
    int numDetections = detectedBboxesHuman.size();
    bool boolLeft = false;

    //reset index.
    index_candidates_left_.clear();
    index_candidates_right_.clear();
    human_current_left.clear();
    human_current_right.clear();
    //////////////

    //extract keypoints.
    for (int i = 0; i < numDetections; i++)//for all humans
    {
        std::vector<std::vector<int>> keyPointsTemp;
        for (int j = 5; j < 11; j++)//for all joints
        {
            if (detectedBboxesHuman[i][3 * j + 7].item<float>() > ConfThreshold)//keypoints' criteria
            {
                //left
                if ((static_cast<int>(((double)frameWidth / (double)yoloWidth) * detectedBboxesHuman[i][0].item<double>()) < originalWidth))//center position is less than original width
                {
                    boolLeft = true; //left person
                    keyPointsTemp.push_back({ static_cast<int>(((double)frameWidth / (double)yoloWidth) * detectedBboxesHuman[i][3 * j + 7 - 2].item<double>()), static_cast<int>(((double)frameHeight / (double)yoloHeight) * detectedBboxesHuman[i][3 * j + 7 - 1].item<double>()) }); /*(xCenter,yCenter)*/
                }
                //right
                else
                {
                    boolLeft = false;
                    keyPointsTemp.push_back({ static_cast<int>(((double)frameWidth / (double)yoloWidth) * detectedBboxesHuman[i][3 * j + 7 - 2].item<double>() - (double)originalWidth), static_cast<int>(((double)frameHeight / (double)yoloHeight) * detectedBboxesHuman[i][3 * j + 7 - 1].item<double>()) }); /*(xCenter,yCenter)*/
                }
            }
            else
            {
                keyPointsTemp.push_back({ -1, -1 });
            }
        }

        keyPoints.push_back(keyPointsTemp);
        if (boolLeft) {
            humanPos.push_back(LEFT);
            index_candidates_left_.push_back(i);//add index for matching the order.
            //update current human list of the left image.
            human_current_left.push_back({ (((double)frameWidth / (double)yoloWidth) * detectedBboxesHuman[i][0].item<double>()),(((double)frameHeight / (double)yoloHeight) * detectedBboxesHuman[i][1].item<double>()),(((double)frameWidth / (double)yoloWidth) * detectedBboxesHuman[i][2].item<double>()),(((double)frameHeight / (double)yoloHeight) * detectedBboxesHuman[i][3].item<double>()) });
            counter_yolo_left++;
        }
        else {
            humanPos.push_back(RIGHT);
            index_candidates_right_.push_back(i);
            //update current human list of the right image
            human_current_right.push_back({ (((double)frameWidth / (double)yoloWidth) * detectedBboxesHuman[i][0].item<double>() - (double)originalWidth),(((double)frameHeight / (double)yoloHeight) * detectedBboxesHuman[i][1].item<double>()),(((double)frameWidth / (double)yoloWidth) * detectedBboxesHuman[i][2].item<double>()),(((double)frameHeight / (double)yoloHeight) * detectedBboxesHuman[i][3].item<double>()) });
            counter_yolo_right++;
        }
    }

    if (!bool_oneHuman) {//multiple human tracking

        //LEFT -> matching human by comparing the current and previous human position.
        //reset idx_match
        idx_match_left.clear();
        index_delete_send_left_.clear();
        if (!human_previous_left.empty()) {
            matching(human_previous_left, human_current_left, idx_match_left, counter_notUpdate_left_, index_delete_send_left_);
            //std::cout << "YOLO (LEFT) :: index_delete_send_left_.size()=" << index_delete_send_left_.size() << ", human_previous_left.size()=" << human_previous_left.size() << std::endl;
            //if (!index_delete_send_left_.empty()) {
            //    std::cout << "index_delete_send_left_=";
            //    for (int& idx : index_delete_send_left_)
            //        std::cout << idx << ",";
            //    std::cout << std::endl;
            //}
        }
        else {
            if (!human_current_left.empty()) {//first time
                int idx = 0;
                for (std::vector<double>& newHuman : human_current_left) {//add new human
                    human_previous_left.push_back(newHuman);
                    counter_notUpdate_left_.push_back(0);
                    idx_match_left.push_back(idx);
                    idx++;
                }
            }
        }

        //RIGHT
        //reset idx_match
        idx_match_right.clear();
        index_delete_send_right_.clear();
        if (!human_previous_right.empty()) {
            matching(human_previous_right, human_current_right, idx_match_right, counter_notUpdate_right_, index_delete_send_right_);
            //std::cout << "YOLO (RIGHT) :: index_delete_send_left_.size()=" << index_delete_send_right_.size() << ", human_previous_left.size()=" << human_previous_right.size() << std::endl;
            //if (!index_delete_send_right_.empty()) {
            //    std::cout << "index_delete_send_left_=";
            //    for (int& idx : index_delete_send_right_)
            //        std::cout << idx << ",";
            //    std::cout << std::endl;
            //}
        }
        else {
            if (!human_current_right.empty()) {//first time
                int idx = 0;
                for (std::vector<double>& newHuman : human_current_right) {
                    human_previous_right.push_back(newHuman);
                    counter_notUpdate_right_.push_back(0);
                    idx_match_right.push_back(idx);
                    idx++;
                }
            }
        }
    }
}

void YOLOPoseBatch::matching(std::vector<std::vector<double>>& candidates, std::vector<std::vector<double>>& newData, std::vector<int>& idx_match, std::vector<int>& counter_notUpdate, std::vector<int>& index_delete_send)
{
    std::vector<std::vector<double>> costMatrix;
    double distance, aspect, area;
    double cost_pos, cost_size, cost_total;
    cv::Rect2d roi_prev;
    if (bool_debug)
        std::cout << "matching in YOLO" << std::endl;

    if (!newData.empty()) {//new Human was found
        //std::cout << "newData.size()=" << newData.size() << ", candidates.size()=" << candidates.size() <<", counter_notUpdate.size()="<<counter_notUpdate.size() << std::endl;
        std::vector<int> idx_original;
        for (std::vector<double>& prev : candidates) {//for each candidate {0:xCenter,1:yCenter,2:width,3:height}
            std::vector<double> cost_row;
            int counter = 0;
            for (std::vector<double>& current : newData) {//for each new human
                //position
                distance = std::sqrt(std::pow(prev[0] - current[0], 2.0) + std::pow(prev[1] - current[1], 2.0));
                //area
                area = std::max(((prev[2] * prev[3]) / (current[2] * current[3])), ((current[2] * current[3]) / (prev[2] * prev[3])));
                //total
                cost_total = lambda_dist_ * std::min(distance / dist_thresh_, 1.0) + lambda_area_ * std::min((area - 1.0) / (area_thresh_ - 1.0), 1.0);
                if (distance > dist_thresh_)
                    cost_total = Cost_max_;
                //append
                cost_row.push_back(cost_total);

                //add index
                if (costMatrix.empty()) {//only if the order is empty
                    idx_original.push_back(counter);
                    counter++;
                }
            }
            costMatrix.push_back(cost_row);
        }
        //std::cout << "1" << std::endl;
        //match tracker based on hungarian algorithm. Update previousData.
        if (!costMatrix.empty()) {
            std::vector<int> assignment;
            std::vector<int> idx_delete;
            double cost = HungAlgo.Solve(costMatrix, assignment);
            cv::Rect2d roi_match;
            double frame_prev;

            //assign data according to indexList_tm, indexList_yolo and assign
            for (unsigned int x = 0; x < assignment.size(); x++) {//for each candidate from 0 to the end of the candidates. ascending way.
                int index_match = assignment[x];
                if (index_match >= 0 && costMatrix[x][index_match] < Cost_max_) {//matching tracker is found.
                    //matching index
                    idx_match.push_back(index_match);
                    //index to delete in newData
                    idx_delete.push_back(index_match);//In new Detections, index to delete.
                    //update previous data & counter
                    candidates[x] = newData[index_match];//update with newData.
                    counter_notUpdate[x] = 0;
                }
                else {//matching tracker hasn't been found. Keep current position.
                    //matching index isn't valid -> -1
                    idx_match.push_back(-1);
                    //increment counter for not updating.
                    counter_notUpdate[x]++;
                    //check the counter for judging whehter we delete the data.
                    if (counter_notUpdate[x] >= COUNTER_LOST_HUMAN) {//over lifespan
                        index_delete_send.push_back(x);
                    }
                }
            }
            //std::cout << "2" << std::endl;
            //new human
            if (!idx_delete.empty()) {//data to delete exists
                std::sort(idx_delete.rbegin(), idx_delete.rend());//reverse the order for maintaining the data consistency.
                //std::cout << "newData.size()="<<newData.size() <<", idx_original.size()="<<idx_original.size() << ",idx_delete=";
                for (int& idx : idx_delete) {//erase data
                    //std::cout << idx << ",";
                    newData.erase(newData.begin() + idx);
                    idx_original.erase(idx_original.begin() + idx);
                }
                //std::cout << std::endl;
            }
            //std::cout << "3" << std::endl;
            //add new human.
            if (!newData.empty()) {
                int counter = 0;
                for (std::vector<double>& newHuman : newData) {
                    candidates.push_back(newHuman);
                    idx_match.push_back(idx_original[counter]);
                    counter_notUpdate.push_back(0);
                    counter++;
                }
            }

            //std::cout << "4" << std::endl;
            //delete lost human
            if (!index_delete_send.empty()) {
                //reverse the order
                std::sort(index_delete_send.rbegin(), index_delete_send.rend());

                //delete the data.
                for (int& idx : index_delete_send) {
                    candidates.erase(candidates.begin() + idx);
                    idx_match.erase(idx_match.begin() + idx);
                    counter_notUpdate.erase(counter_notUpdate.begin() + idx);
                }
            }
            //std::cout << "5" << std::endl;
        }
    }
    else {//new human wasn't found
        //std::cout << "6" << std::endl;
        for (int x = 0; x < candidates.size(); x++) {//for each previous human.
            idx_match.push_back(-1);
            counter_notUpdate[x]++;
            if (counter_notUpdate[x] >= COUNTER_LOST) {//over lifespan
                index_delete_send.push_back(x);
            }
        }
        //std::cout << "7" << std::endl;
        //delete lost human
        if (!index_delete_send.empty()) {
            //reverse the order
            std::sort(index_delete_send.rbegin(), index_delete_send.rend());

            //delete the data.
            for (int& idx : index_delete_send) {
                candidates.erase(candidates.begin() + idx);
                idx_match.erase(idx_match.begin() + idx);
                counter_notUpdate.erase(counter_notUpdate.begin() + idx);
            }
        }
    }
}

void YOLOPoseBatch::drawCircle(cv::Mat1b& frame, std::vector<std::vector<std::vector<int>>>& ROI, int& counter)
{
    /*number of detections */
    for (int k = 0; k < ROI.size(); k++)
    {
        /*for all joints */
        for (int i = 0; i < ROI[k].size(); i++)
        {
            if (ROI[k][i][0] != -1)
            {
                cv::circle(frame, cv::Point(ROI[k][i][0], ROI[k][i][1]), 5, cv::Scalar(125), -1);
            }
        }
    }
    std::string save_path = std::to_string(counter) + ".jpg";
    cv::imwrite(save_path, frame);
}

void YOLOPoseBatch::push2Queue(cv::Mat1b& frame, int& frameIndex, std::vector<std::vector<std::vector<int>>>& keyPoints,
    std::vector<cv::Rect2i>& roiLatest, std::vector<int>& humanPos,
    std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver_left, std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver_right)
{
    /* check roi Latest
    * if tracking was successful -> update and
    * else : update roi and imgSearch and calculate features. push data to queue
    */
    if (bool_oneHuman) {//detect only one human per time.
        bool bool_left;//left image or right image
        if (!keyPoints.empty())
        {
            std::vector<std::vector<cv::Rect2i>> humanJoints_left, humanJoints_right; // for every human
            std::vector<std::vector<cv::Mat1b>> imgHuman_left, imgHuman_right;
            std::vector<std::vector<std::vector<int>>> humanJointsCenter_left, humanJointsCenter_right;
            //left
            for (int i = 0; i < keyPoints.size(); i++)
            {
                std::vector<cv::Rect2i> joints; // for every joint
                std::vector<cv::Mat1b> imgJoint;
                std::vector<std::vector<int>> jointsCenter;
                //left person
                if (humanPos[i] == LEFT)
                {
                    /* for every joints : (ls,rs,le,re,lw,rw) */
                    if (bool_dynamic_roi) //dynamic roi
                    {
                        bool_left = true;
                        std::vector<float> default_distances(num_joints, max_half_diagonal);
                        std::vector<std::vector<float>>  distances(num_joints, std::vector<float>(num_joints, 10000.0));
                        organizeRoi(frame, frameIndex, bool_left, keyPoints[i], distances, joints, imgJoint, jointsCenter);
                        humanJoints_left.push_back(joints);
                        if (!imgJoint.empty())
                        {
                            imgHuman_left.push_back(imgJoint);
                        }
                        humanJointsCenter_left.push_back(jointsCenter);
                    }
                    else if (!bool_dynamic_roi) //static roi
                    {
                        for (int j = 0; j < keyPoints[i].size(); j++)
                        {
                            organize_left(frame, frameIndex, keyPoints[i][j], joints, imgJoint, jointsCenter);
                        }
                        humanJoints_left.push_back(joints);
                        if (!imgJoint.empty())
                        {
                            imgHuman_left.push_back(imgJoint);
                        }
                        humanJointsCenter_left.push_back(jointsCenter);
                    }
                }
                //right person
                else
                {
                    if (bool_dynamic_roi) //dynamic roi
                    {
                        bool_left = false;
                        std::vector<float> default_distances(num_joints, max_half_diagonal);
                        std::vector<std::vector<float>>  distances(num_joints, default_distances);
                        organizeRoi(frame, frameIndex, bool_left, keyPoints[i], distances, joints, imgJoint, jointsCenter);
                        humanJoints_right.push_back(joints);
                        if (!imgJoint.empty())
                        {
                            imgHuman_right.push_back(imgJoint);
                        }
                        humanJointsCenter_right.push_back(jointsCenter);
                    }
                    //std::cout << "right" << std::endl;
                    /* for every joints */
                    if (!bool_dynamic_roi)
                    {
                        for (int j = 0; j < keyPoints[i].size(); j++)
                        {
                            organize_right(frame, frameIndex, keyPoints[i][j], joints, imgJoint, jointsCenter);
                        }
                        humanJoints_right.push_back(joints);
                        if (!imgJoint.empty())
                        {
                            imgHuman_right.push_back(imgJoint);
                        }
                        humanJointsCenter_right.push_back(jointsCenter);
                    }
                }
            }

            //pop before push
            Yolo2optflow left, right;
            left.roi = humanJoints_left;
            right.roi = humanJoints_right;
            if (!imgHuman_left.empty()) left.img_search = imgHuman_left;
            if (!imgHuman_right.empty()) right.img_search = imgHuman_right;

            //push data
            {
                std::unique_lock<std::mutex> lock_left(mtxYolo_left);
                std::unique_lock<std::mutex> lock_right(mtxYolo_right);
                if (!q_yolo2optflow_left.empty()) q_yolo2optflow_left.pop();
                if (!q_yolo2optflow_right.empty()) q_yolo2optflow_right.pop();
                //push and save data
                q_yolo2optflow_left.push(left);
                q_yolo2optflow_right.push(right);
            }

            posSaver_left.push_back(humanJointsCenter_left);
            posSaver_right.push_back(humanJointsCenter_right);
        }
    }
    else {//detect multiple humans per time
        std::vector<std::vector<cv::Rect2i>> humanJoints_left, humanJoints_right; // for every human
        std::vector<std::vector<cv::Mat1b>> imgHuman_left, imgHuman_right;
        std::vector<std::vector<std::vector<int>>> humanJointsCenter_left, humanJointsCenter_right;

        //LEFT
        if (!index_candidates_left_.empty())//human in the left image is found.
        {
            bool bool_left = true;
            for (int j = 0; j < idx_match_left.size(); j++) {//for each human 
                if (idx_match_left[j] != -1) {//found new data.
                    std::vector<cv::Rect2i> joints; // for every joint
                    std::vector<cv::Mat1b> imgJoint;
                    std::vector<std::vector<int>> jointsCenter;

                    /* for every joint : (ls,rs,le,re,lw,rw) */
                    if (bool_dynamic_roi) //dynamic roi
                    {
                        std::vector<float> default_distances(num_joints, max_half_diagonal);
                        std::vector<std::vector<float>>  distances(num_joints, std::vector<float>(num_joints, 10000.0));
                        int idx_update = index_candidates_left_[idx_match_left[j]];//matching in a left image -> idx_match_left[j]. humans in a left image -> idx_candidates_left_. 
                        organizeRoi(frame, frameIndex, bool_left, keyPoints[idx_update], distances, joints, imgJoint, jointsCenter);
                        humanJoints_left.push_back(joints);
                        if (!imgJoint.empty())
                        {
                            imgHuman_left.push_back(imgJoint);
                        }
                        else
                            imgHuman_left.push_back(std::vector<cv::Mat1b>{});
                        humanJointsCenter_left.push_back(jointsCenter);
                    }
                    else if (!bool_dynamic_roi) //static roi
                    {
                        int idx_update = index_candidates_left_[idx_match_left[j]];//matching in a left image -> idx_match_left[j]. humans in a left image -> idx_candidates_left_. 
                        for (int j = 0; j < keyPoints[idx_update].size(); j++)
                        {
                            organize_left(frame, frameIndex, keyPoints[idx_update][j], joints, imgJoint, jointsCenter);
                        }
                        humanJoints_left.push_back(joints);
                        if (!imgJoint.empty())
                        {
                            imgHuman_left.push_back(imgJoint);
                        }
                        else
                            imgHuman_left.push_back(std::vector<cv::Mat1b>{});
                        humanJointsCenter_left.push_back(jointsCenter);
                    }
                }
                else {//lost data->add lost data.
                    //std::cout << "lost person" << std::endl;
                    humanJoints_left.push_back(lostHuman_search_);
                    humanJointsCenter_left.push_back(lostHuman_center_);
                    imgHuman_left.push_back(std::vector<cv::Mat1b>{});//add empty vector to maintain the data consistency
                }
            }
        }

        //RIGHT
        if (!index_candidates_right_.empty())//human in the left image is found.
        {
            bool bool_left = false;
            for (int j = 0; j < idx_match_right.size(); j++) {//for each human 
                if (idx_match_right[j] != -1) {//found new data.
                    std::vector<cv::Rect2i> joints; // for every joint
                    std::vector<cv::Mat1b> imgJoint;
                    std::vector<std::vector<int>> jointsCenter;
                    /* for every joints : (ls,rs,le,re,lw,rw) */
                    if (bool_dynamic_roi) //dynamic roi
                    {
                        std::vector<float> default_distances(num_joints, max_half_diagonal);
                        std::vector<std::vector<float>>  distances(num_joints, std::vector<float>(num_joints, 10000.0));
                        int idx_update = index_candidates_right_[idx_match_right[j]];//matching in a left image -> idx_match_left[j]. humans in a left image -> idx_candidates_left_. 
                        organizeRoi(frame, frameIndex, bool_left, keyPoints[idx_update], distances, joints, imgJoint, jointsCenter);
                        humanJoints_right.push_back(joints);
                        if (!imgJoint.empty())//found new img
                        {
                            imgHuman_right.push_back(imgJoint);
                        }
                        else//add empty search template image.
                            imgHuman_right.push_back(std::vector<cv::Mat1b>{});
                        humanJointsCenter_right.push_back(jointsCenter);
                    }
                    else if (!bool_dynamic_roi) //static roi
                    {
                        int idx_update = index_candidates_right_[idx_match_right[j]];//matching in a left image -> idx_match_left[j]. humans in a left image -> idx_candidates_left_. 
                        for (int j = 0; j < keyPoints[idx_update].size(); j++)
                        {
                            organize_right(frame, frameIndex, keyPoints[idx_update][j], joints, imgJoint, jointsCenter);
                        }
                        humanJoints_right.push_back(joints);
                        if (!imgJoint.empty())
                        {
                            imgHuman_right.push_back(imgJoint);
                        }
                        else
                            imgHuman_right.push_back(std::vector<cv::Mat1b>{});
                        humanJointsCenter_right.push_back(jointsCenter);
                    }
                }
                else {//lost data->add lost data.
                    humanJoints_right.push_back(lostHuman_search_);
                    humanJointsCenter_right.push_back(lostHuman_center_);
                    imgHuman_right.push_back(std::vector<cv::Mat1b>{});
                }
            }
        }

        //pop before push
        if (!keyPoints.empty() || !index_delete_send_left_.empty() || !index_delete_send_right_.empty()) {
            Yolo2optflow left, right;
            //ROI
            left.roi = humanJoints_left;
            right.roi = humanJoints_right;
            //std::cout << "humanSearchRoi_left=";
            //for (int k = 0; k < humanJoints_left.size(); k++)
            //    std::cout << humanJoints_left[k][0].width << ",";
            //std::cout << std::endl;
            //std::cout << "imgHuman_left=";
            //for (int k = 0; k < imgHuman_left.size(); k++)
            //    std::cout << imgHuman_left[k].size() << ",";
            //std::cout << std::endl;
            //std::cout << "humanSearchRoi_right=";
            //for (int k = 0; k < humanJoints_right.size(); k++)
            //    std::cout << humanJoints_right[k][0].width << ",";
            //std::cout << std::endl;
            //std::cout << "imgHuman_right=";
            //for (int k = 0; k < imgHuman_right.size(); k++)
            //    std::cout << imgHuman_right[k].size() << ",";
            //std::cout << std::endl;
            //search area image
            if (!imgHuman_left.empty()) left.img_search = imgHuman_left;
            if (!imgHuman_right.empty()) right.img_search = imgHuman_right;
            //index to delete.
            if (!index_delete_send_left_.empty()) left.index_delete = index_delete_send_left_;
            if (!index_delete_send_right_.empty()) right.index_delete = index_delete_send_right_;

            //push data
            {
                std::unique_lock<std::mutex> lock_left(mtxYolo_left);
                std::unique_lock<std::mutex> lock_right(mtxYolo_right);
                if (!q_yolo2optflow_left.empty()) q_yolo2optflow_left.pop();
                if (!q_yolo2optflow_right.empty()) q_yolo2optflow_right.pop();
                //push and save data
                q_yolo2optflow_left.push(left);
                q_yolo2optflow_right.push(right);
            }

            //save data
            //if (bool_oneHuman) {
            //    posSaver_left.push_back(humanJointsCenter_left);
            //    posSaver_right.push_back(humanJointsCenter_right);
            //}
            //else {//for multiple people detections.
            //    //Left
            //    if (seqHuman_left.empty()) {//first detection
            //        for (std::vector<std::vector<int>>& d : humanJointsCenter_left)//for each human
            //            seqHuman_left.push_back({ d });
            //    }
            //    else {//after first detection
            //        if (!index_delete_send_left_.empty()) {//data should be deleted
            //            for (int& idx : index_delete_send_left_) {//save data in the saveHuman list.
            //                if (seqHuman_left[idx].size() >= 3) {//save
            //                    saveHuman_left.push_back(seqHuman_left[idx]);
            //                }
            //                seqHuman_left.erase(seqHuman_left.begin() + idx);//delete
            //            }
            //        }

            //        //append data.
            //        for (int i = 0; i < humanJointsCenter_left.size(); i++) {//for each human
            //            if (i < seqHuman_left.size()) {//within existed human
            //                if (humanJointsCenter_left[i][0][0] > 0)//valid detection
            //                    seqHuman_left[i].push_back(humanJointsCenter_left[i]);
            //            }
            //            else {//new human
            //                seqHuman_left.push_back({ humanJointsCenter_left[i] });
            //            }
            //        }
            //    }

            //    //Right
            //    if (seqHuman_right.empty()) {//first detection
            //        for (std::vector<std::vector<int>>& d : humanJointsCenter_right)
            //            seqHuman_right.push_back({ d });
            //    }
            //    else {//after first detection
            //        if (!index_delete_send_right_.empty()) {//data should be deleted
            //            for (int& idx : index_delete_send_right_) {//save data in the saveHuman list.
            //                if (seqHuman_right[idx].size() >= 3) {//save
            //                    saveHuman_right.push_back(seqHuman_right[idx]);
            //                }
            //                seqHuman_right.erase(seqHuman_right.begin() + idx);//delete
            //            }
            //        }

            //        //append data.
            //        for (int i = 0; i < humanJointsCenter_right.size(); i++) {
            //            if (i < seqHuman_right.size()) {//within existed human
            //                if (humanJointsCenter_right[i][0][0] > 0)//valid detection
            //                    seqHuman_right[i].push_back(humanJointsCenter_right[i]);
            //            }
            //            else {//new human
            //                seqHuman_right.push_back({ humanJointsCenter_right[i] });
            //            }
            //        }
            //    }
            //}
        }
    }
}

void YOLOPoseBatch::organizeRoi(cv::Mat1b& frame, int& frameIndex, bool& bool_left, std::vector<std::vector<int>>& pos,
    std::vector<std::vector<float>>& distances, std::vector<cv::Rect2i>& joints, std::vector<cv::Mat1b>& imgJoint,
    std::vector<std::vector<int>>& jointsCenter)
{
    //calculate each joints distances -> save into vector
    //if pos[0] < 0 -> distances is remained
    float distance;
    auto start = std::chrono::high_resolution_clock::now();
    //calculate distance between each joint
    for (int i = 0; i < pos.size() - 1; i++) //for each joint
    {
        if (pos[i][0] > 0)//keypoints found
        {
            int j = i + 1;
            while (j < pos.size()) //calculate distances for each distances
            {
                if (pos[j][0] > 0) //keypoints found
                {
                    distance = std::pow((std::pow((pos[i][0] - pos[j][0]), 2) + std::pow((pos[i][1] - pos[j][1]), 2)), 0.5); //calculate distance
                    distances[i][j] = distance; //save distance
                    distances[j][i] = distance;
                }
                j++;
            }
        }
    }
    //setting roi for each joint
    for (int i = 0; i < pos.size(); i++)
    {

        if (pos[i][0] > 0) //found
        {
            if (i == 0)//left shoulder
            {
                std::vector<int> joints_neighbor{ 1,1 };//1,2
                setRoi(frameIndex, frame, bool_left, distances, i, joints_neighbor, pos, joints, imgJoint, jointsCenter);
            }
            else if (i == 1)//right shoulder
            {
                std::vector<int> joints_neighbor{ 0,0 };//0,3
                setRoi(frameIndex, frame, bool_left, distances, i, joints_neighbor, pos, joints, imgJoint, jointsCenter);
            }
            else if (i == 2)//left elbow
            {
                std::vector<int> joints_neighbor{ 0,0 };//0,4
                setRoi(frameIndex, frame, bool_left, distances, i, joints_neighbor, pos, joints, imgJoint, jointsCenter);
            }
            else if (i == 3)//right elbow
            {
                std::vector<int> joints_neighbor{ 1,1 };//1,5
                setRoi(frameIndex, frame, bool_left, distances, i, joints_neighbor, pos, joints, imgJoint, jointsCenter);
            }
            else if (i == 4)//left wrist
            {
                std::vector<int> joints_neighbor{ 2, 2 };//2,2
                setRoi(frameIndex, frame, bool_left, distances, i, joints_neighbor, pos, joints, imgJoint, jointsCenter);
            }
            else if (i == 5)//right wrist
            {
                std::vector<int> joints_neighbor{ 3, 3 };//3,3
                setRoi(frameIndex, frame, bool_left, distances, i, joints_neighbor, pos, joints, imgJoint, jointsCenter);
            }
        }
        else //not found
        {
            jointsCenter.push_back({ frameIndex, -1, -1 ,-1,-1 });
            joints.emplace_back(-1, -1, -1, -1);
        }
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    //std::cout << "!!!! time taken by setting roi=" << duration.count() << " microseconds !!!!" << std::endl;
}

void YOLOPoseBatch::setRoi(int& frameIndex, cv::Mat1b& frame, bool& bool_left, std::vector<std::vector<float>>& distances,
    int& index_joint, std::vector<int>& compareJoints, std::vector<std::vector<int>>& pos,
    std::vector<cv::Rect2i>& joints, std::vector<cv::Mat1b>& imgJoint, std::vector<std::vector<int>>& jointsCenter)
{
    float vx, vy;
    auto distance_min = std::min_element(distances[index_joint].begin(), distances[index_joint].end());
    //std::cout << "minimum distance=" << (float)(*distance_min) << std::endl;
    float half_diagonal = std::max(min_half_diagonal, ((float)(1.0 - (float)((1.0 - min_ratio) * ((float)(*distance_min) - min_half_diagonal) / (max_half_diagonal - min_half_diagonal))) * (float)(*distance_min)));//calculate half diagonal
    //std::cout << "half diagonal=" << half_diagonal << std::endl;
    if (bool_rotate_roi) //rotate roi
    {
        if (pos[compareJoints[0]][0] > 0)//right shoulder found
        {
            vx = ((float)(pos[compareJoints[0]][0] - pos[index_joint][0])) / distances[index_joint][compareJoints[0]]; //unit direction vector
            vy = ((float)(pos[compareJoints[0]][1] - pos[index_joint][1])) / distances[index_joint][compareJoints[0]];
            //std::cout << " first choice found :: unit direction vector: vx=" << vx << ", vy=" << vy << std::endl;
            if (std::abs(vx) / roi_direction_threshold <= std::abs(vy) && roi_direction_threshold * std::abs(vx) >= std::abs(vy))//withing good region
            {
                if (bool_left)  //left
                    defineRoi_left(frameIndex, frame, index_joint, vx, vy, half_diagonal, pos, joints, imgJoint, jointsCenter);
                else //right
                    defineRoi_right(frameIndex, frame, index_joint, vx, vy, half_diagonal, pos, joints, imgJoint, jointsCenter);
            }
            else //use default vector
            {
                vx = default_neighbor[0];
                vy = default_neighbor[1];
                if (bool_left)  //left
                    defineRoi_left(frameIndex, frame, index_joint, vx, vy, half_diagonal, pos, joints, imgJoint, jointsCenter);
                else //right
                    defineRoi_right(frameIndex, frame, index_joint, vx, vy, half_diagonal, pos, joints, imgJoint, jointsCenter);
            }
        }
        else if (pos[compareJoints[0]][0] <= 0 && pos[compareJoints[1]][0] > 0)//right shoulder not found, left elbow found
        {
            vx = ((float)(pos[compareJoints[1]][0] - pos[index_joint][0])) / distances[index_joint][compareJoints[1]]; //unit direction vector
            vy = ((float)(pos[compareJoints[1]][1] - pos[index_joint][1])) / distances[index_joint][compareJoints[1]];
            //std::cout << " second choice found :: unit direction vector: vx=" << vx << ", vy=" << vy << std::endl;
            if (std::abs(vx) / roi_direction_threshold <= std::abs(vy) && roi_direction_threshold * std::abs(vx) >= std::abs(vy))//withing good region
            {
                if (bool_left)  //left
                    defineRoi_left(frameIndex, frame, index_joint, vx, vy, half_diagonal, pos, joints, imgJoint, jointsCenter);
                else //right
                    defineRoi_right(frameIndex, frame, index_joint, vx, vy, half_diagonal, pos, joints, imgJoint, jointsCenter);
            }
            else //uuse default vector
            {
                vx = default_neighbor[0];
                vy = default_neighbor[1];
                if (bool_left)  //left
                    defineRoi_left(frameIndex, frame, index_joint, vx, vy, half_diagonal, pos, joints, imgJoint, jointsCenter);
                else //right
                    defineRoi_right(frameIndex, frame, index_joint, vx, vy, half_diagonal, pos, joints, imgJoint, jointsCenter);
            }
        }
        else //no neighbors found
        {
            //std::cout << "no neightbors :"<< std::endl;
            vx = default_neighbor[0];
            vy = default_neighbor[1];
            if (bool_left)  //left
                defineRoi_left(frameIndex, frame, index_joint, vx, vy, half_diagonal, pos, joints, imgJoint, jointsCenter);
            else //right
                defineRoi_right(frameIndex, frame, index_joint, vx, vy, half_diagonal, pos, joints, imgJoint, jointsCenter);
        }
    }
    else //not rotate roi
    {
        //std::cout << "no neightbors :"<< std::endl;
        vx = default_neighbor[0];
        vy = default_neighbor[1];
        if (bool_left)  //left
            defineRoi_left(frameIndex, frame, index_joint, vx, vy, half_diagonal, pos, joints, imgJoint, jointsCenter);
        else //right
            defineRoi_right(frameIndex, frame, index_joint, vx, vy, half_diagonal, pos, joints, imgJoint, jointsCenter);
    }
}

void YOLOPoseBatch::defineRoi_left(int& frameIndex, cv::Mat1b& frame, int& index_joint, float& vx, float& vy,
    float& half_diagonal, std::vector<std::vector<int>>& pos,
    std::vector<cv::Rect2i>& joints, std::vector<cv::Mat1b>& imgJoint, std::vector<std::vector<int>>& jointsCenter)
{
    float x1, x2, y1, y2; //candidate points for corners of rectangle
    int left, top, right, bottom; //bbox corners
    x1 = pos[index_joint][0] + half_diagonal * vx;
    y1 = pos[index_joint][1] + half_diagonal * vy;
    x2 = pos[index_joint][0] - half_diagonal * vx;
    y2 = pos[index_joint][1] - half_diagonal * vy;
    left = std::min(std::max((int)(std::min(x1, x2)), 0), originalWidth);
    right = std::max(std::min((int)(std::max(x1, x2)), originalWidth), 0);
    top = std::min(std::max((int)(std::min(y1, y2)), 0), originalHeight);
    bottom = std::max(std::min((int)(std::max(y1, y2)), originalHeight), 0);
    //std::cout << "left=" << left << ", right=" << right << ", top=" << top << ", bottom=" << bottom << std::endl;
    if ((right - left) >= MIN_SEARCH && (bottom - top) >= MIN_SEARCH)
    {
        cv::Rect2i roi(left, top, right - left, bottom - top);
        jointsCenter.push_back({ frameIndex, left,top,(right - left),(bottom - top) });
        joints.push_back(roi);
        imgJoint.push_back(frame(roi));
        //std::cout << "||||| YOLO::roi.width = " << roi.width << ", roi.height = " << roi.height << std::endl;
    }
    else
    {
        jointsCenter.push_back({ frameIndex, -1, -1 ,-1,-1 });
        joints.emplace_back(-1, -1, -1, -1);
    }
}

void YOLOPoseBatch::defineRoi_right(int& frameIndex, cv::Mat1b& frame, int& index_joint, float& vx, float& vy, float& half_diagonal, std::vector<std::vector<int>>& pos,
    std::vector<cv::Rect2i>& joints, std::vector<cv::Mat1b>& imgJoint, std::vector<std::vector<int>>& jointsCenter)
{
    float x1, x2, y1, y2; //candidate points for corners of rectangle
    int left, top, right, bottom; //bbox corners
    x1 = pos[index_joint][0] + half_diagonal * vx;
    y1 = pos[index_joint][1] + half_diagonal * vy;
    x2 = pos[index_joint][0] - half_diagonal * vx;
    y2 = pos[index_joint][1] - half_diagonal * vy;
    left = std::min(std::max((int)(std::min(x1, x2)), 0), originalWidth);
    right = std::max(std::min((int)(std::max(x1, x2)), originalWidth), 0);
    top = std::min(std::max((int)(std::min(y1, y2)), 0), originalHeight);
    bottom = std::max(std::min((int)(std::max(y1, y2)), originalHeight), 0);
    //std::cout << "left=" << left << ", right=" << right << ", top=" << top << ", bottom=" << bottom << std::endl;
    if ((right - left) > 0 && (bottom - top) > 0)
    {
        cv::Rect2i roi(left, top, right - left, bottom - top);
        jointsCenter.push_back({ frameIndex, left,top, (right - left),(bottom - top) });
        joints.push_back(roi);
        roi.x += originalWidth;
        imgJoint.push_back(frame(roi));
        //std::stringstream fileNameStream;
        //fileNameStream <<"yolo-"<<frameIndex << ".jpg";
        //std::string fileName = fileNameStream.str();
        //cv::imwrite(fileName, frame(roi));
        //std::cout << "||||| YOLO::roi.width = " << roi.width << ", roi.height = " << roi.height << std::endl;
    }
    else
    {
        jointsCenter.push_back({ frameIndex, -1, -1,-1,-1 });
        joints.emplace_back(-1, -1, -1, -1);
    }
}

void YOLOPoseBatch::organize_left(cv::Mat1b& frame, int& frameIndex, std::vector<int>& pos, std::vector<cv::Rect2i>& joints, std::vector<cv::Mat1b>& imgJoint, std::vector<std::vector<int>>& jointsCenter)
{
    if (static_cast<int>(pos[0]) >= 0)
    {
        int left = std::min(std::max(static_cast<int>(pos[0] - roiWidthYolo / 2), 0), originalWidth);
        int top = std::min(std::max(static_cast<int>(pos[1] - roiHeightYolo / 2), 0), originalHeight);
        int right = std::max(std::min(static_cast<int>(pos[0] + roiWidthYolo / 2), originalWidth), 0);
        int bottom = std::max(std::min(static_cast<int>(pos[1] + roiHeightYolo / 2), originalHeight), 0);
        cv::Rect2i roi(left, top, right - left, bottom - top);
        jointsCenter.push_back({ frameIndex, roi.x,roi.y,roi.width,roi.height });
        joints.push_back(roi);
        imgJoint.push_back(frame(roi));
        //std::stringstream fileNameStream;
        //fileNameStream <<"yolo-"<<frameIndex << ".jpg";
        //std::string fileName = fileNameStream.str();
        //cv::imwrite(fileName, frame(roi));
    }
    /* keypoints can't be detected */
    else
    {
        jointsCenter.push_back({ frameIndex, -1, -1,-1,-1 });
        joints.emplace_back(-1, -1, -1, -1);
    }
}

void YOLOPoseBatch::organize_right(cv::Mat1b& frame, int& frameIndex, std::vector<int>& pos, std::vector<cv::Rect2i>& joints, std::vector<cv::Mat1b>& imgJoint, std::vector<std::vector<int>>& jointsCenter)
{
    if (static_cast<int>(pos[0]) >= 0)
    {
        int left = std::min(std::max(static_cast<int>(pos[0] - roiWidthYolo / 2), 0), originalWidth);
        int top = std::min(std::max(static_cast<int>(pos[1] - roiHeightYolo / 2), 0), originalHeight);
        int right = std::max(std::min(static_cast<int>(pos[0] + roiWidthYolo / 2), originalWidth), 0);
        int bottom = std::max(std::min(static_cast<int>(pos[1] + roiHeightYolo / 2), originalHeight), 0);
        cv::Rect2i roi(left, top, right - left, bottom - top);
        joints.push_back(roi);
        jointsCenter.push_back({ frameIndex, roi.x,roi.y,roi.width,roi.height });
        roi.x += originalWidth;
        imgJoint.push_back(frame(roi));

    }
    /* keypoints can't be detected */
    else
    {
        jointsCenter.push_back({ frameIndex, -1, -1,-1,-1 });
        joints.emplace_back(-1, -1, -1, -1);
    }
}