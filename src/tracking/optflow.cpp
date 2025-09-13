#include "optflow.h"


void OpticalFlow::main(cv::Mat1b& frame, const int& frameIndex,
    std::queue<Yolo2optflow>& q_yolo2optflow,
    std::queue<Optflow2optflow>& q_optflow2optflow,
    std::queue<Optflow2tri>& q_optflow2tri,
    std::vector<std::vector<std::vector<int>>>& updatedPositionsHuman, bool bool_left)
{
    /* optical flow process for each joints */
    index_delete.clear();//reset
    std::vector<std::vector<cv::Mat1b>> previousImg;           //[number of human,0~6,cv::Mat1b]
    std::vector<std::vector<cv::Rect2i>> searchRoi;            //[number of human,6,cv::Rect2i], if tracker was failed, roi.x == -1
    std::vector<std::vector<std::vector<float>>> previousMove; //[number of human,6,movement in x and y] ROI movement of each joint
    std::vector<std::vector<cv::Ptr<cv::DISOpticalFlow>>> previousDIS;
    //std::cout << "optflow : 0" << std::endl;
    getPreviousData(frame, previousImg, searchRoi, previousMove, previousDIS, q_yolo2optflow, q_optflow2optflow, bool_left);
    //std::cout << "optflow : 1" << std::endl;
    /* prepare storage */
    /* for every human */
    std::vector<std::vector<cv::Mat1b>> updatedImgHuman;
    std::vector<std::vector<cv::Rect2i>> updatedSearchRoiHuman;

    std::vector<std::vector<std::vector<float>>> updatedMoveDists;
    std::vector<std::vector<cv::Ptr<cv::DISOpticalFlow>>> updatedDISHuman;

    //start tracking with optflow
    if (!searchRoi.empty())
    {
        if (!bool_multithread_ || searchRoi.size() == 1) {//with one thread
            for (int i = 0; i < searchRoi.size(); i++)//for each human.
            {
                //std::cout << i << "-th human::" << "previousMove.size()=" << previousMove[i].size() << std::endl;
                //std::cout<<"previousImg.size()=" << previousImg[i].size() << std::endl;
                /* for every joints */
                std::vector<cv::Mat1b> updatedImgJoints;
                std::vector<cv::Rect2i> updatedSearchRoi;
                std::vector<std::vector<float>> moveJoints; //roi movement
                std::vector<cv::Ptr<cv::DISOpticalFlow>> disJoints;
                std::vector<std::vector<int>> updatedPositions;

                iteration(i, frameIndex, frame, previousImg, searchRoi, previousMove, previousDIS,
                    updatedImgJoints, updatedSearchRoi, moveJoints, disJoints, updatedPositions);

                /* combine all data for one human */
                updatedSearchRoiHuman.push_back(updatedSearchRoi);
                updatedPositionsHuman.push_back(updatedPositions);

                //every time pass data to the next generation to maintain data consistency.
                updatedImgHuman.push_back(updatedImgJoints);
                updatedMoveDists.push_back(moveJoints);
                updatedDISHuman.push_back(disJoints);

                //std::cout << "optflow : 5" << std::endl;
                //if (!updatedImgJoints.empty())
                //{
                //    updatedImgHuman.push_back(updatedImgJoints);
                //    updatedMoveDists.push_back(moveJoints);
                //    updatedDISHuman.push_back(disJoints);
                //}
            }

            /* push updated data to queue */
            Optflow2optflow newData;
            newData.roi = updatedSearchRoiHuman;

            //pass all the data to the next generation to maintain the data consistency.
            newData.img_search = updatedImgHuman;
            newData.move = updatedMoveDists;
            newData.ptr_dis = updatedDISHuman;

            q_optflow2optflow.push(newData);
        }
        else {//with two threads
            int n_human = searchRoi.size();
            int counter_human = 0;
            while (counter_human < n_human) {
                if (n_human - counter_human >= 2) {//multi thread
                    std::vector<cv::Mat1b> updatedImgJoints1, updatedImgJoints2;
                    std::vector<cv::Rect2i> updatedSearchRoi1, updatedSearchRoi2;
                    std::vector<std::vector<float>> moveJoints1, moveJoints2; //roi movement
                    std::vector<cv::Ptr<cv::DISOpticalFlow>> disJoints1, disJoints2;
                    std::vector<std::vector<int>> updatedPositions1, updatedPositions2;

                    std::thread thread_1st(&OpticalFlow::iteration, this, counter_human, std::ref(frameIndex), std::ref(frame), std::ref(previousImg), std::ref(searchRoi), std::ref(previousMove), std::ref(previousDIS),
                        std::ref(updatedImgJoints1), std::ref(updatedSearchRoi1), std::ref(moveJoints1), std::ref(disJoints1), std::ref(updatedPositions1));
                    counter_human++;//increment human counter

                    iteration(counter_human, frameIndex, frame, previousImg, searchRoi, previousMove, previousDIS,
                        updatedImgJoints2, updatedSearchRoi2, moveJoints2, disJoints2, updatedPositions2);
                    counter_human++;//increment by 2

                    thread_1st.join();//wait for the first thread

                    //FIRST
                    /* combine all data for one human */
                    updatedSearchRoiHuman.push_back(updatedSearchRoi1);
                    updatedPositionsHuman.push_back(updatedPositions1);
                    //every time pass data to the next generation to maintain data consistency.
                    updatedImgHuman.push_back(updatedImgJoints1);
                    updatedMoveDists.push_back(moveJoints1);
                    updatedDISHuman.push_back(disJoints1);

                    //SECOND
                    /* combine all data for one human */
                    updatedSearchRoiHuman.push_back(updatedSearchRoi2);
                    updatedPositionsHuman.push_back(updatedPositions2);
                    //every time pass data to the next generation to maintain data consistency.
                    updatedImgHuman.push_back(updatedImgJoints2);
                    updatedMoveDists.push_back(moveJoints2);
                    updatedDISHuman.push_back(disJoints2);

                }
                else {//single thread
                    std::vector<cv::Mat1b> updatedImgJoints;
                    std::vector<cv::Rect2i> updatedSearchRoi;
                    std::vector<std::vector<float>> moveJoints; //roi movement
                    std::vector<cv::Ptr<cv::DISOpticalFlow>> disJoints;
                    std::vector<std::vector<int>> updatedPositions;

                    iteration(counter_human, frameIndex, frame, previousImg, searchRoi, previousMove, previousDIS,
                        updatedImgJoints, updatedSearchRoi, moveJoints, disJoints, updatedPositions);

                    /* combine all data for one human */
                    updatedSearchRoiHuman.push_back(updatedSearchRoi);
                    updatedPositionsHuman.push_back(updatedPositions);

                    //every time pass data to the next generation to maintain data consistency.
                    updatedImgHuman.push_back(updatedImgJoints);
                    updatedMoveDists.push_back(moveJoints);
                    updatedDISHuman.push_back(disJoints);
                    counter_human++;//increment
                }
            }

            /* push updated data to queue */
            Optflow2optflow newData;
            newData.roi = updatedSearchRoiHuman;

            //pass all the data to the next generation to maintain the data consistency.
            newData.img_search = updatedImgHuman;
            newData.move = updatedMoveDists;
            newData.ptr_dis = updatedDISHuman;

            q_optflow2optflow.push(newData);

        }
    }
    // no data
    else
    {
        //nothing to do
    }

}

void OpticalFlow::getPreviousData(cv::Mat1b& frame, std::vector<std::vector<cv::Mat1b>>& previousImg, std::vector<std::vector<cv::Rect2i>>& searchRoi,
    std::vector<std::vector<std::vector<float>>>& moveDists, std::vector<std::vector<cv::Ptr<cv::DISOpticalFlow>>>& previousDIS,
    std::queue<Yolo2optflow>& q_yolo2optflow, std::queue<Optflow2optflow>& q_optflow2optflow, bool& bool_left)
{
    /*
    *  if yolo data available -> update if tracking was failed
    *  else -> if tracking of optical flow was successful -> update features
    *  else -> if tracking of optical flow was failed -> wait for next update of yolo
    *  [TO DO]
    *  get Yolo data and OF data from queue if possible
    *  organize data
    */

    if (!q_optflow2optflow.empty())
    {
        Optflow2optflow prevData = q_optflow2optflow.front();
        q_optflow2optflow.pop();
        if (!prevData.img_search.empty())
        {
            previousImg = prevData.img_search;
            searchRoi = prevData.roi;
            moveDists = prevData.move;
            previousDIS = prevData.ptr_dis;
        }
        else if (prevData.img_search.empty() && !prevData.roi.empty())
            searchRoi = prevData.roi;
    }

    std::vector<std::vector<cv::Mat1b>> previousYoloImg;
    std::vector<std::vector<cv::Rect2i>> searchYoloRoi;
    if (!q_yolo2optflow.empty())
    {
        //std::cout << "yolo data is available" << std::endl;
        getYoloData(previousYoloImg, searchYoloRoi, q_yolo2optflow, bool_left);
        //delete lost human first
        if (!index_delete.empty()) {//the order is in a descending order to maintain the index consistency
            //std::cout << "Optflow :: searchRoi.size()=" << searchRoi.size() << ", previousImg.size()=" << previousImg.size() << ", moveDists.size()=" << moveDists.size() << ", previousDIS.size()=" << previousDIS.size() << std::endl;
            //std::cout << "Optflow :: delete data=";
            for (int& idx : index_delete) {
                //std::cout << idx << ",";
                searchRoi.erase(searchRoi.begin() + idx);
                previousImg.erase(previousImg.begin() + idx);
                moveDists.erase(moveDists.begin() + idx);
                previousDIS.erase(previousDIS.begin() + idx);
            }
            //std::cout << "all" << std::endl;
        }
        //std::cout << "get YOLO data" << std::endl;
        /* update data here */
        /* iterate for all human detection */
        if (!previousYoloImg.empty()) {
            //std::cout << "searchYoloRoi.size()=" << searchYoloRoi.size() << ", searchRoi.size()=" << searchRoi.size() << ", previousImg.size()=" << previousImg.size() << std::endl;
            for (int i = 0; i < searchYoloRoi.size(); i++)//for each human
            {
                /* some OF tracking were successful */
                if (!previousImg.empty())
                {
                    /* existed human detection */
                    if (i < previousImg.size())
                    {
                        //std::cout << "searchYoloRoi["<<i<<"].size()="<<searchYoloRoi[i].size()<<", previousYoloImg["<<i<<"].size()=" <<previousYoloImg[i].size() << ", previousImg[" << i << "].size() = " << previousImg[i].size() << "saerchRoi[" << i << "].size() = " << searchRoi[i].size() << "moveDists[" << i << "].size() = " << moveDists[i].size() << "previousDIS[" << i << "].size() = " << previousDIS[i].size() << std::endl;
                        //std::cout << "previousImg : num of human : " << previousImg.size() << std::endl;
                        /* for all joints */
                        int counterJoint = 0;
                        int counterYoloImg = 0;
                        int counterTrackerOF = 0; // number of successful trackers by Optical Flow
                        for (cv::Rect2i& roi : searchRoi[i])//for each joint
                        {
                            //std::cout << "searchRoi.width=" << roi.width <<", searchYoloRoi.width="<<searchYoloRoi[i][counterJoint].width<<", counterJoint="<<counterJoint << std::endl;
                            /* tracking is failed -> update data with yolo data */
                            if (roi.width <= 0)
                            {
                                /* yolo detect joints -> update data */
                                if (searchYoloRoi[i][counterJoint].width > 0)
                                {
                                    //std::cout << "update OF tracker features with yolo detection" << std::endl;
                                    //std::cout << "previousRoi.x=" << searchRoi[i][counterJoint].x << ", width=" << searchRoi[i][counterJoint].width << std::endl;
                                    searchRoi[i][counterJoint] = searchYoloRoi[i][counterJoint];
                                    previousImg[i].insert(previousImg[i].begin() + counterTrackerOF, previousYoloImg[i][counterYoloImg]);
                                    moveDists[i].insert(moveDists[i].begin() + counterTrackerOF, { 0.0,0.0 });
                                    //make new DIS ptr
                                    if (dis_mode == 0) //ultrafast mode
                                    {
                                        cv::Ptr<cv::DISOpticalFlow> dis = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_ULTRAFAST);
                                        //dis->setUseSpatialPropagation(true);
                                        if (bool_manual_patch_dis)
                                        {
                                            dis->setPatchSize(disPatch);
                                            dis->setPatchStride(disStride);
                                        }
                                        previousDIS[i].insert(previousDIS[i].begin() + counterTrackerOF, dis);
                                    }
                                    else if (dis_mode == 1) //fast mode
                                    {
                                        cv::Ptr<cv::DISOpticalFlow> dis = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_FAST);
                                        if (bool_manual_patch_dis)
                                        {
                                            dis->setPatchSize(disPatch);
                                            dis->setPatchStride(disStride);
                                        }
                                        //dis->setUseSpatialPropagation(true);
                                        previousDIS[i].insert(previousDIS[i].begin() + counterTrackerOF, dis);
                                    }
                                    else if (dis_mode == 2) //medium mode
                                    {
                                        cv::Ptr<cv::DISOpticalFlow> dis = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM);
                                        if (bool_manual_patch_dis)
                                        {
                                            dis->setPatchSize(disPatch);
                                            dis->setPatchStride(disStride);
                                        }
                                        //dis->setUseSpatialPropagation(true);
                                        previousDIS[i].insert(previousDIS[i].begin() + counterTrackerOF, dis);
                                    }
                                    //std::cout << "after updating with Yolo data :: previousRoi.x=" << searchRoi[i][counterJoint].x << ", width=" << searchRoi[i][counterJoint].width << std::endl;
                                    counterJoint++;
                                    counterYoloImg++;
                                    counterTrackerOF++;
                                }
                                /* yolo can't detect joint -> not updated data */
                                else
                                {
                                    //std::cout << "Yolo didn't detect joint" << std::endl;
                                    counterJoint++;
                                }
                            }
                            /* tracking is successful */
                            else
                            {
                                //std::cout << "tracking was successful" << std::endl;
                                /* update template image with Yolo's one */
                                if (boolChange)
                                {
                                    if (searchYoloRoi[i][counterJoint].width > 0)
                                    {
                                        //if ((float)std::abs((roi.x - searchYoloRoi[i][counterJoint].x)) >= DIF_THRESHOLD || (float)std::abs((searchRoi[i][counterJoint].y - searchYoloRoi[i][counterJoint].y)) >= DIF_THRESHOLD ||
                                        //    (std::pow(static_cast<float>(roi.x - searchYoloRoi[i][counterJoint].x), 2) + std::pow(static_cast<float>(roi.x - searchYoloRoi[i][counterJoint].x), 2) >= DIF_THRESHOLD * DIF_THRESHOLD))
                                        //{
                                        //std::cout << "update OF tracker features with yolo detection" << std::endl;
                                        //std::cout << "====== previousRoi.x=" << searchRoi[i][counterJoint].x << ", width=" << searchRoi[i][counterJoint].width << std::endl;
                                        if (bool_check_pos)//check position when updating
                                        {
                                            //current
                                            float left_current = (float)roi.x;
                                            float top_current = (float)roi.y;
                                            float centerX_current = (float)(roi.x + roi.width / 2);
                                            float centerY_current = (float)(roi.y + roi.height / 2);
                                            //yolo
                                            float left_yolo = (float)searchYoloRoi[i][counterJoint].x;
                                            float top_yolo = (float)searchYoloRoi[i][counterJoint].y;
                                            float centerX_yolo = (float)((float)searchYoloRoi[i][counterJoint].x + (float)searchYoloRoi[i][counterJoint].width / 2.0);
                                            float centerY_yolo = (float)((float)searchYoloRoi[i][counterJoint].y + (float)searchYoloRoi[i][counterJoint].height / 2.0);
                                            float moveX = std::abs(moveDists[i][counterTrackerOF][0]); float moveY = (moveDists[i][counterTrackerOF][1]);
                                            //update data with Yolo detection
                                            if (std::abs((left_current - left_yolo)) >= DIF_THRESHOLD || std::abs((top_current - top_yolo)) >= DIF_THRESHOLD || //position difference
                                                (std::pow((centerX_current - centerX_yolo), 2) + std::pow((centerY_current - centerY_yolo), 2) >= DIF_THRESHOLD * DIF_THRESHOLD)  //position difference
                                                )
                                            {
                                                previousImg[i][counterTrackerOF] = previousYoloImg[i][counterYoloImg];
                                                moveDists[i][counterTrackerOF] = std::vector<float>{ 0.0,0.0 };
                                                searchRoi[i][counterJoint] = searchYoloRoi[i][counterJoint];

                                                //make new DIS ptr
                                                if (dis_mode == 0) //ultrafast mode
                                                {
                                                    cv::Ptr<cv::DISOpticalFlow> dis = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_ULTRAFAST);
                                                    if (bool_manual_patch_dis)
                                                    {
                                                        dis->setPatchSize(disPatch);
                                                        dis->setPatchStride(disStride);
                                                    }
                                                    //dis->setUseSpatialPropagation(true);
                                                    previousDIS[i][counterTrackerOF] = dis;
                                                }
                                                else if (dis_mode == 1) //fast mode
                                                {
                                                    cv::Ptr<cv::DISOpticalFlow> dis = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_FAST);
                                                    if (bool_manual_patch_dis)
                                                    {
                                                        dis->setPatchSize(disPatch);
                                                        dis->setPatchStride(disStride);
                                                    }
                                                    //dis->setUseSpatialPropagation(true);
                                                    previousDIS[i][counterTrackerOF] = dis;
                                                }
                                                else if (dis_mode == 2) //medium mode
                                                {
                                                    cv::Ptr<cv::DISOpticalFlow> dis = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM);
                                                    if (bool_manual_patch_dis)
                                                    {
                                                        dis->setPatchSize(disPatch);
                                                        dis->setPatchStride(disStride);
                                                    }
                                                    //dis->setUseSpatialPropagation(true);
                                                    previousDIS[i][counterTrackerOF] = dis;
                                                }

                                                //std::cout << "update OF tracker features with yolo detection" << std::endl;
                                                //std::cout << "====== previousRoi.x=" << searchRoi[i][counterJoint].x << ", width=" << searchRoi[i][counterJoint].width << std::endl;
                                            }
                                            else //valid tracking -> only update configuration of search area
                                            {
                                                searchRoi[i][counterJoint].width = searchYoloRoi[i][counterJoint].width;
                                                searchRoi[i][counterJoint].height = searchYoloRoi[i][counterJoint].height;
                                                previousImg[i][counterTrackerOF] = frame(searchRoi[i][counterJoint]);
                                                //moveDists[i][counterTrackerOF] = std::vector<float>{ 0.0,0.0 };
                                                //make new DIS ptr
                                                if (dis_mode == 0) //ultrafast mode
                                                {
                                                    cv::Ptr<cv::DISOpticalFlow> dis = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_ULTRAFAST);
                                                    if (bool_manual_patch_dis)
                                                    {
                                                        dis->setPatchSize(disPatch);
                                                        dis->setPatchStride(disStride);
                                                    }
                                                    //dis->setUseSpatialPropagation(true);
                                                    previousDIS[i][counterTrackerOF] = dis;
                                                }
                                                else if (dis_mode == 1) //fast mode
                                                {
                                                    cv::Ptr<cv::DISOpticalFlow> dis = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_FAST);
                                                    if (bool_manual_patch_dis)
                                                    {
                                                        dis->setPatchSize(disPatch);
                                                        dis->setPatchStride(disStride);
                                                    }
                                                    //dis->setUseSpatialPropagation(true);
                                                    previousDIS[i][counterTrackerOF] = dis;
                                                }
                                                else if (dis_mode == 2) //medium mode
                                                {
                                                    cv::Ptr<cv::DISOpticalFlow> dis = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM);
                                                    if (bool_manual_patch_dis)
                                                    {
                                                        dis->setPatchSize(disPatch);
                                                        dis->setPatchStride(disStride);
                                                    }
                                                    //dis->setUseSpatialPropagation(true);
                                                    previousDIS[i][counterTrackerOF] = dis;
                                                }
                                            }
                                        }
                                        else if (!bool_check_pos) //check position when updating
                                        {
                                            previousImg[i][counterTrackerOF] = previousYoloImg[i][counterYoloImg];
                                            moveDists[i][counterTrackerOF] = std::vector<float>{ 0.0,0.0 };
                                            searchRoi[i][counterJoint] = searchYoloRoi[i][counterJoint];

                                            //make new DIS ptr
                                            if (dis_mode == 0) //ultrafast mode
                                            {
                                                cv::Ptr<cv::DISOpticalFlow> dis = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_ULTRAFAST);
                                                if (bool_manual_patch_dis)
                                                {
                                                    dis->setPatchSize(disPatch);
                                                    dis->setPatchStride(disStride);
                                                }
                                                //dis->setUseSpatialPropagation(true);
                                                previousDIS[i][counterTrackerOF] = dis;
                                            }
                                            else if (dis_mode == 1) //fast mode
                                            {
                                                cv::Ptr<cv::DISOpticalFlow> dis = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_FAST);
                                                if (bool_manual_patch_dis)
                                                {
                                                    dis->setPatchSize(disPatch);
                                                    dis->setPatchStride(disStride);
                                                }
                                                //dis->setUseSpatialPropagation(true);
                                                previousDIS[i][counterTrackerOF] = dis;
                                            }
                                            else if (dis_mode == 2) //medium mode
                                            {
                                                cv::Ptr<cv::DISOpticalFlow> dis = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM);
                                                if (bool_manual_patch_dis)
                                                {
                                                    dis->setPatchSize(disPatch);
                                                    dis->setPatchStride(disStride);
                                                }
                                                //dis->setUseSpatialPropagation(true);
                                                previousDIS[i][counterTrackerOF] = dis;
                                            }

                                        }
                                        //std::cout << " ~~~~~~~~~~~~~ after updating with Yolo data :: previousRoi.x=" << searchRoi[i][counterJoint].x << ", width=" << searchRoi[i][counterJoint].width << std::endl;
                                        //}
                                        counterYoloImg++;
                                    }
                                }
                                /* not update template images -> keep tracking */
                                else
                                {
                                    if (searchYoloRoi[i][counterJoint].x >= 0)
                                        counterYoloImg++;
                                }
                                counterJoint++;
                                counterTrackerOF++;
                                //std::cout << "update iterator" << std::endl;
                            }
                        }
                    }
                    /* new human detected */
                    else
                    {
                        //std::cout << "new human was detected by Yolo " << std::endl;
                        int counterJoint = 0;
                        int counterYoloImg = 0;
                        std::vector<cv::Rect2i> joints;
                        std::vector<cv::Mat1b> imgJoints;
                        std::vector<std::vector<float>> moveJoints;
                        std::vector<cv::Ptr<cv::DISOpticalFlow>> disJoints;
                        /* for every joints */
                        for (const cv::Rect2i& roi : searchYoloRoi[i])
                        {
                            /* keypoint is found */
                            if (roi.x >= 0)
                            {
                                joints.push_back(roi);
                                imgJoints.push_back(previousYoloImg[i][counterYoloImg]);
                                moveJoints.push_back(defaultMove);

                                //make new DIS ptr
                                if (dis_mode == 0) //ultrafast mode
                                {
                                    cv::Ptr<cv::DISOpticalFlow> dis = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_ULTRAFAST);
                                    if (bool_manual_patch_dis)
                                    {
                                        dis->setPatchSize(disPatch);
                                        dis->setPatchStride(disStride);
                                    }
                                    //dis->setUseSpatialPropagation(true);
                                    disJoints.push_back(dis);
                                }
                                else if (dis_mode == 1) //fast mode
                                {
                                    cv::Ptr<cv::DISOpticalFlow> dis = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_FAST);
                                    if (bool_manual_patch_dis)
                                    {
                                        dis->setPatchSize(disPatch);
                                        dis->setPatchStride(disStride);
                                    }
                                    //dis->setUseSpatialPropagation(true);
                                    disJoints.push_back(dis);
                                }
                                else if (dis_mode == 2) //medium mode
                                {
                                    cv::Ptr<cv::DISOpticalFlow> dis = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM);
                                    if (bool_manual_patch_dis)
                                    {
                                        dis->setPatchSize(disPatch);
                                        dis->setPatchStride(disStride);
                                    }
                                    //dis->setUseSpatialPropagation(true);
                                    disJoints.push_back(dis);
                                }
                                counterJoint++;
                                counterYoloImg++;
                            }
                            /* keypoints not found */
                            else
                            {
                                joints.push_back(roi);
                                counterJoint++;
                            }
                        }
                        searchRoi.push_back(joints);
                        if (!imgJoints.empty())
                        {
                            previousImg.push_back(imgJoints);
                            moveDists.push_back(moveJoints);
                            previousDIS.push_back(disJoints);
                        }
                    }
                }
                else/* no OF tracking was successful or first yolo detection */
                {
                    //std::cout << "Optflow :: First detection" << std::endl;
                    searchRoi.clear(); // initialize searchRoi for avoiding data
                    //std::cout << "Optical Flow :: failed or first Yolo detection " << std::endl;
                    int counterJoint = 0;
                    int counterYoloImg = 0;
                    std::vector<cv::Rect2i> joints;
                    std::vector<cv::Mat1b> imgJoints;
                    std::vector<std::vector<float>> moveJoints;
                    std::vector<std::vector<cv::Point2f>> features;
                    std::vector<cv::Ptr<cv::DISOpticalFlow>> disJoints;
                    /* for every joint */
                    for (const cv::Rect2i& roi : searchYoloRoi[i])
                    {
                        /* keypoint is found */
                        if (roi.width > 0)
                        {
                            joints.push_back(roi);
                            imgJoints.push_back(previousYoloImg[i][counterYoloImg]);
                            moveJoints.push_back({ 0.0, 0.0 });
                            //make new DIS ptr
                            if (dis_mode == 0) //ultrafast mode
                            {
                                cv::Ptr<cv::DISOpticalFlow> dis = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_ULTRAFAST);
                                if (bool_manual_patch_dis)
                                {
                                    dis->setPatchSize(disPatch);
                                    dis->setPatchStride(disStride);
                                }
                                //dis->setUseSpatialPropagation(true);
                                disJoints.push_back(dis);
                            }
                            else if (dis_mode == 1) //fast mode
                            {
                                cv::Ptr<cv::DISOpticalFlow> dis = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_FAST);
                                //dis->setUseSpatialPropagation(true);
                                if (bool_manual_patch_dis)
                                {
                                    dis->setPatchSize(disPatch);
                                    dis->setPatchStride(disStride);
                                }
                                disJoints.push_back(dis);
                            }
                            else if (dis_mode == 2) //medium mode
                            {
                                cv::Ptr<cv::DISOpticalFlow> dis = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM);
                                //dis->setUseSpatialPropagation(true);
                                if (bool_manual_patch_dis)
                                {
                                    dis->setPatchSize(disPatch);
                                    dis->setPatchStride(disStride);
                                }
                                disJoints.push_back(dis);
                            }

                            counterJoint++;
                            counterYoloImg++;
                        }
                        /* keypoints not found */
                        else
                        {
                            joints.push_back(roi);
                            counterJoint++;
                        }
                    }
                    searchRoi.push_back(joints);
                    if (!imgJoints.empty())
                    {
                        previousImg.push_back(imgJoints);
                        moveDists.push_back(moveJoints);
                        previousDIS.push_back(disJoints);
                    }
                }
            }
        }
    }
}

void OpticalFlow::getYoloData(std::vector<std::vector<cv::Mat1b>>& previousYoloImg, std::vector<std::vector<cv::Rect2i>>& searchYoloRoi,
    std::queue<Yolo2optflow>& q_yolo2optflow, bool& bool_left)
{
    Yolo2optflow prevData;
    bool bool_success = false;
    //{
    //    if (bool_left)//left mutex
    //        std::unique_lock<std::mutex> lock(mtxYolo_left);
    //    else//right mutex
    //        std::unique_lock<std::mutex> lock(mtxYolo_right);
    if (!q_yolo2optflow.empty()) {
        prevData = q_yolo2optflow.front();
        q_yolo2optflow.pop();
        bool_success = true;
    }
    else
        bool_success = false;
    //}
    //std::unique_lock<std::mutex> lock(mtxYolo);
    if (bool_success) {
        if (!prevData.img_search.empty()) previousYoloImg = prevData.img_search;
        if (!prevData.roi.empty()) searchYoloRoi = prevData.roi;
        if (!prevData.index_delete.empty()) index_delete = prevData.index_delete;
    }
}

void OpticalFlow::iteration(int i, const int& frameIndex, cv::Mat1b& frame, std::vector<std::vector<cv::Mat1b>>& previousImg, std::vector<std::vector<cv::Rect2i>>& searchRoi,
    std::vector<std::vector<std::vector<float>>>& previousMove, std::vector<std::vector<cv::Ptr<cv::DISOpticalFlow>>>& previousDIS,
    std::vector<cv::Mat1b>& updatedImgJoints, std::vector<cv::Rect2i>& updatedSearchRoi, std::vector<std::vector<float>>& moveJoints,
    std::vector<cv::Ptr<cv::DISOpticalFlow>>& disJoints, std::vector<std::vector<int>>& updatedPositions)
{

    std::vector<int> updatedPosLeftShoulder, updatedPosRightShoulder, updatedPosLeftElbow, updatedPosRightElbow, updatedPosLeftWrist, updatedPosRightWrist;
    cv::Mat1b updatedImgLeftShoulder, updatedImgRightShoulder, updatedImgLeftElbow, updatedImgRightElbow, updatedImgLeftWrist, updatedImgRightWrist;
    cv::Rect2i updatedSearchRoiLeftShoulder, updatedSearchRoiRightShoulder, updatedSearchRoiLeftElbow, updatedSearchRoiRightElbow, updatedSearchRoiLeftWrist, updatedSearchRoiRightWrist;
    std::vector<float> moveLS, moveRS, moveLE, moveRE, moveLW, moveRW;
    cv::Ptr<cv::DISOpticalFlow> updatedDIS_ls, updatedDIS_rs, updatedDIS_le, updatedDIS_re, updatedDIS_lw, updatedDIS_rw;
    bool boolLeftShoulder = false;
    bool boolRightShoulder = false;
    bool boolLeftElbow = false;
    bool boolRightElbow = false;
    bool boolLeftWrist = false;
    bool boolRightWrist = false;
    std::vector<std::thread> threadJoints;
    /* start optical flow process for each joints */
    /* left shoulder */
    int counterTracker = 0;
    //std::cout << "LS" << std::endl;
    if (searchRoi[i][0].width > 0)
    {
        threadJoints.emplace_back(&OpticalFlow::opticalFlow, this, frame, std::ref(frameIndex),
            std::ref(previousImg[i][counterTracker]), std::ref(searchRoi[i][0]), std::ref(previousMove[i][counterTracker]), std::ref(previousDIS[i][counterTracker]),
            std::ref(updatedImgLeftShoulder), std::ref(updatedSearchRoiLeftShoulder), std::ref(updatedDIS_ls), std::ref(moveLS), std::ref(updatedPosLeftShoulder));
        boolLeftShoulder = true;
        counterTracker++;
    }
    else
    {
        updatedSearchRoiLeftShoulder = cv::Rect2i(-1, -1, -1, -1);
        updatedPosLeftShoulder = std::vector<int>{ frameIndex, -1, -1,-1,-1 };
    }
    /* right shoulder */
    //std::cout << "RS" << std::endl;
    if (searchRoi[i][1].width > 0)
    {
        threadJoints.emplace_back(&OpticalFlow::opticalFlow, this, frame, std::ref(frameIndex),
            std::ref(previousImg[i][counterTracker]), std::ref(searchRoi[i][1]), std::ref(previousMove[i][counterTracker]), std::ref(previousDIS[i][counterTracker]),
            std::ref(updatedImgRightShoulder), std::ref(updatedSearchRoiRightShoulder), std::ref(updatedDIS_rs), std::ref(moveRS), std::ref(updatedPosRightShoulder));
        boolRightShoulder = true;
        counterTracker++;
    }
    else
    {
        updatedSearchRoiRightShoulder = cv::Rect2i(-1, -1, -1, -1);
        updatedPosRightShoulder = std::vector<int>{ frameIndex, -1, -1,-1,-1 };
    }
    /* left elbow */
    //std::cout << "LE" << std::endl;
    if (searchRoi[i][2].width > 0)
    {
        threadJoints.emplace_back(&OpticalFlow::opticalFlow, this, frame, std::ref(frameIndex),
            std::ref(previousImg[i][counterTracker]), std::ref(searchRoi[i][2]), std::ref(previousMove[i][counterTracker]), std::ref(previousDIS[i][counterTracker]),
            std::ref(updatedImgLeftElbow), std::ref(updatedSearchRoiLeftElbow), std::ref(updatedDIS_le), std::ref(moveLE), std::ref(updatedPosLeftElbow));
        boolLeftElbow = true;
        counterTracker++;
    }
    else
    {
        updatedSearchRoiLeftElbow = cv::Rect2i(-1, -1, -1, -1);
        updatedPosLeftElbow = std::vector<int>{ frameIndex, -1, -1,-1,-1 };
    }
    /* right elbow */
    //std::cout << "RE" << std::endl;
    if (searchRoi[i][3].width > 0)
    {
        threadJoints.emplace_back(&OpticalFlow::opticalFlow, this, frame, std::ref(frameIndex),
            std::ref(previousImg[i][counterTracker]), std::ref(searchRoi[i][3]), std::ref(previousMove[i][counterTracker]), std::ref(previousDIS[i][counterTracker]),
            std::ref(updatedImgRightElbow), std::ref(updatedSearchRoiRightElbow), std::ref(updatedDIS_re), std::ref(moveRE), std::ref(updatedPosRightElbow));
        boolRightElbow = true;
        counterTracker++;
    }
    else
    {
        updatedSearchRoiRightElbow = cv::Rect2i(-1, -1, -1, -1);
        updatedPosRightElbow = std::vector<int>{ frameIndex, -1, -1,-1,-1 };
    }
    /* left wrist */
    //std::cout << "LW" << std::endl;
    if (searchRoi[i][4].width > 0)
    {
        threadJoints.emplace_back(&OpticalFlow::opticalFlow, this, frame, std::ref(frameIndex),
            std::ref(previousImg[i][counterTracker]), std::ref(searchRoi[i][4]), std::ref(previousMove[i][counterTracker]), std::ref(previousDIS[i][counterTracker]),
            std::ref(updatedImgLeftWrist), std::ref(updatedSearchRoiLeftWrist), std::ref(updatedDIS_lw), std::ref(moveLW), std::ref(updatedPosLeftWrist));
        boolLeftWrist = true;
        counterTracker++;
    }
    else
    {
        updatedSearchRoiLeftWrist = cv::Rect2i(-1, -1, -1, -1);
        updatedPosLeftWrist = std::vector<int>{ frameIndex, -1, -1,-1,-1 };
    }
    /* right wrist */
    //std::cout << "RW" << std::endl;
    if (searchRoi[i][5].width > 0)
    {
        //std::cout << "search roi of RW, x=" << searchRoi[i][5].x << ", y=" << searchRoi[i][5].y << ", width=" << searchRoi[i][5].width << ", height=" << searchRoi[i][5].height << std::endl;
        threadJoints.emplace_back(&OpticalFlow::opticalFlow, this, frame, std::ref(frameIndex),
            std::ref(previousImg[i][counterTracker]), std::ref(searchRoi[i][5]), std::ref(previousMove[i][counterTracker]), std::ref(previousDIS[i][counterTracker]),
            std::ref(updatedImgRightWrist), std::ref(updatedSearchRoiRightWrist), std::ref(updatedDIS_rw), std::ref(moveRW), std::ref(updatedPosRightWrist));
        boolRightWrist = true;
        counterTracker++;
    }
    else
    {
        updatedSearchRoiRightWrist = cv::Rect2i(-1, -1, -1, -1);
        updatedPosRightWrist = std::vector<int>{ frameIndex, -1, -1,-1,-1 };
    }
    //std::cout << "all threads have started :: " << threadJoints.size() << std::endl;
    /* wait for all thread has finished */
    int counterThread = 0;
    if (!threadJoints.empty())
    {
        for (std::thread& thread : threadJoints)
        {
            thread.join();
            counterThread++;
        }
        //std::cout << counterThread << " threads have finished!" << std::endl;
    }
    else
    {
        //std::cout << "no thread has started" << std::endl;
        //std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
    //std::cout << i << "-th human's process has finished" << std::endl;
    /* combine all data and push data to queue */
    /* search roi */
    updatedSearchRoi.push_back(updatedSearchRoiLeftShoulder);
    updatedSearchRoi.push_back(updatedSearchRoiRightShoulder);
    updatedSearchRoi.push_back(updatedSearchRoiLeftElbow);
    updatedSearchRoi.push_back(updatedSearchRoiRightElbow);
    updatedSearchRoi.push_back(updatedSearchRoiLeftWrist);
    updatedSearchRoi.push_back(updatedSearchRoiRightWrist);
    updatedPositions.push_back(updatedPosLeftShoulder);
    updatedPositions.push_back(updatedPosRightShoulder);
    updatedPositions.push_back(updatedPosLeftElbow);
    updatedPositions.push_back(updatedPosRightElbow);
    updatedPositions.push_back(updatedPosLeftWrist);
    updatedPositions.push_back(updatedPosRightWrist);
    /* updated img */
    /* left shoulder */
    if (updatedSearchRoi[0].width > 0)
    {
        updatedImgJoints.push_back(updatedImgLeftShoulder);
        moveJoints.push_back(moveLS);
        disJoints.push_back(updatedDIS_ls);
    }
    /* right shoulder*/
    if (updatedSearchRoi[1].width > 0)
    {
        updatedImgJoints.push_back(updatedImgRightShoulder);
        moveJoints.push_back(moveRS);
        disJoints.push_back(updatedDIS_rs);
    }
    /*left elbow*/
    if (updatedSearchRoi[2].width > 0)
    {
        updatedImgJoints.push_back(updatedImgLeftElbow);
        moveJoints.push_back(moveLE);
        disJoints.push_back(updatedDIS_le);
    }
    /*right elbow */
    if (updatedSearchRoi[3].width > 0)
    {
        updatedImgJoints.push_back(updatedImgRightElbow);
        moveJoints.push_back(moveRE);
        disJoints.push_back(updatedDIS_re);
    }
    /* left wrist*/
    if (updatedSearchRoi[4].width > 0)
    {
        updatedImgJoints.push_back(updatedImgLeftWrist);
        moveJoints.push_back(moveLW);
        disJoints.push_back(updatedDIS_lw);
    }
    /*right wrist*/
    if (updatedSearchRoi[5].width > 0)
    {
        updatedImgJoints.push_back(updatedImgRightWrist);
        moveJoints.push_back(moveRW);
        disJoints.push_back(updatedDIS_rw);
    }
}

void OpticalFlow::opticalFlow(const cv::Mat1b frame, const int& frameIndex,
    cv::Mat1b& previousImg, cv::Rect2i& searchRoi, std::vector<float>& previousMove, cv::Ptr<cv::DISOpticalFlow>& dis,
    cv::Mat1b& updatedImg, cv::Rect2i& updatedSearchRoi, cv::Ptr<cv::DISOpticalFlow>& updatedDIS, std::vector<float>& updatedMove, std::vector<int>& updatedPos)
{
    // Calculate optical flow
    std::vector<uchar> status;
    std::vector<float> err;
    float deltaX_past = previousMove[0]; float deltaY_past = previousMove[1];
    //std::cout << "frame size=" << frame.size() << std::endl;
    //std::cout << "crop image :: searchRoi.x = " << searchRoi.x << ", y=" << searchRoi.y << ", width=" << searchRoi.width << ", height=" << searchRoi.height << std::endl;
    cv::Mat1b croppedImg = frame.clone();
    croppedImg = croppedImg(searchRoi);
    //std::stringstream fileNameStream;
    //fileNameStream << frameIndex << ".png";
    //std::string fileName = fileNameStream.str();
    //cv::imwrite(fileName, croppedImg);
    if (bool_dynamic_roi) //dynamic roi
    {
        if (croppedImg.rows != previousImg.rows || croppedImg.cols != previousImg.cols)
        {
            //std::cout << "resize image" << std::endl;
            if (croppedImg.rows != previousImg.rows && croppedImg.cols != previousImg.cols)
            {
                cv::Size size{ std::max(croppedImg.cols, previousImg.cols), std::max(croppedImg.rows, previousImg.rows) };
                cv::resize(croppedImg, croppedImg, size);
                cv::resize(previousImg, previousImg, size);
            }
            else if (croppedImg.rows == previousImg.rows && croppedImg.cols != previousImg.cols)
            {
                cv::Size size{ std::max(croppedImg.cols, previousImg.cols), croppedImg.rows };
                cv::resize(croppedImg, croppedImg, size);
                cv::resize(previousImg, previousImg, size);
            }
            else if (croppedImg.rows != previousImg.rows && croppedImg.cols == previousImg.cols)
            {
                cv::Size size{ croppedImg.cols,std::max(croppedImg.rows, previousImg.rows) };
                cv::resize(croppedImg, croppedImg, size);
                cv::resize(previousImg, previousImg, size);
            }
        }
    }
    else //static roi
    {
        //adjust roi img
        if (croppedImg.rows < roiWidthOF || croppedImg.cols < roiHeightOF) cv::resize(croppedImg, croppedImg, ROISize);
        if (previousImg.rows < roiWidthOF || previousImg.cols < roiHeightOF) cv::resize(previousImg, previousImg, ROISize);
    }

    // Calculate Optical Flow
    cv::Size size{ croppedImg.cols,croppedImg.rows };
    cv::Mat flow(size, CV_32FC2);
    // prepare matrix for saving dense optical flow
    //std::cout << "flos size=" << flow.size() << std::endl;
    //std::cout << "make dense optical flow storage, flow" << std::endl;
    //std::cout << "previousImg.size()" << previousImg.size() << ", croppedImg.size()~" << croppedImg.size() << std::endl;
    if (optflow_method == 0)
    {
        //cv::Mat prevs, next;
        //cv::cvtColor(previousImg, prevs, cv::COLOR_GRAY2BGR);
        //cv::cvtColor(croppedImg, next, cv::COLOR_GRAY2BGR);
        //cv::optflow::calcOpticalFlowSF(previousImg, croppedImg, flow, 3, 3, 6);
        //cv::optflow::calcOpticalFlowSparseToDense(previousImg, croppedImg, flow, 3, 16, 0.05f, true, 500, 1.5);//3, 5, 0.05, true, 500, 1.5
        if (!previousImg.isContinuous()) previousImg = previousImg.clone();
        if (!croppedImg.isContinuous()) croppedImg = croppedImg.clone();
        dis->calc(previousImg, croppedImg, flow);
    }
    else if (optflow_method == 1)
    {
        cv::calcOpticalFlowFarneback(previousImg, croppedImg, flow, 0.5, 2, 3, 3, 5, 1.2, 0); // calculate dense optical flow default :: 0.5, 2, 4, 4, 3, 1.6, cv::OPTFLOW_USE_INITIAL_FLOW :: uses the input flow as an initial flow approximation
    }
    else if (optflow_method == 2)
    {
        cv::Mat prevs, next;
        cv::cvtColor(previousImg, prevs, cv::COLOR_GRAY2BGR);
        cv::cvtColor(croppedImg, next, cv::COLOR_GRAY2BGR);
        cv::optflow::calcOpticalFlowDenseRLOF(prevs, next, flow, cv::Ptr<cv::optflow::RLOFOpticalFlowParameter>(), 1.f, cv::Size(3, 3),
            cv::optflow::InterpolationType::INTERP_EPIC, 16, 0.05f, 999.0f, 5, 20, true, 500.0f, 1.5f, false);
    }

    /*
     *void cv::calcOpticalFlowFarneback(
     *InputArray prev,         // Input previous frame (grayscale image)
     *InputArray next,         // Input next frame (grayscale image)
     *InputOutputArray flow,   // Output optical flow image (2-channel floating-point)
     *double pyr_scale,        // Image scale (< 1) to build pyramids
     *int levels,              // Number of pyramid levels
     *int winsize,             // Average window size ::presmoothing
     *int iterations,          // Number of iterations at each pyramid level
     *int poly_n,              // Polynomial expansion size
     *double poly_sigma,       // Standard deviation for Gaussian filter
     *int flags                // Operation flags
     *);
     */
     // calculate velocity

    float vecX = 0.;
    float vecY = 0.;
    float secondLargestX = 0.;
    float secondLargestY = 0.;
    int numPixels = 0;
    int numBackGround = 0;
    //for dense_vel_method == 5
    int num_pp = 0; // direction : positive-positive
    int num_np = 0; //direction : negative-positive
    int num_nn = 0; //direction : negative-negative
    int num_pn = 0; //direction : positive-negative
    float dot_pp, dot_np, dot_nn, dot_pn;
    float vecX_pp = 0; float vecY_pp = 0;
    float vecX_np = 0; float vecY_np = 0;
    float vecX_nn = 0; float vecY_nn = 0;
    float vecX_pn = 0; float vecY_pn = 0;
    //defining skip rate
    int rows = static_cast<int>(flow.rows / (skipPixel + 1)); // number of rows adapted pixels
    int cols = static_cast<int>(flow.cols / (skipPixel + 1)); // number of cols adapted pixels
    std::vector<float> velocities;
    std::vector<std::vector<float>> candidates_vel;
    float max = MIN_MOVE;
    //std::cout << "previousMove[0]=" << previousMove[0] << ", previousMove[1]=" << previousMove[1] << std::endl;
    for (int y = 1; y <= rows + 1; ++y)
    {
        //std::cout << std::min(std::max((y - 1) * (skipPixel + 1), 0), flow.rows - 1) << std::endl;
        for (int x = 1; x <= cols + 1; ++x)
        {
            //std::cout << "optical flow adapted point : y=" << (y - 1) * (skipPixel + 1) << ", x=" << (x - 1) * (skipPixel + 1) << std::endl;
            cv::Point2f flowVec = flow.at<cv::Point2f>(std::min(std::max((y - 1) * (skipPixel + 1), 0), flow.rows - 1), std::min(std::max((x - 1) * (skipPixel + 1), 0), flow.cols - 1)); // get velocity of position (y,x)
            // Access flowVec.x and flowVec.y for the horizontal and vertical components of velocity.
            if ((std::abs(flowVec.x) <= MIN_MOVE || std::abs(flowVec.y) <= MIN_MOVE) || (std::abs(flowVec.x) >= MAX_MOVE || std::abs(flowVec.y) >= MAX_MOVE) ||
                ((flowVec.x >= (-deltaX_past - epsironMove) && flowVec.x <= (-deltaX_past + epsironMove)) && (flowVec.y >= (-deltaY_past - epsironMove) && flowVec.y <= (-deltaY_past + epsironMove))))
            {
                numBackGround += 1;/* this may seem background optical flow */
            }
            else
            {
                //adapt average value
                if (dense_vel_method == 0)
                {
                    //std::cout << "flowVec.x=" << flowVec.x << ", flowVec.y=" << flowVec.y << std::endl;
                    vecX += flowVec.x;
                    vecY += flowVec.y;
                }
                //adapt max value
                else if (dense_vel_method == 1)
                {
                    if (std::pow(flowVec.x, 2) + std::pow(flowVec.y, 2) > max)
                    {
                        secondLargestX = vecX;
                        secondLargestY = vecY;
                        vecX = flowVec.x;
                        vecY = flowVec.y;
                        max = std::pow(flowVec.x, 2) + std::pow(flowVec.y, 2);
                    }
                }
                //adapt first, second, or third quarter value
                else if (dense_vel_method == 2 || dense_vel_method == 3 || dense_vel_method == 4)
                {
                    velocities.push_back(std::pow(flowVec.x, 2) + std::pow(flowVec.y, 2));
                    candidates_vel.push_back({ flowVec.x,flowVec.y });
                }
                else if (dense_vel_method == 5)
                {
                    //decide max value
                    if (flowVec.x >= 0 && flowVec.y >= 0)
                    {
                        num_pp++;
                        vecX_pp += flowVec.x;
                        vecY_pp += flowVec.y;
                    }
                    else if (flowVec.x < 0 && flowVec.y >= 0)
                    {
                        num_np++;
                        vecX_np += flowVec.x;
                        vecY_np += flowVec.y;
                    }
                    else if (flowVec.x < 0 && flowVec.y < 0)
                    {
                        num_nn++;
                        vecX_nn += flowVec.x;
                        vecY_nn += flowVec.y;
                    }
                    else if (flowVec.x >= 0 && flowVec.y < 0)
                    {
                        num_pn++;
                        vecX_pn += flowVec.x;
                        vecY_pn += flowVec.y;
                    }
                }
                numPixels += 1;
            }
        }
    }
    //std::cout << "background = " << numBackGround << ", adopted optical flow = " << numPixels << std::endl;
    //average method
    if (dense_vel_method == 0)
    {
        if (numPixels >= 1)
        {
            vecX /= numPixels;
            vecY /= numPixels;
        }
        else
        {
            vecX = 0;
            vecY = 0;
        }
    }
    else if (dense_vel_method == 1)
    {
        if (secondLargestX > 0)
        {
            vecX = secondLargestX;
            vecY = secondLargestY;
            //std::cout << "vecX=" << vecX << "vecY=" << vecY << std::endl;
        }
    }
    else if (dense_vel_method == 5)
    {
        //std::cout << "num_pp=" << num_pp << ", num_np=" << num_np << ", num_nn=" << num_nn << ", num_pn=" << num_pn << std::endl;
        int num_max = 0;
        if (num_pp > num_np)
            num_max = num_pp;
        else
            num_max = num_np;
        if (num_max < num_nn)
            num_max = num_nn;
        if (num_max < num_pn)
            num_max = num_pn;
        if (num_max > 0)
        {
            if (num_max / (numPixels) >= 0.5 && numPixels >= 4)
            {
                if (num_max == num_pp)
                {
                    vecX = vecX_pp / num_pp;
                    vecY = vecY_pp / num_pp;
                }
                else if (num_max == num_np)
                {
                    vecX = vecX_np / num_np;
                    vecY = vecY_np / num_np;
                }
                else if (num_max == num_nn)
                {
                    vecX = vecX_nn / num_nn;
                    vecY = vecY_nn / num_nn;
                }
                else if (num_max == num_pn)
                {
                    vecX = vecX_pn / num_pn;
                    vecY = vecY_pn / num_pn;
                }
            }
            else
            {
                vecX = (vecX_pp + vecX_np + vecX_nn + vecX_pn) / numPixels;
                vecY = (vecY_pp + vecY_np + vecY_nn + vecY_pn) / numPixels;
            }

        }
        else
        {
            vecX = 0;
            vecY = 0;
        }
    }
    if (!velocities.empty())
    {
        //median value
        if (dense_vel_method == 2 || (dense_vel_method == 3 && velocities.size() <= 3) || (dense_vel_method == 4 && velocities.size() <= 3))
        {
            float median = calculateMedian(velocities);
            // Iterate over the vector and find indices with the specified value
            auto it = std::find(velocities.begin(), velocities.end(), median);
            size_t index = std::distance(velocities.begin(), it);
            vecX = candidates_vel[index][0];
            vecY = candidates_vel[index][1];
        }
        //third quarter value
        else if (dense_vel_method == 3 && velocities.size() >= 4)
        {
            float thirdQuarter = calculateThirdQuarter(velocities);
            // Iterate over the vector and find indices with the specified value
            auto it = std::find(velocities.begin(), velocities.end(), thirdQuarter);
            size_t index = std::distance(velocities.begin(), it);
            vecX = candidates_vel[index][0];
            vecY = candidates_vel[index][1];
            //std::cout << "third quarter value :: vx = " << vecX << ", vy=" << vecY << std::endl;
        }
        //first quarter value
        else if (dense_vel_method == 4 && velocities.size() >= 4)
        {
            float firstQuarter = calculateFirstQuarter(velocities);
            // Iterate over the vector and find indices with the specified value
            auto it = std::find(velocities.begin(), velocities.end(), firstQuarter);
            size_t index = std::distance(velocities.begin(), it);
            vecX = candidates_vel[index][0];
            vecY = candidates_vel[index][1];
        }
    }
    if (vecX >= 0) vecX = vecX + 0.5;//if bigger than 0.5, adopt as 1
    else if (vecX < 0) vecX = vecX - 0.5;//if bigger than 0.5, adopt as 1
    if (vecY >= 0) vecY = vecY + 0.5;//if bigger than 0.5, adopt as 1
    else if (vecY < 0) vecY = vecY - 0.5;//if bigger than 0.5, adopt as 1
    //std::cout << "after adding 0.5 :: vx = " << vecX << ", vy=" << vecY << std::endl;
    if ((std::pow(vecX, 2) + std::pow(vecY, 2) >= MoveThreshold) && (std::pow(vecX, 2) + std::pow(vecY, 2) <= MAX_MOVE * MAX_MOVE))
    {
        float vecX_future = alpha * vecX + (1 - alpha) * deltaX_past;
        float vecY_future = alpha * vecY + (1 - alpha) * deltaY_past;
        if (vecX_future < 0)
            vecX_future = std::max(vecX_future, -max_vel);
        else
            vecX_future = std::min(vecX_future, max_vel);
        if (vecY_future < 0)
            vecY_future = std::max(vecY_future, -max_vel);
        else
            vecY_future = std::min(vecY_future, max_vel);
        int updatedLeft = searchRoi.x + static_cast<int>(vecX_future);
        int updatedTop = searchRoi.y + static_cast<int>(vecY_future);
        int left = std::min(std::max(updatedLeft, 0), frame.cols);
        int top = std::min(std::max(updatedTop, 0), frame.rows);
        int right = std::max(std::min(left + previousImg.cols, frame.cols), 0);
        int bottom = std::max(std::min(top + previousImg.rows, frame.rows), 0);
        cv::Rect2i roi(left, top, right - left, bottom - top);
        //tracking was successful
        if (roi.width >= MIN_SEARCH && roi.height >= MIN_SEARCH)
        {
            //std::cout << "roi.x=" << roi.x << ", roi.y=" << roi.y << ", roi.width=" << roi.width << ", roi.height=" << roi.height << std::endl;
            if (bool_moveROI) //dynamic roi :: move ROI according to the current motoin
            {
                updatedSearchRoi = roi;
                updatedMove = { vecX_future,vecY_future };//for not moving roi  : std::vector<float>{0.0,0.0};// for moving search roi:: { vecX, vecY };
                //Update the previous frame and previous points
                updatedImg = croppedImg; //for not moving roi : previousImg;//for moving search roi :: adopt croppedImg.clone();
                updatedDIS = dis;
                updatedPos = std::vector<int>{ frameIndex, roi.x,roi.y,roi.width,roi.height };
            }
            else //static roi
            {
                updatedSearchRoi = searchRoi;
                //updatedMove = { 0.0,0.0 };
                if (bool_check_pos) //if check move dists
                {
                    deltaX_past += vecX;
                    deltaY_past += vecY;
                    updatedMove = { deltaX_past,deltaY_past };//for not moving roi  : std::vector<float>{0.0,0.0};// for moving search roi:: { vecX, vecY };
                }
                else
                    updatedMove = { 0.0,0.0 };//for not moving roi  : std::vector<float>{0.0,0.0};// for moving search roi:: { vecX, vecY };
                updatedImg = croppedImg;//update reference img
                updatedPos = std::vector<int>{ frameIndex,  roi.x,roi.y,roi.width,roi.height };
                updatedDIS = dis;
            }
        }
        //out of range
        else
        {
            updatedSearchRoi.x = -1;
            updatedSearchRoi.y = -1;
            updatedSearchRoi.width = -1;
            updatedSearchRoi.height = -1;
            updatedPos = std::vector<int>{ frameIndex, -1, -1 ,-1,-1 };
        }
    }
    /* not move -> tracking was failed */
    else
    {
        updatedSearchRoi.x = -1;
        updatedSearchRoi.y = -1;
        updatedSearchRoi.width = -1;
        updatedSearchRoi.height = -1;
        updatedPos = std::vector<int>{ frameIndex, -1, -1 ,-1,-1 };
    }
}

float OpticalFlow::calculateMedian(std::vector<float> vec) {
    // Check if the vector is not empty
    if (vec.empty()) {
        throw std::invalid_argument("Vector is empty");
    }

    // Calculate the middle index
    size_t size = vec.size();
    size_t middleIndex = size / 2;

    // Use std::nth_element to find the median
    std::nth_element(vec.begin(), vec.begin() + middleIndex, vec.end());

    return vec[middleIndex];
}

float OpticalFlow::calculateThirdQuarter(std::vector<float> vec) {
    // Check if the vector is not empty
    if (vec.empty()) {
        throw std::invalid_argument("Vector is empty");
    }

    // Calculate the middle index
    size_t size = vec.size();
    size_t middleIndex = static_cast<size_t>(size * (3 / 4));

    // Use std::nth_element to find the median
    std::nth_element(vec.begin(), vec.begin() + middleIndex, vec.end());

    return vec[middleIndex];
}

float OpticalFlow::calculateFirstQuarter(std::vector<float> vec) {
    // Check if the vector is not empty
    if (vec.empty()) {
        throw std::invalid_argument("Vector is empty");
    }

    // Calculate the middle index
    size_t size = vec.size();
    size_t middleIndex = static_cast<size_t>(size * (1 / 4));

    // Use std::nth_element to find the median
    std::nth_element(vec.begin(), vec.begin() + middleIndex, vec.end());

    return vec[middleIndex];
}