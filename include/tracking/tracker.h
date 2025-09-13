#pragma once

#ifndef Template_H
#define Template_H

#include "stdafx.h"
#include "global_parameters.h"
#include "utility.h"
#include "mosse.h"

extern const float DIF_THRESHOLD;
extern const float MIN_MOVE; //minimum opticalflow movement
extern const int roiSize_wrist;
extern const int roiSize_elbow;
extern const int roiSize_shoulder;
extern const int dense_vel_method;
extern const double threshold_mosse;
extern const bool boolChange; //whether change tracker or not
//save file
extern const std::string file_of_left;
extern const std::string file_of_right;

extern std::queue<std::array<cv::Mat1b, 2>> queueFrame;
extern std::queue<int> queueFrameIndex;

/* left */
extern std::queue<std::vector<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>>> queueYoloTracker_left;      // queue for old image for optical flow. vector size is [num human,6]
extern std::queue<std::vector<std::vector<cv::Mat1b>>> queueYoloTemplate_left; // queue for yolo template       // queue for old image for optical flow. vector size is [num human,6]
extern std::queue<std::vector<std::vector<cv::Rect2i>>> queueYoloRoi_left;        // queue for search roi for optical flow. vector size is [num human,6]
extern std::queue<std::vector<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>>> queueMosseTracker_left;
extern std::queue<std::vector<std::vector<cv::Rect2i>>> queueRoi_left;          // queue for search roi for optical flow. vector size is [num human,6]
extern std::queue<std::vector<std::vector<cv::Mat1b>>> queueTemplate_left;
extern std::queue<std::vector<std::vector<bool>>> queueScale_left; //search area scale
extern std::queue<std::vector<std::vector<std::vector<int>>>> queueMoveLeft; //queue for saving previous move

/* right */
extern std::queue<std::vector<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>>> queueYoloTracker_right;      // queue for old image for optical flow. vector size is [num human,6]
extern std::queue<std::vector<std::vector<cv::Mat1b>>> queueYoloTemplate_right;      // queue for old image for optical flow. vector size is [num human,6]
extern std::queue<std::vector<std::vector<cv::Rect2i>>> queueYoloRoi_right;        // queue for search roi for optical flow. vector size is [num human,6]
extern std::queue<std::vector<std::vector<cv::Rect2i>>> queueRoi_right;          // queue for search roi for optical flow. vector size is [num human,6]
extern std::queue<std::vector<std::vector<cv::Mat1b>>> queueTemplate_right;
extern std::queue<std::vector<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>>> queueMosseTracker_right;
extern std::queue<std::vector<std::vector<bool>>> queueScale_right; //search area scale
extern std::queue<std::vector<std::vector<std::vector<int>>>> queueMoveRight; //queue for saving previous move

class TemplateMatching
{
private:
    //const cv::Size ROISize{ roiWidthOF, roiHeightOF };
    const int originalWidth = 640;
    const int originalHeight = 640;
    const float MAX_CHANGE = 100; //max change of joints per frame
    const bool bool_moveROI = false;
    const float alpha = 0.3; //predict velocity :: alpha*v_t+(1-alpha)*v_(t-1)
    // template matching constant value setting
    const double scaleXTM = 1.3; //2.0 //for throwing //1.5 //for switching // search area scale compared to roi
    const double scaleYTM = 1.3;
    const double scaleXYolo = 2.5; //3.5 //for throwing //2.0 //for switching
    const double scaleYYolo = 2.5; //3.5 //for throwing
    const double matchingThreshold = 0.7;             // matching threshold
    const int MATCHINGMETHOD = cv::TM_SQDIFF_NORMED; // //cv::TM_SQDIFF_NORMED -> unique background, cv::TM_CCOEFF_NORMED :: ‘ŠŠÖŒW”, cv::TM_CCORR_NORMED -> patterned background // TM_SQDIFF_NORMED is good for small template
    const double MoveThreshold = 0.0;                 // move threshold of objects

    bool bool_multithread = false; //done -> false if run MOSSE in multiple threads
    bool bool_iouCheck = true; //done -> true;check current tracker iou -> if overlapped
    std::vector<int> defaultMove{ 0, 0 };
    const double gamma = 0.3; //ration of current and past for tracker velocity, the larger, more important current :: gamma*current+(1-gamma)*past
    const double MIN_SEARCH = 10;
    const int MAX_VEL = 20; // max roi move 
    const bool bool_dynamicScale = true; //done -> true :: scaleXTM ::smaller is good //dynamic scale according to current motion
    const bool bool_TBD = false; //done-> false //if true : update position too
    const bool bool_kf = true; //done->true
    const bool bool_skip = true; //skip updating for occlusion and switching prevention
    const bool bool_check_psr = true; //done->true //which tracker adopt : detection or tracking
    const double min_keep_psr = 7.0;
public:

    float t_elapsed = 0;

    TemplateMatching()
    {
        std::cout << "construct Template class" << std::endl;
    }

    void main()
    {
        /* construction of class */
        Utility utof;
        /* prepare storage */
        std::vector<std::vector<std::vector<std::vector<int>>>> posSaver_left; //[sequence,numHuman,numJoints,position] :{frameIndex,xCenter,yCenter}
        std::vector<std::vector<std::vector<std::vector<int>>>> posSaver_right; //[sequence,numHuman,numJoints,position] :{frameIndex,xCenter,yCenter}
        posSaver_left.reserve(2000);
        posSaver_right.reserve(2000);
        int countTM = 0;
        int counterStart = 0;
        while (true)
        {
            if (counterStart == 2) break;
            if (!queueYoloRoi_left.empty() || !queueYoloRoi_right.empty())
            {
                counterStart++;
                int countIteration = 1;
                while (true)
                {
                    if (queueYoloRoi_left.empty() && queueYoloRoi_right.empty()) break;
                    //std::cout << countIteration << ":: remove yolo data ::" << std::endl;
                    if (counterStart == 2) break;
                    countIteration++;
                    if (!queueYoloTemplate_left.empty()) queueYoloTemplate_left.pop();
                    if (!queueYoloTemplate_right.empty()) queueYoloTemplate_right.pop();
                    if (!queueYoloRoi_left.empty()) queueYoloRoi_left.pop();
                    if (!queueYoloRoi_right.empty()) queueYoloRoi_right.pop();
                }
            }
            if (!queueFrame.empty()) std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        std::cout << "start tracking" << std::endl;
        /* frame is available */
        int counter = 1;
        int counterFinish = 0;
        while (true)
        {
            if (counterFinish == 10)

            {
                break;
            }
            if (queueFrame.empty())
            {
                counterFinish++;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            /* frame available */
            else if (!queueFrame.empty())
            {
                if (!queueYoloRoi_left.empty() || !queueYoloRoi_right.empty() || !queueTemplate_left.empty() || !queueTemplate_right.empty()) //some Template exist
                {
                    /* get images from queue */
                    std::array<cv::Mat1b, 2> frames;
                    int frameIndex;
                    auto start = std::chrono::high_resolution_clock::now();
                    utof.getImages(frames, frameIndex);
                    cv::Mat1b frame_left = frames[0];
                    cv::Mat1b frame_right = frames[1];
                    if (frame_left.rows > 0 && frame_right.rows > 0)
                    {
                        std::thread thread_left(&TemplateMatching::iteration, this, std::ref(frame_left), std::ref(frameIndex), std::ref(posSaver_left),std::ref(queueYoloTracker_left),
                            std::ref(queueYoloTemplate_left), std::ref(queueYoloRoi_left), std::ref(queueMosseTracker_left),std::ref(queueTemplate_left), std::ref(queueRoi_left), 
                            std::ref(queueScale_left), std::ref(queueMoveLeft),std::ref(queueTriangulation_left));
                        iteration(frame_right, frameIndex, posSaver_right, queueYoloTracker_right,queueYoloTemplate_right, queueYoloRoi_right, 
                            queueMosseTracker_right, queueTemplate_right, queueRoi_right, queueScale_right, queueMoveRight,queueTriangulation_right);
                        //std::cout << "both OF threads have started" << std::endl;
                        thread_left.join();
                        auto stop = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                        t_elapsed = t_elapsed + static_cast<float>(duration.count());
                        countTM++;
                        //std::cout << "***** posSaver_left size=" << posSaver_left.size() << ", posSaver_right size=" << posSaver_right.size() << "********" << std::endl;
                        std::cout << " Time taken by Template Matching : " << duration.count() << " microseconds" << std::endl;
                    }
                }
            }
        }
        std::cout << "TemplateMatching" << std::endl;
        std::cout << " process speed :: " << static_cast<int>(countTM / t_elapsed * 1000000) << " Hz for " << countTM << " cycles" << std::endl;
        std::cout << "*** LEFT *** posSaver_left.size()=" << posSaver_left.size() << std::endl;
        utof.save(posSaver_left, file_of_left);
        std::cout << "*** RIGHT *** posSaver_right.size()=" << posSaver_right.size() << std::endl;
        utof.save(posSaver_right, file_of_right);
    }

    void iteration(cv::Mat1b& frame, int& frameIndex, std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver,
        std::queue<std::vector<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>>>& queueYoloTracker, std::queue<std::vector<std::vector<cv::Mat1b>>>& queueYoloTemplate, std::queue<std::vector<std::vector<cv::Rect2i>>>& queueYoloRoi,
        std::queue<std::vector<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>>>& queueMosseTracker, std::queue<std::vector<std::vector<cv::Mat1b>>>& queueTemplate, std::queue<std::vector<std::vector<cv::Rect2i>>>& queueRoi,
        std::queue<std::vector<std::vector<bool>>>& queueScale, std::queue<std::vector<std::vector<std::vector<int>>>>& queueMove, std::queue<std::vector<std::vector<std::vector<int>>>>& queueTriangulation)
    {
        /* optical flow process for each joints */
        std::vector<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>> previousTracker;
        std::vector<std::vector<cv::Mat1b>> previousTemplate;           //[number of human,0~6,cv::Mat1b]
        std::vector<std::vector<cv::Rect2i>> searchRoi;            //[number of human,6,cv::Rect2i], if Template was failed, roi.x == -1
        std::vector<std::vector<bool>> searchScales; //search Roi scales
        std::vector<std::vector<std::vector<int>>> previousMove;
        //auto start_pre = std::chrono::high_resolution_clock::now();
        getPreviousData(previousTracker, previousTemplate, searchRoi, searchScales, previousMove, queueYoloTracker,queueYoloTemplate, queueYoloRoi, queueMosseTracker, queueTemplate, queueRoi, queueScale,queueMove);
        //auto stop_pre = std::chrono::high_resolution_clock::now();
        //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_pre - start_pre);
        //std::cout << " Time taken by prepare data : " << duration.count() << " microseconds" << std::endl;
        //std::cout << "finish getting previous data " << std::endl;
        /* start optical flow process */
        /* for every human */
        std::vector<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>> updatedTrackerHuman;
        std::vector<std::vector<cv::Mat1b>> updatedTemplateHuman;
        std::vector<std::vector<cv::Rect2i>> updatedRoiHuman;
        std::vector<std::vector<bool>> updatedScalesHuman;
        std::vector<std::vector<std::vector<int>>> updatedMoveHuman;
        std::vector<std::vector<std::vector<int>>> updatedPositionsHuman;
        /*std::cout << "human =" << searchRoi.size() << " !!!!!!!!!!!!" << std::endl;
        if (!searchRoi.empty())
        {
            for (cv::Rect2i& roi : searchRoi[0])
                std::cout << roi.x << "," << roi.y << "," << roi.width << "," << roi.height << std::endl;
            std::cout << "previousMove.size()=" << previousMove.size() << ", previousImg.size()=" << previousImg.size() << std::endl;
        }*/
        if (!searchRoi.empty())
        {
            for (int i = 0; i < searchRoi.size(); i++)//for every human
            {
                //std::cout << i << "-th human::" << "previousMove.size()=" << previousMove[i].size() << std::endl;
                //std::cout<<"previousImg.size()=" << previousImg[i].size() << std::endl;
                /* for every joints */
                std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>> updatedTrackerJoints;
                std::vector<cv::Mat1b> updatedTemplateJoints;
                std::vector<cv::Rect2i> updatedSearchRoi;
                std::vector<bool> updatedScalesJoints; //roi movement
                std::vector<std::vector<int>> updatedMoveJoints;
                std::vector<std::vector<int>> updatedPositions;
                std::vector<int> updatedPosLeftShoulder, updatedPosRightShoulder, updatedPosLeftElbow, updatedPosRightElbow, updatedPosLeftWrist, updatedPosRightWrist;
                cv::Ptr<cv::mytracker::TrackerMOSSE> updatedTrackerLeftShoulder, updatedTrackerRightShoulder, updatedTrackerLeftElbow, updatedTrackerRightElbow, updatedTrackerLeftWrist, updatedTrackerRightWrist;
                cv::Mat1b updatedTemplateLeftShoulder, updatedTemplateRightShoulder, updatedTemplateLeftElbow, updatedTemplateRightElbow, updatedTemplateLeftWrist, updatedTemplateRightWrist;
                cv::Rect2i updatedSearchRoiLeftShoulder, updatedSearchRoiRightShoulder, updatedSearchRoiLeftElbow, updatedSearchRoiRightElbow, updatedSearchRoiLeftWrist, updatedSearchRoiRightWrist;
                std::vector<int> updatedMoveLeftShoulder, updatedMoveRightShoulder, updatedMoveLeftElbow, updatedMoveRightElbow, updatedMoveLeftWrist, updatedMoveRightWrist;
                bool updatedScaleLeftShoulder, updatedScaleRightShoulder, updatedScaleLeftElbow, updatedScaleRightElbow, updatedScaleLeftWrist, updatedScaleRightWrist;
                bool boolLeftShoulder = false;
                bool boolRightShoulder = false;
                bool boolLeftElbow = false;
                bool boolRightElbow = false;
                bool boolLeftWrist = false;
                bool boolRightWrist = false;
                std::vector<std::thread> threadJoints;
                /* start optical flow process for each joints */
                /* left shoulder */
                int counterTemplate = 0;
                //std::cout << "LS" << std::endl;
                //auto start_mosse = std::chrono::high_resolution_clock::now();
                if (searchRoi[i][0].width > MIN_SEARCH)
                {
                    bool boolScale = searchScales[i][counterTemplate];
                    threadJoints.emplace_back(&TemplateMatching::track, this, std::ref(frame), std::ref(frameIndex),
                       std::ref(previousTracker[i][counterTemplate]), std::ref(previousTemplate[i][counterTemplate]), std::ref(searchRoi[i][0]), std::ref(boolScale),std::ref(previousMove[i][counterTemplate]),
                       std::ref(updatedTrackerLeftShoulder),std::ref(updatedTemplateLeftShoulder), std::ref(updatedSearchRoiLeftShoulder), std::ref(updatedScaleLeftShoulder),std::ref(updatedMoveLeftShoulder), std::ref(updatedPosLeftShoulder));
                    boolLeftShoulder = true;
                    counterTemplate++;
                }
                else
                {
                    updatedSearchRoiLeftShoulder = cv::Rect2i(-1, -1, -1, -1);
                    updatedPosLeftShoulder = std::vector<int>{ frameIndex, -1, -1 ,-1,-1};
                }
                /* right shoulder */
                //std::cout << "RS" << std::endl;
                if (searchRoi[i][1].width > MIN_SEARCH)
                {
                    bool boolScale = searchScales[i][counterTemplate];
                    threadJoints.emplace_back(&TemplateMatching::track, this, std::ref(frame), std::ref(frameIndex),
                        std::ref(previousTracker[i][counterTemplate]),std::ref(previousTemplate[i][counterTemplate]), std::ref(searchRoi[i][1]), std::ref(boolScale),std::ref(previousMove[i][counterTemplate]),
                        std::ref(updatedTrackerRightShoulder), std::ref(updatedTemplateRightShoulder), std::ref(updatedSearchRoiRightShoulder), std::ref(updatedScaleRightShoulder), std::ref(updatedMoveRightShoulder),std::ref(updatedPosRightShoulder));
                    boolRightShoulder = true;
                    counterTemplate++;
                }
                else
                {
                    updatedSearchRoiRightShoulder = cv::Rect2i(-1, -1, -1, -1); 
                    updatedPosRightShoulder = std::vector<int>{ frameIndex, -1, -1 ,-1,-1 };
                }
                /* left elbow */
                //std::cout << "LE" << std::endl;
                if (searchRoi[i][2].width > MIN_SEARCH)
                {
                    bool boolScale = searchScales[i][counterTemplate];
                    threadJoints.emplace_back(&TemplateMatching::track, this, std::ref(frame), std::ref(frameIndex),
                        std::ref(previousTracker[i][counterTemplate]), std::ref(previousTemplate[i][counterTemplate]), std::ref(searchRoi[i][2]), std::ref(boolScale), std::ref(previousMove[i][counterTemplate]),
                        std::ref(updatedTrackerLeftElbow), std::ref(updatedTemplateLeftElbow), std::ref(updatedSearchRoiLeftElbow), std::ref(updatedScaleLeftElbow), std::ref(updatedMoveLeftElbow), std::ref(updatedPosLeftElbow));
                    boolLeftElbow = true;
                    counterTemplate++;
                }
                else
                {
                    updatedSearchRoiLeftElbow = cv::Rect2i(-1, -1, -1, -1); 
                    updatedPosLeftElbow = std::vector<int>{ frameIndex, -1, -1 ,-1,-1 };
                }
                /* right elbow */
                //std::cout << "RE" << std::endl;
                if (searchRoi[i][3].width > MIN_SEARCH)
                {
                    bool boolScale = searchScales[i][counterTemplate];
                    threadJoints.emplace_back(&TemplateMatching::track, this, std::ref(frame), std::ref(frameIndex),
                        std::ref(previousTracker[i][counterTemplate]), std::ref(previousTemplate[i][counterTemplate]), std::ref(searchRoi[i][3]), std::ref(boolScale), std::ref(previousMove[i][counterTemplate]),
                        std::ref(updatedTrackerRightElbow), std::ref(updatedTemplateRightElbow), std::ref(updatedSearchRoiRightElbow), std::ref(updatedScaleRightElbow), std::ref(updatedMoveRightElbow), std::ref(updatedPosRightElbow));
                    boolRightElbow = true;
                    counterTemplate++;
                }
                else
                {
                    updatedSearchRoiRightElbow = cv::Rect2i(-1, -1, -1, -1); 
                    updatedPosRightElbow = std::vector<int>{ frameIndex, -1, -1 ,-1,-1 };
                }
                /* left wrist */
                //std::cout << "LW" << std::endl;
                if (searchRoi[i][4].width > MIN_SEARCH)
                {
                    bool boolScale = searchScales[i][counterTemplate];
                    threadJoints.emplace_back(&TemplateMatching::track, this, std::ref(frame), std::ref(frameIndex),
                        std::ref(previousTracker[i][counterTemplate]), std::ref(previousTemplate[i][counterTemplate]), std::ref(searchRoi[i][4]), std::ref(boolScale), std::ref(previousMove[i][counterTemplate]),
                        std::ref(updatedTrackerLeftWrist), std::ref(updatedTemplateLeftWrist), std::ref(updatedSearchRoiLeftWrist), std::ref(updatedScaleLeftWrist),std::ref(updatedMoveLeftWrist),std::ref(updatedPosLeftWrist));
                    boolLeftWrist = true;
                    counterTemplate++;
                }
                else
                {
                    updatedSearchRoiLeftWrist = cv::Rect2i(-1, -1, -1, -1); 
                    updatedPosLeftWrist = std::vector<int>{ frameIndex, -1, -1 ,-1,-1 };
                }
                /* right wrist */
                //std::cout << "RW" << std::endl;
                if (searchRoi[i][5].width > MIN_SEARCH)
                {
                    bool boolScale = searchScales[i][counterTemplate];
                    //std::cout << "search roi of RW, x=" << searchRoi[i][5].x << ", y=" << searchRoi[i][5].y << ", width=" << searchRoi[i][5].width << ", height=" << searchRoi[i][5].height << std::endl;
                    threadJoints.emplace_back(&TemplateMatching::track, this, std::ref(frame), std::ref(frameIndex),
                        std::ref(previousTracker[i][counterTemplate]), std::ref(previousTemplate[i][counterTemplate]), std::ref(searchRoi[i][5]), std::ref(boolScale), std::ref(previousMove[i][counterTemplate]),
                        std::ref(updatedTrackerRightWrist), std::ref(updatedTemplateRightWrist), std::ref(updatedSearchRoiRightWrist), std::ref(updatedScaleRightWrist), std::ref(updatedMoveRightWrist), std::ref(updatedPosRightWrist));
                    boolRightWrist = true;
                    counterTemplate++;
                }
                else
                {
                    updatedSearchRoiRightWrist = cv::Rect2i(-1, -1, -1, -1); 
                    updatedPosRightWrist = std::vector<int>{ frameIndex, -1, -1 ,-1,-1 };
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
                    std::cout << counterThread << " threads have finished!" << std::endl;
                    //auto stop_mosse = std::chrono::high_resolution_clock::now();
                    //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_mosse - start_mosse);
                    //std::cout << " Time taken by tracking : " << duration.count() << " microseconds" << std::endl;
                }
                else
                {
                    std::cout << "no thread has started" << std::endl;
                    //std::this_thread::sleep_for(std::chrono::milliseconds(30));
                }
                //std::cout << i << "-th human" << std::endl;
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
                //std::cout << "push roi data" << std::endl;
                /* updated img */
                /* left shoulder */
                if (updatedSearchRoi[0].width > MIN_SEARCH)
                {
                    updatedTrackerJoints.push_back(updatedTrackerLeftShoulder);
                    updatedTemplateJoints.push_back(updatedTemplateLeftShoulder);
                    updatedScalesJoints.push_back(updatedScaleLeftShoulder);
                    updatedMoveJoints.push_back(updatedMoveLeftShoulder);
                }
                /* right shoulder*/
                if (updatedSearchRoi[1].width > MIN_SEARCH)
                {
                    updatedTrackerJoints.push_back(updatedTrackerRightShoulder);
                    updatedTemplateJoints.push_back(updatedTemplateRightShoulder);
                    updatedScalesJoints.push_back(updatedScaleRightShoulder);
                    updatedMoveJoints.push_back(updatedMoveRightShoulder);
                }
                /*left elbow*/
                if (updatedSearchRoi[2].width > MIN_SEARCH)
                {
                    updatedTrackerJoints.push_back(updatedTrackerLeftElbow);
                    updatedTemplateJoints.push_back(updatedTemplateLeftElbow);
                    updatedScalesJoints.push_back(updatedScaleLeftElbow);
                    updatedMoveJoints.push_back(updatedMoveLeftElbow);
                }
                /*right elbow */
                if (updatedSearchRoi[3].width > MIN_SEARCH)
                {
                    updatedTrackerJoints.push_back(updatedTrackerRightElbow);
                    updatedTemplateJoints.push_back(updatedTemplateRightElbow);
                    updatedScalesJoints.push_back(updatedScaleRightElbow);
                    updatedMoveJoints.push_back(updatedMoveRightElbow);
                }
                /* left wrist*/
                if (updatedSearchRoi[4].width > MIN_SEARCH)
                {
                    //std::stringstream fileNameStream;
                    //fileNameStream << "leftWrist-" << frameIndex << ".jpg";
                    //std::string fileName = fileNameStream.str();
                    //cv::imwrite(fileName, frame(updatedSearchRoi[4]));
                    updatedTrackerJoints.push_back(updatedTrackerLeftWrist);
                    updatedTemplateJoints.push_back(updatedTemplateLeftWrist);
                    updatedScalesJoints.push_back(updatedScaleLeftWrist);
                    updatedMoveJoints.push_back(updatedMoveLeftWrist);
                    //std::cout << " !! LEFT WRIST :: SUCCESS !! left= " << updatedSearchRoi[4].x << ", top=" << updatedSearchRoi[4].y << std::endl;
                }
                /*right wrist*/
                if (updatedSearchRoi[5].width > MIN_SEARCH)
                {
                    updatedTrackerJoints.push_back(updatedTrackerRightWrist);
                    updatedTemplateJoints.push_back(updatedTemplateRightWrist);
                    updatedScalesJoints.push_back(updatedScaleRightWrist);
                    updatedMoveJoints.push_back(updatedMoveRightWrist);
                    //std::cout << " !! RIGHT WRIST :: SUCCESS !! left= " << updatedSearchRoi[5].x << ", top=" << updatedSearchRoi[5].y << std::endl;
                }
                //std::cout << "finishing pushing data" << std::endl;
                /* combine all data for one human */
                updatedRoiHuman.push_back(updatedSearchRoi);
                updatedPositionsHuman.push_back(updatedPositions);
                if (!updatedTemplateJoints.empty())
                {
                    updatedTrackerHuman.push_back(updatedTrackerJoints);
                    updatedTemplateHuman.push_back(updatedTemplateJoints);
                    updatedScalesHuman.push_back(updatedScalesJoints);
                    updatedMoveHuman.push_back(updatedMoveJoints);
                }
            }
            /* push updated data to queue */
            queueRoi.push(updatedRoiHuman);
            if (!updatedTemplateHuman.empty() && !updatedTrackerHuman.empty())
            {
                queueMosseTracker.push(updatedTrackerHuman);
                queueTemplate.push(updatedTemplateHuman);
                queueScale.push(updatedScalesHuman);
                queueMove.push(updatedMoveHuman);
            }
            //std::cout << "save to posSaver! posSaver size=" << posSaver.size() << std::endl;

            /* arrange posSaver */
            std::vector<int> pastData;
            if (!posSaver.empty())
            {
                //auto start = std::chrono::high_resolution_clock::now();
                std::vector<std::vector<std::vector<int>>> all; //all human data
                // for each human
                for (int i = 0; i < updatedPositionsHuman.size(); i++)
                {
                    std::vector<std::vector<int>> tempHuman;
                    /* same human */
                    if (posSaver.back().size() > i)
                    {
                        //std::cout << "updatedPosition.size()=" << updatedPositionsHuman[i].size() << std::endl;
                        // for each joint
                        for (int j = 0; j < updatedPositionsHuman[i].size(); j++)
                        {
                            //std::cout << "updatedPosition 0=" << updatedPositionsHuman[i][j] << std::endl;;
                            // detected
                            if (updatedPositionsHuman[i][j][1] > 0)
                            {
                                if (posSaver.back()[i][j][1] > 0 && posSaver.back()[i][j][2] > 0) //previous data exists
                                {
                                    //boundary check for updated data 
                                    if ((float)(std::abs((posSaver.back()[i][j][1] + posSaver.back()[i][j][3] /2)- (updatedPositionsHuman[i][j][1]+ updatedPositionsHuman[i][j][3]/2))) < MAX_CHANGE && 
                                        (float)(std::abs((posSaver.back()[i][j][2]+ posSaver.back()[i][j][4]/2) - (updatedPositionsHuman[i][j][2]+ updatedPositionsHuman[i][j][4]/2))) < MAX_CHANGE)
                                    {
                                        //std::cout << "CORRECT :: dif_x = " << std::abs(posSaver.back()[i][j][1] - updatedPositionsHuman[i][j][1]) << ", diff_y=" << std::abs(posSaver.back()[i][j][2] - updatedPositionsHuman[i][j][2]) << std::endl;
                                        tempHuman.push_back(updatedPositionsHuman[i][j]);
                                    }
                                    else
                                    {
                                        //std::cout << "!!!!!! OUT OF BOUNDARY !!!!! dif_x = " << std::abs(posSaver.back()[i][j][1] - updatedPositionsHuman[i][j][1]) << ", diff_y=" << std::abs(posSaver.back()[i][j][2] - updatedPositionsHuman[i][j][2]) << std::endl;
                                        pastData = posSaver.back()[i][j];
                                        pastData[0] = updatedPositionsHuman[i][j][0];
                                        //std::cout << "pastData = " << pastData[0] << "," << pastData[1] << ", " << pastData[2] << std::endl;
                                        tempHuman.push_back(pastData); //adopt last detection
                                    }
                                }
                                else //no previous data
                                {
                                    tempHuman.push_back(updatedPositionsHuman[i][j]);
                                }
                            }
                            // not detected
                            else
                            {
                                // already detected
                                if (posSaver.back()[i][j][1] > 0)
                                {
                                    pastData = posSaver.back()[i][j];
                                    pastData[0] = updatedPositionsHuman[i][j][0];
                                    tempHuman.push_back(pastData); //adopt last detection
                                }
                                // not detected yet
                                else
                                    tempHuman.push_back(updatedPositionsHuman[i][j]); //-1
                            }
                        }
                    }
                    //new human
                    else
                        tempHuman = updatedPositionsHuman[i];
                    all.push_back(tempHuman); //push human data
                }
                posSaver.push_back(all);
                if (!queueTriangulation.empty()) queueTriangulation.pop();
                queueTriangulation.push(all);
                //std::cout << "pushing all data" << std::endl;
                //auto stop = std::chrono::high_resolution_clock::now();
                //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                //std::cout << " Time taken by organize data : " << duration.count() << " microseconds" << std::endl;
            }
            // first detection
            else
            {
                posSaver.push_back(updatedPositionsHuman);
                if (!queueTriangulation.empty()) queueTriangulation.pop();
                queueTriangulation.push(updatedPositionsHuman);
            }
        }
        // no data
        else
        {
            //nothing to do
        }
    }

    void getPreviousData(std::vector<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>>& previousTracker, std::vector<std::vector<cv::Mat1b>>& previousTemplate, 
        std::vector<std::vector<cv::Rect2i>>& roi, std::vector<std::vector<bool>>& scales, std::vector<std::vector<std::vector<int>>>& previousMove,
        std::queue<std::vector<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>>>& queueYoloTracker, std::queue<std::vector<std::vector<cv::Mat1b>>>& queueYoloTemplate,
        std::queue<std::vector<std::vector<cv::Rect2i>>>& queueYoloRoi,
        std::queue<std::vector<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>>>& queueMosseTracker, std::queue<std::vector<std::vector<cv::Mat1b>>>& queueTemplate, 
        std::queue<std::vector<std::vector<cv::Rect2i>>>& queueRoi,
        std::queue<std::vector<std::vector<bool>>>& queueScale, std::queue<std::vector<std::vector<std::vector<int>>>>& queueMove)
    {
        /*
        *  if yolo data available -> update if tracking was failed
        *  else -> if tracking of optical flow was successful -> update features
        *  else -> if tracking of optical flow was failed -> wait for next update of yolo
        *
        *  [TO DO]
        *  get Yolo data and OF data from queue if possible
        *  organize data
        */
        if (!queueMosseTracker.empty() && !queueTemplate.empty() && !queueRoi.empty() && !queueScale.empty() && !queueMove.empty())
        {
            //std::cout << "queueMosseTemplate.empty" << queueMosseTemplate.empty() << ", queueMosseRoi.empty()" << queueMosseRoi.empty() << ", queueScale.empty()" << queueScale.empty() << std::endl;
            previousTracker = queueMosseTracker.front();
            previousTemplate = queueTemplate.front();
            roi = queueRoi.front();
            scales = queueScale.front();
            previousMove = queueMove.front();
            queueMosseTracker.pop();
            queueTemplate.pop();
            queueRoi.pop();
            queueScale.pop();
            queueMove.pop();
        }
        else if (queueTemplate.empty() && !queueRoi.empty())
        {
            //std::cout << "queueMosseRoi.empty()" << queueMosseRoi.empty() << std::endl;
            roi = queueRoi.front();
            queueRoi.pop();
        }
        std::vector<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>> yoloTracker;
        std::vector<std::vector<cv::Mat1b>> yoloTemplate;
        std::vector<std::vector<cv::Rect2i>> yoloRoi;
        if (!queueYoloTracker.empty() && !queueYoloTemplate.empty())
        {
            //std::cout << "yolo data is available" << std::endl;
            getYoloData(yoloTracker, yoloTemplate, yoloRoi,queueYoloTracker, queueYoloTemplate, queueYoloRoi);
            /* update data here */
            /* iterate for all human detection */
            //std::cout << "searchYolo Template size : " << yoloTemplate.size() << std::endl;
            //std::cout << "searchRoi by optical flow size : " << roi.size() <<std::endl;
            //std::cout << "successful Templates of optical flow : " << previousTemplate.size() << std::endl;
            //std::cout << "successful Templates of Yolo : " << yoloTemplate.size() << std::endl;
            //std::cout << "scales size=" << scales.size() << std::endl;
            for (int i = 0; i < yoloRoi.size(); i++) //for each person
            {
                //std::cout << i << "-th human, num of joints=" <<yoloRoi[i].size()<< std::endl;
                /* some OF tracking were successful */
                if (!yoloTemplate[i].empty())
                {
                    //std::cout << "num of Templates=" << yoloTemplate[i].size() << std::endl;
                    /* existed human detection */
                    if (i < previousTemplate.size())
                    {
                        //std::cout << "previousImg : num of tracked joints : " << previousTemplate[i].size() <<",tracker size="<<previousTracker[i].size() << ", move size=" << previousMove[i].size() << std::endl;
                        /* for all joints */
                        int counterJoint = 0;
                        int counterYoloImg = 0;
                        int counterTemplate = 0; // number of successful Templates by Optical Flow
                        for (cv::Rect2i& roi_joint : roi[i]) //for each joint of previous data
                        {
                            //std::cout << "roi.x=" << roi_joint.x << std::endl;
                            //std::cout << "update data with Yolo detection : " << counterJoint << "-th joint" << std::endl;
                            /* tracking is failed -> update data with yolo data */
                            if (roi_joint.width <= 0)
                            {
                                /* yolo detect joints -> update data */
                                if (yoloRoi[i][counterJoint].width > 0)
                                {
                                    std::cout << "update OF Template features with yolo detection" << std::endl;
                                    roi[i][counterJoint] = yoloRoi[i][counterJoint];
                                    previousTracker[i].insert(previousTracker[i].begin() + counterTemplate, yoloTracker[i][counterYoloImg]);

                                    previousTemplate[i].insert(previousTemplate[i].begin() + counterTemplate, yoloTemplate[i][counterYoloImg]);
                                    scales[i].insert(scales[i].begin() + counterTemplate, false);
                                    previousMove[i].insert(previousMove[i].begin() + counterTemplate, defaultMove);
                                    counterJoint++;
                                    counterYoloImg++;
                                    counterTemplate++;
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
                                    if (yoloRoi[i][counterJoint].width > 0)
                                    {
                                        std::cout << "///// update OF Template with yolo detection" << std::endl;
                                        if (bool_check_psr)
                                        {
                                            if ((previousTracker[i][counterTemplate]->previous_psr) < min_keep_psr)//only when psr is low, update tracker with YOLO
                                            {
                                                previousTracker[i][counterTemplate] = yoloTracker[i][counterYoloImg];
                                                previousTemplate[i][counterTemplate] = yoloTemplate[i][counterYoloImg];
                                            }
                                        }
                                        else
                                        {
                                            previousTracker[i][counterTemplate] = yoloTracker[i][counterYoloImg];
                                            previousTemplate[i][counterTemplate] = yoloTemplate[i][counterYoloImg];
                                        }

                                        //roi[i][counterJoint] = yoloRoi[i][counterJoint];
                                        //scales[i][counterTemplateMosse] = false;
                                        if (static_cast<float>(std::abs((roi_joint.x+roi_joint.width/2) -(yoloRoi[i][counterJoint].x+ yoloRoi[i][counterJoint].width/2))) > DIF_THRESHOLD || 
                                            static_cast<float>(std::abs((roi_joint.y + roi_joint.height/2)- (yoloRoi[i][counterJoint].y+ yoloRoi[i][counterJoint].height/2))) > DIF_THRESHOLD ||
                                            (std::pow(static_cast<float>((roi_joint.x + roi_joint.width / 2) - (yoloRoi[i][counterJoint].x + yoloRoi[i][counterJoint].width / 2)), 2) + std::pow(static_cast<float>((roi_joint.y + roi_joint.height / 2) - (yoloRoi[i][counterJoint].y + yoloRoi[i][counterJoint].height / 2)), 2) > DIF_THRESHOLD * DIF_THRESHOLD))
                                        {
                                            roi[i][counterJoint] = yoloRoi[i][counterJoint];
                                            scales[i][counterTemplate] = false;
                                            previousMove[i][counterTemplate] = defaultMove;
                                        }
                                        counterYoloImg++;
                                    }
                                }
                                /* not update template images -> keep tracking */
                                else
                                {
                                    if (yoloRoi[i][counterJoint].width > 0)
                                    {
                                        if (static_cast<float>(std::abs((roi_joint.x + roi_joint.width / 2) - (yoloRoi[i][counterJoint].x + yoloRoi[i][counterJoint].width / 2))) > DIF_THRESHOLD ||
                                            static_cast<float>(std::abs((roi_joint.y + roi_joint.height / 2) - (yoloRoi[i][counterJoint].y + yoloRoi[i][counterJoint].height / 2))) > DIF_THRESHOLD ||
                                            (std::pow(static_cast<float>((roi_joint.x + roi_joint.width / 2) - (yoloRoi[i][counterJoint].x + yoloRoi[i][counterJoint].width / 2)), 2) + std::pow(static_cast<float>((roi_joint.y + roi_joint.height / 2) - (yoloRoi[i][counterJoint].y + yoloRoi[i][counterJoint].height / 2)), 2) > DIF_THRESHOLD * DIF_THRESHOLD))
                                        {
                                            if (bool_check_psr)
                                            {
                                                if ((previousTracker[i][counterTemplate]->previous_psr) < min_keep_psr)//only when psr is low, update tracker with YOLO
                                                {
                                                    previousTracker[i][counterTemplate] = yoloTracker[i][counterYoloImg];
                                                    previousTemplate[i][counterTemplate] = yoloTemplate[i][counterYoloImg];
                                                }
                                            }
                                            else
                                            {
                                                previousTracker[i][counterTemplate] = yoloTracker[i][counterYoloImg];
                                                previousTemplate[i][counterTemplate] = yoloTemplate[i][counterYoloImg];
                                            }
                                            roi[i][counterJoint] = yoloRoi[i][counterJoint];
                                            scales[i][counterTemplate] = false;
                                            previousMove[i][counterTemplate] = defaultMove;
                                        }
                                        counterYoloImg++;
                                    }
                                }
                                counterJoint++;
                                counterTemplate++;
                                //std::cout << "update iterator" << std::endl;
                            }
                        }
                    }
                    /* new human or all Templates have failed  */
                    else
                    {
                        //std::cout << "new human was detected by Yolo " << std::endl;
                        int counterJoint = 0;
                        int counterYoloImg = 0;
                        std::vector<cv::Rect2i> joints;
                        std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>> trackerJoints;
                        std::vector<cv::Mat1b> TemplateJoints;
                        std::vector<bool> scaleJoints;
                        std::vector<std::vector<int>> moveJoints;
                        /* for every joints */
                        for (cv::Rect2i& roi_yolo_joint : yoloRoi[i])
                        {
                            /* keypoint is found */
                            if (roi_yolo_joint.width > 0)
                            {
                                joints.push_back(roi_yolo_joint);
                                trackerJoints.push_back(yoloTracker[i][counterYoloImg]);
                                TemplateJoints.push_back(yoloTemplate[i][counterYoloImg]);
                                scaleJoints.push_back(false);
                                moveJoints.push_back(defaultMove);
                                counterJoint++;
                                counterYoloImg++;
                            }
                            /* keypoints not found */
                            else
                            {
                                joints.push_back(roi_yolo_joint);
                                counterJoint++;
                            }
                        }
                        if (!roi.empty()) roi = std::vector<std::vector<cv::Rect2i>>{ joints }; //raplace to new data
                        else roi.push_back(joints); //new data :: first detection
                        if (!TemplateJoints.empty()&& !trackerJoints.empty())
                        {
                            previousTemplate.push_back(TemplateJoints);
                            previousTracker.push_back(trackerJoints);
                            scales.push_back(scaleJoints);
                            previousMove.push_back(moveJoints);
                        }
                    }
                }
                /* no yolo tracking was successful or first yolo detection */
                else
                {
                    std::cout << "no Templates detected with Yolo" << std::endl;
                }
            }
        }
    }


    void getYoloData(std::vector<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>>& yoloTracker, std::vector<std::vector<cv::Mat1b>>& yoloTemplate, std::vector<std::vector<cv::Rect2i>>& yoloRoi,
        std::queue< std::vector<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>>>& queueYoloTracker,std::queue<std::vector<std::vector<cv::Mat1b>>>& queueYoloTemplate, std::queue<std::vector<std::vector<cv::Rect2i>>>& queueYoloRoi)
    {
        //std::unique_lock<std::mutex> lock(mtxYolo);
        yoloTracker = queueYoloTracker.front();
        yoloTemplate = queueYoloTemplate.front();
        yoloRoi = queueYoloRoi.front();
        queueYoloTracker.pop();
        queueYoloTemplate.pop();
        queueYoloRoi.pop();
    }

    void track(cv::Mat1b& frame, int& frameIndex,
        cv::Ptr<cv::mytracker::TrackerMOSSE>& tracker, cv::Mat1b& templateImg, cv::Rect2i& roi, bool& scale, std::vector<int>& previousMove,
        cv::Ptr<cv::mytracker::TrackerMOSSE>& updatedTracker, cv::Mat1b& updatedTemplateImg,
        cv::Rect2i& updatedRoi, bool& updatedScale,std::vector<int>& updatedMove,
        std::vector<int>& updatedPos)
    {
        int leftSearch, topSearch, rightSearch, bottomSearch;
        int deltaX_past = previousMove[0]; int deltaY_past = previousMove[1];
        //std::cout << "TM :: previousMove :: deltaX = " << deltaX_past << ", deltaY_past = " << deltaY_past << std::endl;
        if (scale) // scale is set to TM : smaller search area
        {
            double scale_x = scaleXTM;
            double scale_y = scaleYTM;
            if (bool_dynamicScale)
            {
                scale_x = static_cast<double>(std::max(std::min((1.0 + (std::abs(deltaX_past) / roi.width)), scaleXYolo), scaleXTM));
                scale_y = static_cast<double>(std::max(std::min((1.0 + (std::abs(deltaY_past) / roi.height)), scaleYYolo), scaleYTM));
            }
            leftSearch = std::min(std::max(0, static_cast<int>(roi.x + deltaX_past - (scale_x - 1) * roi.width / 2)), frame.cols);
            topSearch = std::min(std::max(0, static_cast<int>(roi.y + deltaY_past - (scale_y - 1) * roi.height / 2)), frame.rows);
            rightSearch = std::max(0, std::min(frame.cols, static_cast<int>(roi.x + deltaX_past + (scale_x + 1) * roi.width / 2)));
            bottomSearch = std::max(0, std::min(frame.rows, static_cast<int>(roi.y + deltaY_past + (scale_y + 1) * roi.height / 2)));
        }
        else // scale is set to YOLO : larger search area
        {
            leftSearch = std::min(frame.cols, std::max(0, static_cast<int>(roi.x + deltaX_past - (scaleXYolo - 1) * roi.width / 2)));
            topSearch = std::min(frame.rows, std::max(0, static_cast<int>(roi.y + deltaY_past - (scaleYYolo - 1) * roi.height / 2)));
            rightSearch = std::max(0, std::min(frame.cols, static_cast<int>(roi.x + deltaX_past + (scaleXYolo + 1) * roi.width / 2)));
            bottomSearch = std::max(0, std::min(frame.rows, static_cast<int>(roi.y + deltaY_past + (scaleYYolo + 1) * roi.height / 2)));
        }
        //tracking
        if ((rightSearch - leftSearch) > templateImg.cols && (bottomSearch - topSearch) > templateImg.rows)
        {
            cv::Rect2i searchArea(leftSearch, topSearch, (rightSearch - leftSearch), (bottomSearch - topSearch));
            //std::cout << "img size : width = " << img.cols << ", height = " << img.rows << std::endl;
            //std::cout << "croppdeImg size: left=" << searchArea.x << ", top=" << searchArea.y << ", width=" << searchArea.width << ", height=" << searchArea.height << std::endl;
            cv::Mat1b croppedImg = frame.clone();
            croppedImg = croppedImg(searchArea); // crop img
            cv::Point matchLoc(-1,-1);//for template matching
            cv::Rect2d mosseRoi; //roi for MOSSE
            //process
            std::thread threadMatching(&TemplateMatching::matching, this, std::ref(croppedImg), std::ref(templateImg), std::ref(matchLoc));
            double psr = track_mosse(croppedImg, roi, searchArea, previousMove, tracker, mosseRoi);
            threadMatching.join(); //wait for TM to finish
            //finished

            //data analysis
            //mosse
            if (psr >= threshold_mosse) //MOSSE succeeded!
            {
                cv::Rect2i newRoi;
                int leftRoi = std::min(std::max(0, static_cast<int>(mosseRoi.x + leftSearch)), frame.cols);
                int topRoi = std::min(std::max(0, static_cast<int>(mosseRoi.y + topSearch)), frame.rows);
                int rightRoi = std::max(std::min(frame.cols, static_cast<int>(leftRoi + mosseRoi.width)), 0);
                int bottomRoi = std::max(std::min(frame.rows, static_cast<int>(topRoi + mosseRoi.height)), 0);
                newRoi.x = leftRoi; newRoi.y = topRoi; newRoi.width = rightRoi - leftRoi; newRoi.height = bottomRoi - topRoi;
                if (newRoi.width > MIN_SEARCH && newRoi.height > MIN_SEARCH)
                {
                    double deltaX_current = (newRoi.x + newRoi.width / 2) - (roi.x + roi.width / 2); double deltaY_current = (newRoi.y + newRoi.height / 2) - (roi.y + roi.height / 2);
                    //std::cout << "MOSSE : current :: deltaX = " << deltaX_current << ", deltaY = " << deltaY_current << std::endl
                    /* moving constraints */
                    if (std::pow(deltaX_current, 2) + std::pow(deltaY_current, 2) >= MoveThreshold)
                    {
                        //update data
                        updatedRoi = newRoi;
                        updatedTracker = tracker;
                        updatedTemplateImg = frame(newRoi);
                        updatedScale = true;
                        updatedPos = std::vector<int>{ frameIndex, newRoi.x,newRoi.y, newRoi.width, newRoi.height };
                        int deltaX_future, deltaY_future;
                        if ((gamma * deltaX_current + (1 - gamma) * (double)deltaX_past) <= 0) //negative
                            deltaX_future = std::max((int)(gamma * deltaX_current + (1 - gamma) * (double)deltaX_past), -MAX_VEL);
                        else //positive
                            deltaX_future = std::min((int)(gamma * deltaX_current + (1 - gamma) * (double)deltaX_past), MAX_VEL);
                        if ((gamma * deltaY_current + (1 - gamma) * (double)deltaY_past) <= 0) //negative
                            deltaY_future = std::max((int)(gamma * deltaY_current + (1 - gamma) * (double)deltaY_past), -MAX_VEL);
                        else //positive
                            deltaY_future = std::min((int)(gamma * deltaY_current + (1 - gamma) * (double)deltaY_past), MAX_VEL);
                        updatedMove = std::vector<int>{ deltaX_future,deltaY_future };
                    }
                }
            }
            //template matching
            else if (psr<threshold_mosse && matchLoc.x>0) //MOSSE failed but TM succeeded
            {
                int leftRoi, topRoi, rightRoi, bottomRoi;
                cv::Mat1b newTemplate;
                cv::Rect2i newRoi;
                leftRoi = std::min(std::max(0, static_cast<int>(matchLoc.x + leftSearch)), frame.cols);
                topRoi = std::min(std::max(0, static_cast<int>(matchLoc.y + topSearch)), frame.rows);
                rightRoi = std::min(frame.cols, static_cast<int>(leftRoi + templateImg.cols));
                bottomRoi = std::min(frame.rows, static_cast<int>(topRoi + templateImg.rows));
                // update roi
                newRoi.x = leftRoi;
                newRoi.y = topRoi;
                newRoi.width = rightRoi - leftRoi;
                newRoi.height = bottomRoi - topRoi;
                /* moving constraints */
                if (newRoi.width > MIN_SEARCH && newRoi.height > MIN_SEARCH)
                {
                    double deltaX_current = (newRoi.x + newRoi.width / 2) - (roi.x + roi.width / 2); double deltaY_current = (newRoi.y + newRoi.height / 2) - (roi.y + roi.height / 2);
                    if (std::pow(deltaX_current, 2) + std::pow(deltaY_current, 2) >= MoveThreshold)
                    {
                        //init mosse tracker with template matching data
                        cv::Ptr<cv::mytracker::TrackerMOSSE>& newTracker = cv::mytracker::TrackerMOSSE::create();
                        newTracker->init(frame, cv::Rect2d((double)newRoi.x, (double)newRoi.y, (double)newRoi.width, (double)newRoi.height));
                        updatedTracker = newTracker;
                        //update data
                        updatedRoi = newRoi;
                        updatedTemplateImg = frame(newRoi);
                        updatedScale = true;
                        updatedPos = std::vector<int>{ frameIndex, newRoi.x, newRoi.y,newRoi.width, newRoi.height };
                        int deltaX_future, deltaY_future;
                        if ((gamma * deltaX_current + (1 - gamma) * (double)deltaX_past) <= 0) //negative
                            deltaX_future = std::max((int)(gamma * deltaX_current + (1 - gamma) * (double)deltaX_past), -MAX_VEL);
                        else //positive
                            deltaX_future = std::min((int)(gamma * deltaX_current + (1 - gamma) * (double)deltaX_past), MAX_VEL);
                        if ((gamma * deltaY_current + (1 - gamma) * (double)deltaY_past) <= 0) //negative
                            deltaY_future = std::max((int)(gamma * deltaY_current + (1 - gamma) * (double)deltaY_past), -MAX_VEL);
                        else //positive
                            deltaY_future = std::min((int)(gamma * deltaY_current + (1 - gamma) * (double)deltaY_past), MAX_VEL);
                        updatedMove = std::vector<int>{ deltaX_future,deltaY_future };
                        //std::cout << "  ------- succeed in tracking with TM ------------------ " << std::endl;_
                    }
                }
            }

            //failed
            if (updatedRoi.empty() || updatedPos.empty())//faile all tracker
            {
                updatedRoi = cv::Rect2i(-1,-1,-1,-1);
                updatedPos = std::vector<int>{ frameIndex, -1, -1,-1,-1 };
            }
        }
        else
        {
            updatedRoi = cv::Rect2i(-1, -1, -1, -1);
            updatedPos = std::vector<int>{ frameIndex, -1, -1,-1,-1 };
        }
    }

    void matching(cv::Mat1b& croppedImg, cv::Mat1b& templateImg, cv::Point& matchLoc)
    {
        cv::Mat result; // for saving template matching results
        int result_cols = croppedImg.cols - templateImg.cols + 1;
        int result_rows = croppedImg.rows - templateImg.rows + 1;
        //std::cout << "result_cols :" << result_cols << ", result_rows:" << result_rows << std::endl;
        // template seems to go out of frame
        if (result_cols > 0 || result_rows > 0)
        {
            //std::cout << "croppedImg :: left=" << leftSearch << ", top=" << topSearch << ", right=" << rightSearch << ", bottom=" << bottomSearch << std::endl;
            result.create(result_rows, result_cols, CV_32FC1); // create result array for matching quality+
            // const char* trackbar_label = "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED"; 2 is not so good
            cv::matchTemplate(croppedImg, templateImg, result, MATCHINGMETHOD); // template Matching
            //std::cout << "finish matchTemplate" << std::endl;
            double minVal;    // minimum score
            double maxVal;    // max score
            cv::Point minLoc; // minimum score left-top points
            cv::Point maxLoc; // max score left-top points

            cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat()); // In C++, we should prepare type-defined box for returns, which is usually pointer
            /* find matching object */
            if ((MATCHINGMETHOD == cv::TM_SQDIFF_NORMED && minVal <= matchingThreshold) || ((MATCHINGMETHOD == cv::TM_CCOEFF_NORMED || MATCHINGMETHOD == cv::TM_CCORR_NORMED) && maxVal >= matchingThreshold))
            {
                if (MATCHINGMETHOD == cv::TM_SQDIFF_NORMED)
                {
                    matchLoc = minLoc;
                }
                else
                {
                    matchLoc = maxLoc;
                }
            }
        }
    }

    double track_mosse(cv::Mat1b& croppedImg, cv::Rect2i& previousRoi, cv::Rect2i& searchArea,std::vector<int>& previousMove,cv::Ptr<cv::mytracker::TrackerMOSSE>& tracker, cv::Rect2d& croppedRoi)
    {
        croppedRoi.x = (double)(previousRoi.x - searchArea.x);
        croppedRoi.y = (double)(previousRoi.y - searchArea.y);
        croppedRoi.width = (double)(previousRoi.width);
        croppedRoi.height = (double)(previousRoi.height);
        // MOSSE Tracker
        double psr = tracker->update(croppedImg, croppedRoi, previousMove, true, bool_skip, threshold_mosse);
        return psr;
    }

};

#endif