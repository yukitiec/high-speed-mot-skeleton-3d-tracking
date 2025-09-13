#include "stdafx.h"
#include "yolopose_batch.h"
#include "tracker.h"
#include "utility.h"
#include "triangulation.h"
#include "global_parameters.h"

extern std::queue<std::array<cv::Mat1b, 2>> queueFrame;
extern std::queue<int> queueFrameIndex;

/* constant valude definition */
extern const std::string filename_left;
extern const std::string filename_right;
extern const bool save;
extern const bool boolSparse;
extern const bool boolGray;
extern const bool boolBatch;
extern const int LEFT;
extern const int RIGHT;
extern const std::string methodDenseOpticalFlow; //"lucasKanade_dense","rlof"
extern const float qualityCorner;
/* roi setting */
extern const int roiSize_shoulder;
extern const int roiSize_elbow;
extern const int roiSize_wrist;
extern const float MoveThreshold;
extern const float epsironMove;
/* dense optical flow skip rate */
extern const int skipPixel;
/*if exchange template of Yolo */
extern const bool boolChange;

void yoloPoseDetect()
{
    /* constructor of YOLOPoseEstimator */
    //else YOLOPose yolo_left, yolo_right;
    YOLOPoseBatch yolo;
    Utility utyolo;
    int countIteration = 0;
    float t_elapsed = 0;
    /* prepare storage */
    std::vector<std::vector<std::vector<std::vector<int>>>> posSaver_left; //[sequence,numHuman,joints,element] :{frameIndex,xCenter,yCenter}
    std::vector<std::vector<std::vector<std::vector<int>>>> posSaver_right; //[sequence,numHuman,joints,element] :{frameIndex,xCenter,yCenter}
    posSaver_left.reserve(300);
    posSaver_right.reserve(300);
    std::this_thread::sleep_for(std::chrono::seconds(3));
    if (queueFrame.empty())
    {
        while (queueFrame.empty())
        {
            if (!queueFrame.empty())
            {
                break;
            }
            //std::cout << "wait for images" << std::endl;
        }
    }
    /* frame is available */
    else
    {
        int counter = 1;
        int counterFinish = 0;
        while (true)
        {
            if (counterFinish == 10)
            {
                break;
            }
            /* frame can't be available */
            if (queueFrame.empty())
            {
                counterFinish++;
                std::cout << "Yolo :: by finish :: " << 10 - counterFinish << std::endl;
                /* waiting */
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
            /* frame available -> start yolo pose estimation */
            else
            {
                counterFinish = 0;
                std::array<cv::Mat1b, 2> frames;
                int frameIndex;
                auto start = std::chrono::high_resolution_clock::now();
                utyolo.getImages(frames, frameIndex);
                //if (boolBatch)
                //{
                cv::Mat1b concatFrame;
                //std::cout << "frames[LEFT]:" << frames[LEFT].rows << "," << frames[LEFT].cols << ", frames[RIGHT]:" << frames[RIGHT].rows << "," << frames[RIGHT].cols << std::endl;
                if (frames[LEFT].rows > 0 && frames[RIGHT].rows > 0)
                {
                    cv::hconcat(frames[LEFT], frames[RIGHT], concatFrame);//concatenate 2 imgs horizontally
                    yolo.detect(concatFrame, frameIndex, counter, posSaver_left, posSaver_right);
                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
                    t_elapsed = t_elapsed + static_cast<float>(duration.count());
                    countIteration++;
                    std::cout << "Time taken by YOLO detection : " << duration.count() << " milliseconds" << std::endl;
                }
            }
        }
    }
    std::cout << "YOLO" << std::endl;
    std::cout << " process speed :: " << static_cast<int>(countIteration / t_elapsed * 1000) << " Hz for " << countIteration << " cycles" << std::endl;
    std::cout << "*** LEFT ***" << std::endl;
    std::cout << "posSaver_left size=" << posSaver_left.size() << std::endl;
    utyolo.saveYolo(posSaver_left, file_yolo_left);
    std::cout << "*** RIGHT ***" << std::endl;
    std::cout << "posSaver_right size=" << posSaver_right.size() << std::endl;
    utyolo.saveYolo(posSaver_right, file_yolo_right);
}


int main()
{
    /* image inference */
    /*
    cv::Mat img = cv::imread("video/0019.jpg");
    cv::Mat1b imgGray;
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
    std::cout << img.size()<< std::endl;
    int counter = 1;
    yoloPoseEstimator.detect(imgGray,counter);
    */
    //constructor 
    Utility ut;
    TemplateMatching tracking;
    Triangulation tri;

    /* video inference */;
    cv::VideoCapture capture_left(filename_left);
    if (!capture_left.isOpened())
    {
        // error in opening the video input
        std::cerr << "Unable to open left file!" << std::endl;
        return 0;
    }
    cv::VideoCapture capture_right(filename_right);
    if (!capture_right.isOpened())
    {
        // error in opening the video input
        std::cerr << "Unable to open right file!" << std::endl;
        return 0;
    }
    int counter = 0;
    /* start multiThread */
    std::thread threadYolo(yoloPoseDetect);
    std::thread threadTrack(&TemplateMatching::main, tracking);
    //std::thread threadRemoveFrame(&Utility::removeFrame, ut);
    //std::thread thread3d(&Triangulation::main, tri);
    while (true)
    {
        // Read the next frame
        cv::Mat frame_left, frame_right;
        capture_left >> frame_left;
        capture_right >> frame_right;
        counter++;
        //std::cout << "left size=" << frame_left.size() << ", right size=" << frame_right.size() << std::endl;
        if (frame_left.empty() || frame_right.empty())
            break;
        cv::Mat1b frameGray_left, frameGray_right;
        cv::cvtColor(frame_left, frameGray_left, cv::COLOR_RGB2GRAY);
        cv::cvtColor(frame_right, frameGray_right, cv::COLOR_RGB2GRAY);
        std::array<cv::Mat1b, 2> frames = { frameGray_left,frameGray_right };
        // cv::Mat1b frameGray;
        //  cv::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);
        ut.pushImg(frames, counter);
    }
    threadYolo.join();
    threadTrack.join();
    //threadRemoveFrame.join();
    //thread3d.join();

    return 0;
}
