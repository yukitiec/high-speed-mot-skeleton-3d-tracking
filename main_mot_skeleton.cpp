#include "mot_skeleton.h"

/* main function */
void Mot_skeleton::main()
{
    try {
        if (std::filesystem::create_directory(rootDir_)) {
            std::cout << "Directory created successfully: " << rootDir_ << std::endl;
        }
        else {
            std::cout << "Directory already exists or could not be created: " << rootDir_ << std::endl;
        }
    }
    catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    }

    /* video inference */
    //constructor 
    RobotControl robot;

    //MOT
    const std::string rootDir = "C:/Users/kawaw/cpp/collaboration_catching_realworld/camera_params/downsample_512";
    Sequence seq(rootDir);

    //skeleton
    Buffer_skeleton buffer_skeleton;
    Triangulation_skeleton tri_skeleton(rootDir);

    int counter = 0;

    //MOT
    //sequence thread
    std::thread threadSeq(&Sequence::main, seq);

    //skeleton
    bool bool_ground_truth = false;
    std::thread threadYolo_pose(&Mot_skeleton::yolo, this, bool_ground_truth);
    std::thread threadOF(&Mot_skeleton::denseOpticalFlow, this);
    std::thread threadBuffer_skeleton(&Buffer_skeleton::main, buffer_skeleton);
    std::thread thread3d(&Triangulation_skeleton::main, tri_skeleton);

    //MOT yolo
    yoloDetect();

    //sequence threead.
    threadSeq.join();

    //skeleton
    threadYolo_pose.join();
    threadOF.join();
    threadBuffer_skeleton.join();
    thread3d.join();

}

/* Yolo thread function definition */
void Mot_skeleton::yoloDetect()
{
    /* Yolo Detection Thread
         * Args:
         *   queueFrame : frame
         *   queueFrameIndex : frame index
         *   queueYoloTemplate : detected template img
         *   queueYoloBbox : detected template bbox
         */

         // YoloDetector initialization
        //YOLODetect yolodetectorLeft;
    float t_elapsed = 0;
    YOLODetect_batch yolo;

    /* initialization */
    if (!q_yolo2seq_left.empty())
    {
        while (!q_yolo2seq_left.empty())
        {
            q_yolo2seq_left.pop();
        }
    }

    if (!q_yolo2seq_right.empty())
    {
        //std::cout << "queueYoloClassIndexesLeft isn't empty" << std::endl;
        while (!q_yolo2seq_right.empty())
        {
            q_yolo2seq_right.pop();
        }
    }

    if (q_startTracking.empty())
    {
        while (true)
        {
            //std::cout << "q_startTracking.empty()" << q_startTracking.empty() << std::endl;
            if (!q_startTracking.empty())
            {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    std::cout << "YOLO has started" << std::endl;

    //start preprocess thread.
    //std::thread thread_preprocess(&YOLODetect_batch::preprocess_loop, yolo);

    int frameIndex;
    int countIteration = 0;
    /* while queueFrame is empty wait until img is provided */
    int counterFinish = 0; // counter before finish
    bool boolImgs;

    //split YOLO inference and preprocess.

    //while (true)
    //{
    //    if (!q_endTracking.empty())
    //        break;

    //    auto start = std::chrono::high_resolution_clock::now();
    //    Info2Gpu i2g;
    //    if (!q_img2gpu.empty())
    //    {
    //        {
    //            std::unique_lock<std::mutex> lock(mtx_gpu);
    //            i2g = q_img2gpu.front();
    //            q_img2gpu.pop();
    //        }

    //        /*start yolo detection */
    //        yolo.detect(i2g.imgTensor,i2g.frame, frameIndex);
    //        auto stop = std::chrono::high_resolution_clock::now();
    //        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    //        countIteration++;
    //        float t_iteration = static_cast<float>(duration.count());
    //        if (t_iteration < 100000.0)
    //            t_elapsed += t_iteration;
    //        if (countIteration % 100 == 0)
    //            std::cout << countIteration << " ### Time taken by YOLO detection : " << duration.count() << " microseconds" << std::endl;
    //    }
    //}
    //thread_preprocess.join();//wait for preprocess loop to finish.

    while (true)
    {
        if (!q_endTracking.empty())
            break;
        auto start = std::chrono::high_resolution_clock::now();
        std::array<cv::Mat1b, 2> frames;
        int frameIndex;
        boolImgs = ut_mot_skeleton.getImagesFromQueueMot(frames, frameIndex);

        if (boolImgs && frames[LEFT_CAMERA].rows > 0 && frames[RIGHT_CAMERA].rows > 0)
        {
            cv::Mat1b concatFrame;
            cv::hconcat(frames[LEFT_CAMERA], frames[RIGHT_CAMERA], concatFrame);

            /*start yolo detection */
            yolo.detect(concatFrame, frameIndex);
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            float t_iteration = static_cast<float>(duration.count());
            if (t_iteration < 300000.0) {
                t_elapsed += t_iteration;
                countIteration++;
            }
            if (countIteration % 50 == 0)
                std::cout << countIteration << " ### Time taken by YOLO detection : " << duration.count() << " microseconds" << std::endl;
        }
    }

    /* check data */
    std::cout << "position saver : Yolo : " << std::endl;
    std::cout << "::::: YOLO Detection :::: process speed : " << static_cast<int>((countIteration) / t_elapsed * 1000000) << " Hz" << std::endl;
}

void Mot_skeleton::yolo(bool bool_ground_truth)
{
    /* constructor of YOLOPoseEstimator */
    //if (boolBatch) 
    YOLOPoseBatch yolo;
    int count_yolo = 0;
    float t_elapsed = 0;
    /* prepare storage */
    std::vector<std::vector<std::vector<std::vector<int>>>> posSaver_left; //[sequence,numHuman,joints,element] :{frameIndex,xCenter,yCenter}
    std::vector<std::vector<std::vector<std::vector<int>>>> posSaver_right; //[sequence,numHuman,joints,element] :{frameIndex,xCenter,yCenter}
    //posSaver_left.reserve(300);
    //posSaver_right.reserve(300);
    if (q_startTracking.empty())
    {
        while (true)
        {
            //std::cout << "q_startTracking.empty()" << q_startTracking.empty() << std::endl;
            if (!q_startTracking.empty())
            {
                break;
            }
            //std::cout << "wait for images" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    /* frame is available */
    std::cout << "YOLO has started" << std::endl;
    int counter = 1;
    int counterFinish = 0;
    //auto start_whole = std::chrono::high_resolution_clock::now();
    std::array<cv::Mat1b, 2> frames;
    int frameIndex;
    bool boolImg = false;
    while (true)
    {
        //auto stop_whole = std::chrono::high_resolution_clock::now();
        //auto duration_whole = std::chrono::duration_cast<std::chrono::seconds>(stop_whole - start_whole);
        //if ((float)duration_whole.count() > 30.0) break;
        if (!q_endTracking.empty())
            break;

        auto start = std::chrono::high_resolution_clock::now();
        //std::cout << "YOLO" << std::endl;
        boolImg = ut_mot_skeleton.getImagesFromQueueYoloPose(frames, frameIndex);
        if (boolImg) {
            cv::Mat1b concatFrame;
            //std::cout << "frames[LEFT]:" << frames[LEFT].rows << "," << frames[LEFT].cols << ", frames[RIGHT]:" << frames[RIGHT].rows << "," << frames[RIGHT].cols << std::endl;
            if (frames[LEFT].rows > 0 && frames[RIGHT].rows > 0)
            {
                cv::hconcat(frames[LEFT], frames[RIGHT], concatFrame);//concatenate 2 imgs horizontally
                yolo.detect(concatFrame, frameIndex, counter, posSaver_left, posSaver_right);
                auto stop = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                if (static_cast<float>(duration.count()) < 300000.0) {
                    t_elapsed = t_elapsed + static_cast<float>(duration.count());
                    count_yolo++;
                }

                if (count_yolo % 50 == 0)
                    std::cout << " ### Time taken by YOLO detection : " << duration.count() << " milliseconds ### " << std::endl;
            }
        }
    }

    std::cout << "YOLO-pose :::: " << std::endl;
    std::cout << " Process Speed : " << static_cast<int>(count_yolo / t_elapsed * 1000000.0) << " Hz for " << count_yolo << " cycles" << std::endl;
    //if (yolo.bool_oneHuman) {//one human detection
    //    std::cout << "*** LEFT ***" << std::endl;
    //    std::cout << " YOLO  :: posSaver_left size=" << posSaver_left.size() << std::endl;
    //    utyolo.saveYolo(posSaver_left, file_yolo_left);
    //    std::cout << "*** RIGHT ***" << std::endl;
    //    std::cout << "YOLO :: posSaver_right size=" << posSaver_right.size() << std::endl;
    //    utyolo.saveYolo(posSaver_right, file_yolo_right);
    //}
    //else {
    //    std::cout << "*** LEFT ***" << std::endl;
    //    if (!yolo.seqHuman_left.empty()) {
    //        for (int i = 0; i < yolo.seqHuman_left.size(); i++) {//for each human
    //            if (yolo.seqHuman_left[i].size() >= 3)
    //                yolo.saveHuman_left.push_back(yolo.seqHuman_left[i]);
    //        }
    //    }

    //    std::cout << " YOLO  :: posSaver_left size=" << yolo.saveHuman_left.size() << std::endl;
    //    utyolo.saveYoloMulti(yolo.saveHuman_left, file_yolo_left);
    //    std::cout << "*** RIGHT ***" << std::endl;
    //    if (!yolo.seqHuman_right.empty()) {
    //        for (int i = 0; i < yolo.seqHuman_right.size(); i++) {//for each human
    //            if (yolo.seqHuman_right[i].size() >= 3)
    //                yolo.saveHuman_right.push_back(yolo.seqHuman_right[i]);
    //        }
    //    }

    //    std::cout << " YOLO  :: posSaver_right size=" << yolo.saveHuman_right.size() << std::endl;
    //    utyolo.saveYoloMulti(yolo.saveHuman_right, file_yolo_right);
    //}
}

void Mot_skeleton::denseOpticalFlow()
{
    /* construction of class */
    OpticalFlow of_left;
    OpticalFlow of_right;
    int count_of = 0;
    float t_elapsed = 0;
    /* prepare storage */
    std::vector<std::vector<std::vector<std::vector<int>>>> posSaver_left; //[sequence,numHuman,numJoints,position] :{frameIndex,xCenter,yCenter}
    std::vector<std::vector<std::vector<std::vector<int>>>> posSaver_right; //[sequence,numHuman,numJoints,position] :{frameIndex,xCenter,yCenter}

    int counterStart = 0;
    //std::string winName_left = "Detection (left)";
    //cv::namedWindow(winName_left);
    //std::string winName_right = "Detection (right)";
    //cv::namedWindow(winName_right);

    while (true)
    {
        if (counterStart == 3) break;
        if (!q_startOptflow.empty()) {
            counterStart++;
            q_startOptflow.pop();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::cout << "start Optflow" << std::endl;
    /* frame is available */
    int counter = 1;
    int counterFinish = 0;
    bool boolImg = false;
    while (true)
    {
        if (!q_endTracking.empty())
            break;

        counterFinish = 0;
        //std::cout << "start opticalflow tracking" << std::endl;
        /* get images from queue */
        std::array<cv::Mat1b, 2> frames;
        int frameIndex;
        std::vector<std::vector<std::vector<int>>> updatedPositionsHuman_left, updatedPositionsHuman_right;
        auto start = std::chrono::high_resolution_clock::now();
        boolImg = ut_mot_skeleton.getImagesFromQueueOptflow(frames, frameIndex);
        if (boolImg) {
            cv::Mat1b frame_left = frames[0];
            cv::Mat1b frame_right = frames[1];
            //cv::Mat1b img_draw_left = frame_left.clone();
            //cv::Mat1b img_draw_right = frame_right.clone();
            if (frame_left.rows > 0 && frame_right.rows > 0)
            {
                std::thread thread_OF_left(&OpticalFlow::main, &of_left, std::ref(frame_left), std::ref(frameIndex),
                    std::ref(q_yolo2optflow_left), std::ref(q_optflow2optflow_left), std::ref(q_optflow2tri_left),
                    std::ref(updatedPositionsHuman_left), true);
                //std::thread thread_OF_right(&OpticalFlow::main, &of, std::ref(frame_right), std::ref(frameIndex), std::ref(posSaver_right), std::ref(q_yolo2optflow_right), std::ref(q_optflow2optflow_right), std::ref(queueTriangulation_right));
                of_right.main(frame_right, frameIndex, q_yolo2optflow_right, q_optflow2optflow_right, q_optflow2tri_right, updatedPositionsHuman_right, false);
                //std::cout << "both OF threads have started" << std::endl;
                thread_OF_left.join();
                if (!updatedPositionsHuman_left.empty() && !updatedPositionsHuman_right.empty() || !of_left.index_delete.empty() || !of_right.index_delete.empty())
                {
                    Optflow2tri data_left, data_right;

                    //lates tracking data.
                    data_left.data = updatedPositionsHuman_left;
                    data_right.data = updatedPositionsHuman_right;

                    //index to delete
                    if (!of_left.index_delete.empty()) data_left.index_delete = of_left.index_delete;
                    if (!of_right.index_delete.empty()) data_right.index_delete = of_right.index_delete;

                    //{
                    //    std::unique_lock<std::mutex> lock_tri(mtxTri);

                    //not pop for preventing access errors.
                    //if (!q_optflow2tri_left.empty())
                    //    q_optflow2tri_left.pop();
                    //if (!q_optflow2tri_right.empty())
                    //    q_optflow2tri_right.pop();

                    q_optflow2tri_left.push(data_left);
                    q_optflow2tri_right.push(data_right);
                    //}

                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                    count_of++;
                    t_elapsed = t_elapsed + static_cast<float>(duration.count());
                    if (count_of % 100 == 0)
                        std::cout << "<<< Time taken by OpticalFlow : " << duration.count() << " microseconds >>>" << std::endl;
                }
                //thread_OF_right.join();


                //draw rectangle in an image.
                //std::cout << "***** posSaver_left size=" << posSaver_left.size() << ", posSaver_right size=" << posSaver_right.size() << "********" << std::endl;
                //for (int i = 0; i < updatedPositionsHuman_left.size(); i++) {//for each human
                //    for (int j = 0; j < updatedPositionsHuman_left[i].size(); j++) {//for each joint.
                //        if (updatedPositionsHuman_left[i][j][1] > 0) {
                //            cv::rectangle(frame_left, cv::Rect(updatedPositionsHuman_left[i][j][1], updatedPositionsHuman_left[i][j][2], updatedPositionsHuman_left[i][j][3], updatedPositionsHuman_left[i][j][4]), cv::Scalar(255), 2);
                //        }
                //    }
                //}
                //for (int i = 0; i < updatedPositionsHuman_right.size(); i++) {//for each human
                //    for (int j = 0; j < updatedPositionsHuman_right[i].size(); j++) {//for each joint.
                //        if (updatedPositionsHuman_right[i][j][1] > 0) {
                //            cv::rectangle(frame_right, cv::Rect(updatedPositionsHuman_right[i][j][1], updatedPositionsHuman_right[i][j][2], updatedPositionsHuman_right[i][j][3], updatedPositionsHuman_right[i][j][4]), cv::Scalar(255), 2);
                //        }
                //    }
                //}
                //cv::resize(frame_left, frame_left, cv::Size(frame_left.cols / 2, frame_left.rows / 2));
                //cv::resize(frame_right, frame_right, cv::Size(frame_right.cols / 2, frame_right.rows / 2));
                //cv::imshow(winName_left, frame_left);
                //cv::imshow(winName_right, frame_right);
                //cv::waitKey(1);

                //std::cout << "***** posSaver_left size=" << posSaver_left.size() << ", posSaver_right size=" << posSaver_right.size() << "********" << std::endl;
            }
        }
    }
    //cv::destroyAllWindows();
    std::cout << "Optical Flow" << std::endl;
    std::cout << " Process Speed : " << static_cast<int>(count_of / t_elapsed * 1000000) << " Hz for " << count_of << " cycles" << std::endl;
    std::cout << "*** LEFT ***" << std::endl;
    ut_mot_skeleton.save(posSaver_left, file_of_left);
    std::cout << "*** RIGHT ***" << std::endl;
    ut_mot_skeleton.save(posSaver_right, file_of_right);
}