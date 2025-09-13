#include "utility.h"


bool Utility::getImagesFromQueueMot(std::array<cv::Mat1b, 2>& imgs, int& frameIndex)
{
    if (!queueFrame_mot.empty() && !queueFrameIndex_mot.empty())
    {
        imgs = queueFrame_mot.front();
        frameIndex = queueFrameIndex_mot.front();
        return true;
    }
    else
        return false;
}

bool Utility::getImagesFromQueueYoloPose(std::array<cv::Mat1b, 2>& imgs, int& frameIndex)
{
    if (!queueFrame_yolopose.empty() && !queueFrameIndex_yolopose.empty())
    {
        imgs = queueFrame_yolopose.front();
        frameIndex = queueFrameIndex_yolopose.front();
        return true;
    }
    else
        return false;

}

bool Utility::getImagesFromQueueOptflow(std::array<cv::Mat1b, 2>& imgs, int& frameIndex)
{
    if (!queueFrame_optflow.empty() && !queueFrameIndex_optflow.empty())
    {
        imgs = queueFrame_optflow.front();
        frameIndex = queueFrameIndex_optflow.front();
        return true;
    }
    else
        return false;

}

bool Utility::getFrameFromQueueRobot(int& frameIndex)
{
    if (!queueFrameIndex_robot.empty())
    {
        frameIndex = queueFrameIndex_robot.front();
        return true;
    }
    else
        return false;
}

void Utility::checkStorage(std::vector<std::vector<cv::Rect2d>>& posSaverYolo, std::vector<int>& detectedFrame, std::string fileName)
{

    // Open the file for writing
    std::ofstream outputFile(fileName);
    if (!outputFile.is_open())
    {
        std::cerr << "Error: Could not open the file." << std::endl;
    }
    //std::cout << "estimated position :: Yolo :: " << std::endl;
    int count = 1;
    //std::cout << "posSaverYolo :: Contensts ::" << std::endl;
    for (int i = 0; i < posSaverYolo.size(); i++)
    {
        //std::cout << (i + 1) << "-th iteration : " << std::endl;
        for (int j = 0; j < posSaverYolo[i].size(); j++)
        {
            //std::cout << detectedFrame[i] << "-th frame :: left=" << posSaverYolo[i][j].x << ", top=" << posSaverYolo[i][j].y << ", width=" << posSaverYolo[i][j].width << ", height=" << posSaverYolo[i][j].height << std::endl;
            outputFile << detectedFrame[i];
            outputFile << ",";
            outputFile << posSaverYolo[i][j].x;
            outputFile << ",";
            outputFile << posSaverYolo[i][j].y;
            outputFile << ",";
            outputFile << posSaverYolo[i][j].width;
            outputFile << ",";
            outputFile << posSaverYolo[i][j].height;
            if (j != posSaverYolo[i].size() - 1)
            {
                outputFile << ",";
            }
        }
        outputFile << "\n";
    }
    // close file
    outputFile.close();
}

void Utility::checkClassStorage(std::vector<std::vector<int>>& classSaverYolo, std::vector<int>& detectedFrame, std::string fileName)
{
    // Open the file for writing
    std::ofstream outputFile(fileName);
    if (!outputFile.is_open())
    {
        std::cerr << "Error: Could not open the file." << std::endl;
    }
    int count = 1;
    //std::cout << "Class saver :: Contensts ::" << std::endl;
    for (int i = 0; i < classSaverYolo.size(); i++)
    {
        //std::cout << detectedFrame[i] << "-th frame : " << std::endl;
        for (int j = 0; j < classSaverYolo[i].size(); j++)
        {
            //std::cout << classSaverYolo[i][j] << " ";
            outputFile << detectedFrame[i];
            outputFile << ",";
            outputFile << classSaverYolo[i][j];
            if (j != classSaverYolo[i].size() - 1)
            {
                outputFile << ",";
            }
        }
        outputFile << "\n";
        //std::cout << std::endl;
    }
    // close file
    outputFile.close();
}

void Utility::checkStorageTM(std::vector<std::vector<cv::Rect2d>>& posSaverYolo, std::vector<int>& detectedFrame, std::string fileName)
{
    // Open the file for writing
    std::ofstream outputFile(fileName);
    if (!outputFile.is_open())
    {
        std::cerr << "Error: Could not open the file." << std::endl;
    }
    //std::cout << "estimated position :: Yolo :: " << std::endl;
    int count = 1;
    //std::cout << "posSaverYolo :: Contents ::" << std::endl;
    for (int i = 0; i < posSaverYolo.size(); i++)
    {
        //std::cout << (i + 1) << "-th iteration : " << std::endl;
        for (int j = 0; j < posSaverYolo[i].size(); j++)
        {
            //std::cout << detectedFrame[i] << "-th frame :: left=" << posSaverYolo[i][j].x << ", top=" << posSaverYolo[i][j].y << ", width=" << posSaverYolo[i][j].width << ", height=" << posSaverYolo[i][j].height << std::endl;
            outputFile << detectedFrame[i];
            outputFile << ",";
            outputFile << posSaverYolo[i][j].x;
            outputFile << ",";
            outputFile << posSaverYolo[i][j].y;
            outputFile << ",";
            outputFile << posSaverYolo[i][j].width;
            outputFile << ",";
            outputFile << posSaverYolo[i][j].height;
            if (j != posSaverYolo[i].size() - 1)
            {
                outputFile << ",";
            }
        }
        outputFile << "\n";
    }
    // close file
    outputFile.close();
}

void Utility::checkClassStorageTM(std::vector<std::vector<int>>& classSaverYolo, std::vector<int>& detectedFrame, std::string fileName)
{
    // Open the file for writing
    std::ofstream outputFile(fileName);
    if (!outputFile.is_open())
    {
        std::cerr << "Error: Could not open the file." << std::endl;
    }
    int count = 1;
    //std::cout << "Class saver :: Contensts ::" << std::endl;
    for (int i = 0; i < classSaverYolo.size(); i++)
    {
        //std::cout << detectedFrame[i] << "-th frame : " << std::endl;
        for (int j = 0; j < classSaverYolo[i].size(); j++)
        {
            //std::cout << classSaverYolo[i][j] << " ";
            outputFile << detectedFrame[i];
            outputFile << ",";
            outputFile << classSaverYolo[i][j];
            if (j != classSaverYolo[i].size() - 1)
            {
                outputFile << ",";
            }
        }
        outputFile << "\n";
        //std::cout << std::endl;
    }
    // close file
    outputFile.close();
}

void Utility::checkSeqData(std::vector<std::vector<std::vector<double>>>& dataLeft, std::string fileName)
{
    // Open the file for writing
    /* bbox data */
    std::ofstream outputFile(fileName);
    std::vector<int> frameIndexes;
    frameIndexes.reserve(2000);
    if (!outputFile.is_open())
    {
        std::cerr << "Error: Could not open the file." << std::endl;
    }
    //std::cout << "file open :: save sequence data in csv file :: data size="<<dataLeft.size() << std::endl;
    for (int i = 0; i < dataLeft.size(); i++) //num objects
    {
        //std::cout << (i + 1) << "-th objects : " << std::endl;
        for (int j = 0; j < dataLeft[i].size(); j++) //num sequence
        {
            //std::cout << j << ":: frameIndex=" << dataLeft[i][j][0] << "class label=" << dataLeft[i][j][1] << ", left=" << dataLeft[i][j][2] << ", top=" << dataLeft[i][j][3] << " "<<", width="<<dataLeft[i][j][4]<<", height="<<dataLeft[i][j][5]<<std::endl;
            auto it = std::find(frameIndexes.begin(), frameIndexes.end(), dataLeft[i][j][0]);
            /* new frame index */
            if (it == frameIndexes.end()) frameIndexes.push_back(dataLeft[i][j][0]);
            outputFile << dataLeft[i][j][0];
            outputFile << ",";
            outputFile << dataLeft[i][j][1];
            outputFile << ",";
            outputFile << dataLeft[i][j][2];
            outputFile << ",";
            outputFile << dataLeft[i][j][3];
            outputFile << ",";
            outputFile << dataLeft[i][j][4];
            outputFile << ",";
            outputFile << dataLeft[i][j][5];
            if (j != dataLeft[i].size() - 1)
            {
                outputFile << ",";
            }
        }
        outputFile << "\n";
        //std::cout << std::endl;
    }
    // close file
    outputFile.close();
}

void Utility::checkKfData(std::vector<std::vector<std::vector<double>>>& dataLeft, std::string fileName)
{
    // Open the file for writing
    /* bbox data */
    std::ofstream outputFile(fileName);
    std::vector<int> frameIndexes;
    frameIndexes.reserve(2000);
    if (!outputFile.is_open())
    {
        std::cerr << "Error: Could not open the file." << std::endl;
    }
    //std::cout << "file open :: save sequence data in csv file :: data size=" << dataLeft.size() << std::endl;
    for (int i = 0; i < dataLeft.size(); i++) //num objects
    {
        //std::cout << (i + 1) << "-th objects : " << std::endl;
        for (int j = 0; j < dataLeft[i].size(); j++) //num sequence
        {
            //std::cout << j << ":: frameIndex=" << dataLeft[i][j][0] << "xCenter=" << dataLeft[i][j][1] << ", yCenter=" << dataLeft[i][j][2]  << std::endl;
            auto it = std::find(frameIndexes.begin(), frameIndexes.end(), dataLeft[i][j][0]);
            /* new frame index */
            if (it == frameIndexes.end()) frameIndexes.push_back(dataLeft[i][j][0]);
            outputFile << dataLeft[i][j][0];
            outputFile << ",";
            outputFile << dataLeft[i][j][1];
            outputFile << ",";
            outputFile << dataLeft[i][j][2];
            if (j != dataLeft[i].size() - 1)
            {
                outputFile << ",";
            }
        }
        outputFile << "\n";
        //std::cout << std::endl;
    }
    // close file
    outputFile.close();
}

void Utility::save3d_mot(std::vector<std::vector<std::vector<double>>>& posSaver, const std::string file)
{
    // Open the file for writing
    std::ofstream outputFile(file);
    if (!outputFile.is_open())
    {
        std::cerr << "Error: Could not open the file." << std::endl;
    }
    /*num of objects */
    for (int i = 0; i < posSaver.size(); i++)
    {
        /*num of sequence*/
        for (int j = 0; j < posSaver[i].size(); j++)
        {
            outputFile << posSaver[i][j][0];//frame
            outputFile << ",";
            outputFile << posSaver[i][j][1];//label
            outputFile << ",";
            outputFile << posSaver[i][j][2];//x
            outputFile << ",";
            outputFile << posSaver[i][j][3];//y
            outputFile << ",";
            outputFile << posSaver[i][j][4];//z
            if (j != posSaver[i].size() - 1)
            {
                outputFile << ",";
            }
        }
        outputFile << "\n";
    }
    // close file
    outputFile.close();
}

void Utility::save_params(std::vector<std::vector<std::vector<double>>>& posSaver, const std::string file)
{
    // Open the file for writing
    std::ofstream outputFile(file);
    if (!outputFile.is_open())
    {
        std::cerr << "Error: Could not open the file." << std::endl;
    }
    /*num of objects */
    for (int i = 0; i < posSaver.size(); i++)
    {
        /*num of sequence*/
        for (int j = 0; j < posSaver[i].size(); j++)
        {
            for (int k = 0; k < posSaver[i][j].size(); k++) {
                outputFile << posSaver[i][j][k];
                if (j == posSaver[i].size() - 1 && k == posSaver[i][j].size() - 1)
                    continue;
                else
                    outputFile << ",";
            }
        }
        outputFile << "\n";
    }
    // close file
    outputFile.close();
}

void Utility::saveMatching(std::vector<std::vector<std::vector<int>>>& dataLeft, std::string fileName)
{
    // Open the file for writing
    /* bbox data */
    std::ofstream outputFile(fileName);
    std::vector<int> frameIndexes;
    frameIndexes.reserve(2000);
    if (!outputFile.is_open())
    {
        std::cerr << "Error: Could not open the file." << std::endl;
    }
    //std::cout << "file open :: save sequence data in csv file :: data size="<<dataLeft.size() << std::endl;
    for (int i = 0; i < dataLeft.size(); i++) //num sequence
    {
        //std::cout << (i + 1) << "-th objects : " << std::endl;
        for (int j = 0; j < dataLeft[i].size(); j++) //num objects
        {
            /* new frame index */
            outputFile << dataLeft[i][j][0];
            outputFile << ",";
            outputFile << dataLeft[i][j][1];
            if (j != dataLeft[i].size() - 1)
            {
                outputFile << ",";
            }
        }
        outputFile << "\n";
        //std::cout << std::endl;
    }
    // close file
    outputFile.close();
}

void Utility::pushFrame(std::array<cv::Mat1b, 2>& src, const int frameIndex)
{
    //{
    //    std::unique_lock<std::mutex> lock(mtx_img);
    if (!queueFrame_optflow.empty()) queueFrame_optflow.pop();
    if (!queueFrameIndex_optflow.empty()) queueFrameIndex_optflow.pop();
    queueFrame_optflow.push(src);
    queueFrameIndex_optflow.push(frameIndex);

    if (!queueFrame_mot.empty()) queueFrame_mot.pop();
    if (!queueFrameIndex_mot.empty()) queueFrameIndex_mot.pop();
    queueFrame_mot.push(src);
    queueFrameIndex_mot.push(frameIndex);

    if (!queueFrame_yolopose.empty()) queueFrame_yolopose.pop();
    if (!queueFrameIndex_yolopose.empty()) queueFrameIndex_yolopose.pop();
    queueFrame_yolopose.push(src);
    queueFrameIndex_yolopose.push(frameIndex);

    if (!queueFrameIndex_robot.empty()) queueFrameIndex_robot.pop();
    queueFrameIndex_robot.push(frameIndex);
    
    //}
}

void Utility::saveYolo(std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver, const std::string& file)
{
    // Open the file for writing
    std::ofstream outputFile(file);
    if (!outputFile.is_open())
    {
        std::cerr << "Error: Could not open the file." << std::endl;
    }
    /* write posSaver data to csv file */
    //std::cout << "estimated position :: YOLO :: " << std::endl;
    /*sequence*/
    for (int i = 0; i < posSaver.size(); i++)
    {
        //std::cout << i << "-th sequence data ::: " << std::endl;
        /*num of humans*/
        for (int j = 0; j < posSaver[i].size(); j++)
        {
            //std::cout << j << "-th human detection:::" << std::endl;
            /*num of joints*/
            for (int k = 0; k < posSaver[i][j].size(); k++)
            {
                //std::cout << k << "-th joint :: frameIndex=" << posSaver[i][j][k][0] << ", xCenter=" << posSaver[i][j][k][1] << ", yCenter=" << posSaver[i][j][k][2] << std::endl;
                outputFile << posSaver[i][j][k][0];
                outputFile << ",";
                outputFile << posSaver[i][j][k][1];
                outputFile << ",";
                outputFile << posSaver[i][j][k][2];
                outputFile << ",";
                outputFile << posSaver[i][j][k][3];
                outputFile << ",";
                outputFile << posSaver[i][j][k][4];
                if (k != posSaver[i][j].size() - 1)
                {
                    outputFile << ",";
                }
            }
            outputFile << "\n";
        }
    }
    // Close the file
    outputFile.close();
}

void Utility::saveYoloMulti(std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver, const std::string& file)
{
    // Open the file for writing
    std::ofstream outputFile(file);
    if (!outputFile.is_open())
    {
        std::cerr << "Error: Could not open the file." << std::endl;
    }
    /* write posSaver data to csv file */
    //std::cout << "estimated position :: YOLO :: " << std::endl;
    /*sequence*/
    for (int i = 0; i < posSaver.size(); i++)//for each human
    {
        for (int j = 0; j < posSaver[i].size(); j++)//for each sequence
        {
            /*num of joints*/
            for (int k = 0; k < posSaver[i][j].size(); k++)//for each joints
            {
                for (int l = 0; l < posSaver[i][j][k].size(); l++) {
                    outputFile << posSaver[i][j][k][l];//frame
                    if (j == (posSaver[i].size() - 1) && k == (posSaver[i][j].size() - 1) && l == (posSaver[i][j][k].size() - 1))
                        continue;
                    else
                        outputFile << ",";
                }
            }
        }
        outputFile << "\n";
    }
    // Close the file
    outputFile.close();
}

void Utility::save(std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver, const std::string& file)
{
    // Open the file for writing
    std::ofstream outputFile(file);
    if (!outputFile.is_open())
    {
        std::cerr << "Error: Could not open the file." << std::endl;
    }
    //std::cout << "estimated position :: Optical Flow :: " << std::endl;
    /*human*/
    for (int i = 0; i < posSaver.size(); i++)
    {
        //std::cout << i << "-th sequence data ::: " << std::endl;
        /*seq*/
        for (int j = 0; j < posSaver[i].size(); j++)
        {
            //std::cout << j << "-th human detection:::" << std::endl;
            /*num of joints*/
            for (int k = 0; k < posSaver[i][j].size(); k++)
            {
                for (int l = 0; l < posSaver[i][j][k].size(); l++) {
                    //std::cout << k << "-th joint :: frameIndex=" << posSaver[i][j][k][0] << ", xCenter=" << posSaver[i][j][k][1] << ", yCenter=" << posSaver[i][j][k][2] << std::endl;
                    outputFile << posSaver[i][j][k][l];//frame
                    if (j == posSaver[i].size() - 1 && k == (posSaver[i][j].size() - 1) && l == posSaver[i][j][k].size() - 1)
                        continue;
                    else
                        outputFile << ",";
                }
            }
        }
        outputFile << "\n";
    }
    // close file
    outputFile.close();
}

void Utility::save3d(std::vector<std::vector<std::vector<std::vector<double>>>>& posSaver, const std::string& file)
{
    // Open the file for writing
    std::ofstream outputFile(file);
    if (!outputFile.is_open())
    {
        std::cerr << "Error: Could not open the file." << std::endl;
    }
    //std::cout << "estimated position :: Optical Flow :: " << std::endl;
    /*human*/
    for (int i = 0; i < posSaver.size(); i++)
    {
        //std::cout << i << "-th sequence data ::: " << std::endl;
        /*seq*/
        for (int j = 0; j < posSaver[i].size(); j++)
        {
            //std::cout << j << "-th human detection:::" << std::endl;
            /*num of joints*/
            for (int k = 0; k < posSaver[i][j].size(); k++)
            {
                for (int l = 0; l < posSaver[i][j][k].size(); l++) {
                    //std::cout << k << "-th joint :: frameIndex=" << posSaver[i][j][k][0] << ", xCenter=" << posSaver[i][j][k][1] << ", yCenter=" << posSaver[i][j][k][2] << std::endl;
                    outputFile << posSaver[i][j][k][l];//frame
                    if (j == posSaver[i].size() - 1 && k == (posSaver[i][j].size() - 1) && l == posSaver[i][j][k].size() - 1)
                        continue;
                    else
                        outputFile << ",";
                }
            }
        }
        outputFile << "\n";
    }
    // close file
    outputFile.close();
}

void Utility::saveDeterminant(std::string fileName, std::vector<double>& d) {
    /**
    * @brief save determinant in csv file
    * @param[in] d list of determinants
    */
    std::ofstream outputFile(fileName);
    if (!outputFile.is_open())
    {
        std::cerr << "Error: Could not open the file." << std::endl;
    }
    int count = 1;
    //std::cout << "Class saver :: Contensts ::" << std::endl;
    for (int i = 0; i < d.size(); i++)//sequence
    {
        outputFile << d[i];
        outputFile << "\n";
    }
    // close file
    outputFile.close();
}

void Utility::saveMat(std::string fileName, std::vector<std::vector<cv::Mat>>& joints) {
    /**
    * @brief save joints data in csv file
    * @param[in] joints : num sequence, num joints, {px,py,pz,nx,ny.nz}
    */
    //save data into csv file
    // Open the file for writing
    std::ofstream outputFile(fileName);
    if (!outputFile.is_open())
    {
        std::cerr << "Error: Could not open the file." << std::endl;
    }
    int count = 1;
    //std::cout << "Class saver :: Contensts ::" << std::endl;
    //std::cout << "joints.size()=" << joints.size() << std::endl;
    for (int i = 0; i < joints.size(); i++)//for sequence.
    {
        //std::cout << "joints[i].size()=" << joints[i].size() << std::endl;
        for (int j = 0; j < joints[i].size(); j++)//for each joint
        {
            for (int k = 0; k < joints[i][j].rows; k++) {
                outputFile << joints[i][j].at<double>(k);
                if ((j == (joints[i].size() - 1)) && (k == (joints[i][j].rows - 1)))
                    continue;
                else
                    outputFile << ",";

            }
        }
        outputFile << "\n";
    }
    // close file
    outputFile.close();
}

void Utility::saveSeqMat(std::string fileName, std::vector<std::vector<cv::Mat>>& joints) {
    /**
    * @brief save joints data in csv file
    * @param[in] joints : num sequence, num joints, {px,py,pz,nx,ny.nz}
    */
    //save data into csv file
    // Open the file for writing
    std::ofstream outputFile(fileName);
    if (!outputFile.is_open())
    {
        std::cerr << "Error: Could not open the file." << std::endl;
    }
    int count = 1;
    //std::cout << "Class saver :: Contensts ::" << std::endl;
    //std::cout << "joints.size()=" << joints.size() << std::endl;
    for (int i = 0; i < joints.size(); i++)//for each sequence
    {
        for (int j = 0; j < joints[i].size(); j++)//for each velocity
        {
            for (int k = 0; k < joints[i][j].rows; k++) {//for each element (vx,vy,vz,rx,ry,rz)
                outputFile << joints[i][j].at<double>(k);
                if ((j == (joints[i].size() - 1)) && (k == (joints[i][j].rows - 1)))
                    continue;
                else
                    outputFile << ",";

            }
        }
        outputFile << "\n";
    }
    // close file
    outputFile.close();
}

void Utility::saveData(std::string fileName, std::vector<std::vector<std::vector<double>>>& joints) {
    /**
    * @brief save joints data in csv file
    * @param[in] joints : num sequence, num joints, {px,py,pz,nx,ny.nz}
    */
    //save data into csv file
    // Open the file for writing
    std::ofstream outputFile(fileName);
    if (!outputFile.is_open())
    {
        std::cerr << "Error: Could not open the file." << std::endl;
    }
    int count = 1;
    //std::cout << "Class saver :: Contensts ::" << std::endl;
    for (int i = 0; i < joints.size(); i++)//sequence
    {
        for (int j = 0; j < joints[i].size(); j++)//joints
        {
            for (int k = 0; k < joints[i][j].size(); k++) {
                outputFile << joints[i][j][k];
                if ((j == (joints[i].size() - 1)) && (k == (joints[i][j].size() - 1)))
                    continue;
                else
                    outputFile << ",";

            }
        }
        outputFile << "\n";
    }
    // close file
    outputFile.close();
}

void Utility::saveData3(std::string fileName, std::vector<std::vector<std::vector<std::vector<double>>>>& joints) {
    /**
    * @brief save joints data in csv file
    * @param[in] joints : num sequence,num_human, num joints, {px,py,pz,nx,ny.nz}
    */

    //save data into csv file
    // Open the file for writing
    std::ofstream outputFile(fileName);
    if (!outputFile.is_open())
    {
        std::cerr << "Error: Could not open the file." << std::endl;
    }
    int count = 1;
    //std::cout << "Class saver :: Contensts ::" << std::endl;
    for (int i = 0; i < joints.size(); i++)//sequence
    {
        for (int l = 0; l < joints[i].size(); l++) {//human
            for (int j = 0; j < joints[i][l].size(); j++)//joints
            {
                for (int k = 0; k < joints[i][l][j].size(); k++) {
                    outputFile << joints[i][l][j][k];
                    if ((l == (joints[i].size() - 1)) && (j == (joints[i][l].size() - 1)) && (k == (joints[i][l][j].size() - 1)))
                        continue;
                    else
                        outputFile << ",";
                }
            }
        }
        outputFile << "\n";
    }
    // close file
    outputFile.close();
}

void Utility::saveData2(std::string fileName, std::vector<std::vector<double>>& joints) {
    /**
    * @brief save joints data in csv file
    * @param[in] joints : num sequence, num joints, {px,py,pz,nx,ny.nz}
    */
    //save data into csv file
    // Open the file for writing
    std::ofstream outputFile(fileName);
    if (!outputFile.is_open())
    {
        std::cerr << "Error: Could not open the file." << std::endl;
    }
    int count = 1;
    //std::cout << "Class saver :: Contensts ::" << std::endl;
    for (int i = 0; i < joints.size(); i++)//sequence
    {
        for (int j = 0; j < joints[i].size(); j++)//joints
        {
            outputFile << joints[i][j];
            if (j != (joints[i].size() - 1))
                outputFile << ",";
        }
        outputFile << "\n";
    }
    // close file
    outputFile.close();
}