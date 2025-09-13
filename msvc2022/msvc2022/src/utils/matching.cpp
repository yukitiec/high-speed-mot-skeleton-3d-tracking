#include "matching.h"


void Matching::main(std::vector<std::vector<std::vector<double>>>& seqData_left, std::vector<std::vector<std::vector<double>>>& seqData_right,
    const double& oY_left, const double& oY_right, std::vector<std::vector<int>>& matching)
{
    /**
    * @brief match objects in 2 cameras with hungarian algorithm in stereo vision
    * @param[in] seqData_left,seqData_right : {n_objects, sequence, (frame,label,left,top,width,height)}
    * @param[in] oY_left,oY_right : optical point in each camera
    * @param[out] matching pairs of index in seqData {n_pairs, (idx_left, idx_right)}
    */

    //diffenrence in y-axis between each camera
    Delta_oy = oY_right - oY_left;
    if (bool_hungarian) {//use Hungarian algorithm
        matchingHung(seqData_left, seqData_right, matching);
    }
    else if (!bool_hungarian) { //don't use Hungarian algorithm
        std::vector<std::vector<double>> ball_left, ball_right, box_left, box_right; //storage for data {n_objects,(frame,label,left,top,width,height)}
        std::vector<int> ball_index_left, box_index_left, ball_index_right, box_index_right; //storage for index
        //extract position and index, and then sort -> 300 microseconds for 2 
        frame_left = arrangeData(seqData_left, ball_left, box_left, ball_index_left, box_index_left); //sort data
        frame_right = arrangeData(seqData_right, ball_right, box_right, ball_index_right, box_index_right); //sort data, retain indexes
        //matching data in y value
        //ball
        int num_left = ball_index_left.size();
        int num_right = ball_index_right.size();
        auto start_time2 = std::chrono::high_resolution_clock::now();
        // more objects detected in left camera -> 79 microseconds for 2
        matchingObj(ball_left, ball_right, ball_index_left, ball_index_right, oY_left, oY_right, matching);
        //box
        matchingObj(box_left, box_right, box_index_left, box_index_right, oY_left, oY_right, matching);
    }

    if (debug)
    {
        for (int i = 0; i < matching.size(); i++)
            std::cout << i << "-th matching :: left : " << matching[i][0] << ", right: " << matching[i][1] << std::endl;
    }
}

int Matching::arrangeData(std::vector<std::vector<std::vector<double>>>& seqData, std::vector<std::vector<double>>& data_ball, std::vector<std::vector<double>>& data_box,
    std::vector<int>& index_ball, std::vector<int>& index_box)
{
    /**
    * @brief arrange data to convert data based on labels
    * @param[in] seqData : {n_objects, sequence, (frame,label,left,top,width,height)}
    * @param[out] data_ball,data_box : latest data for ball and box {n_objects,(frame,label,left,top,width,height)}
    * @param[out] index_ball,index_box : index list of successful data
    * @return latest frame
    */

    int frame_latest = 0;
    for (int i = 0; i < seqData.size(); i++)//for each object
    {
        if (seqData[i].back()[0] != -1) //tracking is successful
        {
            if (seqData[i].back()[1] == 0) {//ball
                data_ball.push_back(seqData[i].back());
                index_ball.push_back(i);
            }
            else if (seqData[i].back()[1] == 1) {//box
                data_box.push_back(seqData[i].back());
                index_box.push_back(i);
            }
            if (seqData[i].back()[0] > frame_latest)
                frame_latest = seqData[i].back()[0];
        }
    }
    if (!bool_hungarian) {
        // sort data in y-coordinate in ascending order
        sortData(data_ball, index_ball);
        sortData(data_box, index_box);
    }
    return frame_latest;
}

void Matching::sortData(std::vector<std::vector<double>>& data, std::vector<int>& classes)
{
    /**
    * @brief sort data in y-coordinate ascending order
    * @param[out] data {n objects, (frame,label,left,top,width,height)}
    * @param[out] classes inddex list
    */

    // Create an index array to remember the original order
    std::vector<size_t> index(data.size());
    for (size_t i = 0; i < index.size(); ++i)
    {
        index[i] = i;
    }
    // Sort data1 based on centerX values and apply the same order to data2 : {frameIndex,label,left,top,width,height}
    std::sort(index.begin(), index.end(), [&](size_t a, size_t b)
        { return (data[a][3] + data[a][5] / 2) >= (data[b][3] + data[b][5] / 2); });

    std::vector<std::vector<double>> sortedData(data.size());
    std::vector<int> sortedClasses(classes.size());

    for (size_t i = 0; i < index.size(); ++i)
    {
        sortedData[i] = data[index[i]];
        sortedClasses[i] = classes[index[i]];
    }

    data = sortedData;
    classes = sortedClasses;
}

void Matching::matchingObj(std::vector<std::vector<double>>& ball_left, std::vector<std::vector<double>>& ball_right, std::vector<int>& ball_index_left, std::vector<int>& ball_index_right,
    const float& oY_left, const float& oY_right, std::vector<std::vector<int>>& matching)
{
    /**
    * matching objects in both images with label, bbox info. and y-position
    * Args:
    *   ball_left, ball_riht : {frameIndex, label,left,top,width,height}
    * Return:
    *   matching : {{index_left, index_right}}
    */

    int dif_min = dif_threshold;
    int matchIndex_right;
    int i = 0;
    int startIndex = 0; //from which index to start comparison
    //for each object
    while (i < ball_left.size() && startIndex < ball_right.size())
    {
        int j = 0;
        bool boolMatch = false;
        dif_min = dif_threshold;
        std::cout << "startIndex = " << startIndex << std::endl;
        //continue right object y-value is under threshold
        while (true)
        {
            //if right detection is too low -> stop searching
            if (((ball_left[i][3] - ball_right[startIndex + j][3]) > dif_threshold) || (startIndex + j < ball_right.size())) break;
            //bbox info. criteria
            if (((float)ball_left[i][4] * MAX_ROI_DIF >= (float)ball_right[startIndex + j][4]) && ((float)ball_left[i][5] * MAX_ROI_DIF >= (float)ball_right[startIndex + j][5]) &&
                ((float)ball_left[i][4] * MIN_ROI_DIF <= (float)ball_right[startIndex + j][4]) && ((float)ball_left[i][5] * MIN_ROI_DIF <= (float)ball_right[startIndex + j][5]))
            {
                std::cout << "dif_in_2imgs=" << std::abs(((float)(ball_left[i][3] + ball_left[i][5] / 2) - oY_left) - ((float)(ball_right[startIndex + j][3] + ball_right[startIndex + j][5] / 2) - oY_right)) << std::endl;
                int dif = std::abs(((float)(ball_left[i][3] + ball_left[i][5] / 2) - oY_left) - ((float)(ball_right[startIndex + j][3] + ball_right[startIndex + j][5] / 2) - oY_right));
                if (dif < dif_min)
                {
                    dif_min = dif;
                    matchIndex_right = startIndex + j;
                    boolMatch = true;
                }
            }
            j++;
        }
        std::cout << "matching objects found? : " << boolMatch << std::endl;
        //match index is the last value
        if (boolMatch && (matchIndex_right == (startIndex + j - 1))) startIndex += j;
        else startIndex += std::max(j - 1, 0);
        /* matching successful*/
        if (boolMatch)
        {
            matching.push_back({ ball_index_left[i],ball_index_right[matchIndex_right] }); //save matching pair
            //delete selected data
            ball_index_left.erase(ball_index_left.begin() + i);
            ball_left.erase(ball_left.begin() + i);
            //ball_index_right.erase(ball_index_right.begin() + matchIndex_right);
            //ball_right.erase(ball_right.begin() + matchIndex_right);
        }
        // can't find matching object
        else
            i++;
    }
}


void Matching::matchingHung(
    std::vector<std::vector<std::vector<double>>>& seqData_left, std::vector<std::vector<std::vector<double>>>& seqData_right,
    std::vector<std::vector<int>>& matching
)
{
    /**
    * @brief match objects in 2 cameras with Hungarian algorithm
    * @param[in] seqData_left,seqData_right {n objects, n_seq,(frame,label,left,top,width,height)}
    * @param[out] matching {{index_left, index_right}}
    */

    //gather latest dadta
    std::vector<std::vector<double>> candidates_left, candidates_right;
    std::vector<int> index_left, index_right;
    if (!seqData_left.empty()) {
        for (int i = 0; i < seqData_left.size(); i++)
            candidates_left.push_back(seqData_left[i].back());//in order to maintaing data consistency, use the all data. if not updated, exclude in Hungarian algorithm.
    }
    if (!seqData_right.empty()) {
        for (int i = 0; i < seqData_right.size(); i++)
            candidates_right.push_back(seqData_right[i].back());
    }

    //calculate cost matrix
    if (!candidates_left.empty() && !candidates_right.empty()) {
        std::vector<std::vector<double>> cost_matrix;
        double cost_label, cost_frame, cost_history;//label, frame matching. previously matched objects.
        cv::Point3d cost_geometry;//x,y,and size difference.
        for (int i_left = 0; i_left < candidates_left.size(); i_left++) {//for each left candidate {frame,label,left,top,width,height}
            std::vector<double> cost_row;
            for (int i_right = 0; i_right < candidates_right.size(); i_right++) {//for each right candidate
                //geometry
                cost_geometry = compareGeometricFeatures(candidates_left[i_left], candidates_right[i_right]);
                //frame
                cost_frame = compareID(candidates_left[i_left][0], candidates_right[i_right][0], true);
                //label
                cost_label = compareID(candidates_left[i_left][1], candidates_right[i_right][1], false);
                //total
                double cost_total = cost_geometry.x + cost_geometry.y + cost_geometry.z + cost_frame + cost_label;
                //history. previously matched?
                if (idx_match_prev.size() > i_left) {
                    if (idx_match_prev[i_left] != -1)
                        if (idx_match_prev[i_left] != i_right)//not matched previously
                            cost_total *= penalty_newMatch;
                }
                //save in the cost_row
                cost_row.push_back(cost_total);
            }
            cost_matrix.push_back(cost_row);
        }
        //Hungarian algorithm
        std::vector<int> assignment, idxes_match_new;//assignment
        double cost = HungAlgo.Solve(cost_matrix, assignment);
        int index_match;
        for (int i_left = 0; i_left < assignment.size(); i_left++) { //for each candidate in a left camera
            index_match = assignment[i_left];
            if (index_match >= 0) {//matching is successful
                if (cost_matrix[i_left][index_match] < Cost_identity) {
                    matching.push_back({ i_left,index_match });
                    idxes_match_new.push_back(index_match);
                }
                else
                    idxes_match_new.push_back(-1);//no match
            }
            else
                idxes_match_new.push_back(-1);//no match
        }
        idx_match_prev = idxes_match_new;//update previous index to be matched.
    }
}

cv::Point3d Matching::compareGeometricFeatures(std::vector<double>& left, std::vector<double>& right) {
    double delta_x, delta_y, delta_size;
    //prepare basic info
    double centerX_left = left[2] + left[4] / 2;
    double centerY_left = left[3] + left[5] / 2;
    double centerX_right = right[2] + right[4] / 2;
    double centerY_right = right[3] + right[5] / 2;
    double w_left = left[4];
    double h_left = left[5];
    double w_right = right[4];
    double h_right = right[5];
    double gamma_left = std::max(1.0, h_left) / std::max(1.0, w_left);
    double gamma_right = std::max(1.0, h_right) / std::max(1.0, w_right);
    double gamma_rate = gamma_left / gamma_right;
    if (gamma_rate < 1.0)
        gamma_rate = 1.0 / gamma_rate;

    //diffenrence in x axis
    delta_x = centerX_left - centerX_right;
    if (delta_x < 0.0) //right x coordinates go over left x coordinate
        delta_x = Cost_max;
    else {//candidate
        delta_x = lambda_x * std::max(0.0, std::min(1.0, std::sin(3.141592 / 2.0 * (1.0 - delta_x / threshold_xdiff_))));
    }

    if (std::isnan(delta_x))
        delta_x = Cost_max;

    //difference in y axis
    delta_y = std::abs(centerY_left - centerY_right + Delta_oy);
    //normalize between 0 and 1
    delta_y = lambda_y * std::max(0.0, std::min(1.0, delta_y / threshold_ydiff_));
    if (delta_y == lambda_y)
        delta_y = Cost_max;

    if (std::isnan(delta_y))
        delta_y = Cost_max;

    //diffenrence in size
    if (h_right * w_right > 0) {
        delta_size = (h_left * w_left) / (h_right * w_right);
        if (0.0 < delta_size && delta_size < 1.0)//delta_size>=1,0
            delta_size = 1.0 / delta_size;
        if (delta_size > 0.0) {
            delta_size = lambda_size * std::max(0.0, (delta_size * gamma_rate - 1.0) / (threshold_size_ - 1.0));
        }
        else
            delta_size = Cost_max;
    }
    else
        delta_size = Cost_max;

    if (std::isnan(delta_size))
        delta_size = Cost_max;

    //result storage.
    cv::Point3d result(delta_x, delta_y, delta_size);

    return result;
}

double Matching::compareID(int label1, int label2, bool bool_frame) {
    /**
    * @brief compare 2 labels and return cost
    * @param[in] label1, label2 labels (detection label)
    * @param[in] bool_frame frame data or not.
    */

    if (bool_frame) {//frame
        if (label1 == label2 && (label1 > 0 && label2 > 0))//updated correctly
            return 0.0;
        else
            return Cost_max;
    }
    else {//detection label.
        if (label1 == label2)
            return 0.0;
        else
            return Cost_max;
    }
}