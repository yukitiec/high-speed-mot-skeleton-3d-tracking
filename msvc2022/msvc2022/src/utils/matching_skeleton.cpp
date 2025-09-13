#include "matching_skeleton.h"


void Matching_skeleton::main(std::vector<std::vector<std::vector<std::vector<int>>>>& seqData_left, std::vector<std::vector<std::vector<std::vector<int>>>>& seqData_right,
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
    matchingHung(seqData_left, seqData_right, matching);

}

void Matching_skeleton::matchingHung(
    std::vector<std::vector<std::vector<std::vector<int>>>>& seqData_left, std::vector<std::vector<std::vector<std::vector<int>>>>& seqData_right,
    std::vector<std::vector<int>>& matching
)
{
    /**
    * @brief match objects in 2 cameras with Hungarian algorithm
    * @param[in] seqData_left,seqData_right {n_human, n_seq, n_joints, (frame,left,top,width,height)}
    * @param[out] matching {{index_left, index_right}}
    */

    //calculate center positions and save in the storages for the candidates.
    std::vector<std::vector<double>> candidates_left, candidates_right;//{n_human, (frame,x_center,y_center)}
    if (!seqData_left.empty()) {
        calculate_centers(seqData_left, candidates_left, true);
    }
    if (!seqData_right.empty()) {
        calculate_centers(seqData_right, candidates_right, false);
    }

    //calculate cost matrix
    if (!candidates_left.empty() && !candidates_right.empty()) {
        std::vector<std::vector<double>> cost_matrix;
        double cost_total, cost_frame;//label, frame matching. previously matched objects.
        cv::Point2d cost_geometry;//x,y,and size difference.
        for (int i_left = 0; i_left < candidates_left.size(); i_left++) {//for each left candidate {frame,label,left,top,width,height}
            std::vector<double> cost_row;
            for (int i_right = 0; i_right < candidates_right.size(); i_right++) {//for each right candidate

                //geometry
                cost_geometry = compareGeometricFeatures(candidates_left[i_left], candidates_right[i_right]);
                //frame matching
                cost_frame = compareID(candidates_left[i_left][0], candidates_right[i_right][0], true);
                //total
                cost_total = cost_geometry.x + cost_geometry.y + cost_frame;

                //history. previously matched?
                if (!idx_match_prev.empty()) {
                    if ((idx_match_prev[i_left] != -1) && idx_match_prev[i_left] != i_right)//not matched previously
                        cost_total *= penalty_newMatch;
                }
                //save in the cost_row
                cost_row.push_back(cost_total);
            }
            if (!cost_row.empty())
                cost_matrix.push_back(cost_row);
        }

        if (!cost_matrix.empty()) {
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
        else {
            if (candidates_left.size() >= 1) {
                std::vector<int> idexes_match_new;
                for (int i = 0; i < candidates_left.size(); i++)
                    idexes_match_new.push_back(-1);
                idx_match_prev = idexes_match_new;
            }
        }
    }
}

void Matching_skeleton::calculate_centers(std::vector<std::vector<std::vector<std::vector<int>>>>& seqData,
    std::vector<std::vector<double>>& candidates, bool bool_left)
{
    /**
    * @brief calculate centers
    * @param[in] seqData {n_human,n_seq,n_joints,(frame,left,top,width,height)}
    * @param[out] candidates {n_human, (x_center,y_center)}
    * @param[in] whether data is from left.
    */

    std::vector<std::vector<int>> data;
    double frame;

    for (int i = 0; i < seqData.size(); i++) {//for each human
        data = seqData[i].back();//latest data.
        double x_center = 0.0; double y_center = 0.0;
        double counter_joint = 0;
        for (int j = 0; j < data.size(); j++) {//for each joint
            if (data[j][0] > 0) {//detected correctly.
                frame = (double)data[j][0];
                x_center += (double)data[j][1] + (double)data[j][3] / 2.0;
                y_center += (double)data[i][2] + (double)data[j][4] / 2.0;
                counter_joint += 1.0;//increment counter_joint for detected joints.
            }
        }
        if (counter_joint > 0) {//calculate center
            x_center /= counter_joint;
            y_center /= counter_joint;
        }
        else {//none was found
            frame = -1.0;
            if (bool_left) {//left
                x_center = -100.0;
                y_center = -100.0;
            }
            else {//right
                x_center = -200.0;
                y_center = -200.0;
            }
        }
        candidates.push_back({ frame,x_center,y_center });
    }
}

cv::Point2d Matching_skeleton::compareGeometricFeatures(std::vector<double>& left, std::vector<double>& right) {
    /**
    *@brief calculate geometric cost in x and y axes.
    * @param[in] left,right : (x_center,y_center)
    */

    //variables setting.
    double delta_x, delta_y, delta_size;
    double centerX_left = left[1];
    double centerY_left = left[2];
    double centerX_right = right[1];
    double centerY_right = right[2];

    //diffenrence in x axis
    if (centerX_right >= centerX_left) //right x coordinates go over left x coordinate
        delta_x = Cost_max;
    else //candidate
        delta_x = 0.0;

    //difference in y axis
    delta_y = std::abs(centerY_left - centerY_right + Delta_oy);

    //result storage.
    cv::Point2d result(delta_x, delta_y);

    return result;
}

double Matching_skeleton::compareID(int label1, int label2, bool bool_frame) {
    /**
    * @brief compare 2 labels and return cost
    * @param[in] label1, label2 labels (detection label)
    * @param[in] bool_frame frame data or not.
    */

    if (bool_frame) {//frame
        if (label1 == label2 && (label1 > 0 && label2 > 0))//updated correctly
            return 0.0;
        else
            return Cost_identity;
    }
    else {//detection label.
        if (label1 == label2)
            return 0.0;
        else
            return Cost_identity;
    }
}