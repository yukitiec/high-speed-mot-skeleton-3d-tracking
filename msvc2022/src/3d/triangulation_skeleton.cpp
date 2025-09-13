#include "triangulation_skeleton.h"

void Triangulation_skeleton::main()
{
    if (bool_debug) {
        cv::namedWindow(winName_left);
        cv::namedWindow(winName_right);
    }

    std::vector<std::vector<std::vector<int>>> newData_left, newData_right;//latest data from optical flow. {n_human,n_joints, (frame,x,y)}
    std::vector<int> index_delete_left, index_delete_right;

    //TRIANGULATION
    //storage
    std::vector<std::vector<std::vector<std::vector<double>>>> save_3d, save_kf_3d, save_measure_3d; //{num of human,num_seq num_joints, {frameIndex, X,Y,Z}}
    std::vector<std::vector<std::vector<std::vector<double>>>> seq_3d, seq_kf_3d, seq_measure_3d; //{num of human,num_seq, num_joints, {frameIndex, X,Y,Z}}

    //matching
    std::vector<std::vector<std::vector<int>>> matching_save;//{n_seq,n_human,{idx_left,idx_right}}

    //instances for new detections.
    std::vector<std::vector<std::vector<double>>> initial_new(1, std::vector<std::vector<double>>(numJoint_, std::vector<double>(4, 0.0)));//{frame,x,y,z};initialize with 0.0
    std::vector<KalmanFilter3D> new_kf(6, KalmanFilter3D(INIT_X, INIT_Y, INIT_Z, INIT_VX, INIT_VY, INIT_VZ, INIT_AX, INIT_AY, INIT_AZ, NOISE_POS, NOISE_VEL, NOISE_ACC, NOISE_SENSOR));
    std::vector<int> index_remove_left, index_remove_right;
    //TRIANGULATION - end - 

    //ROBOT CONTROL
    //move obstacle and target
    bool bool_move = true;

    std::vector<double> joints_ivpf{ -PI,-0.314,-PI / 2.0,-0.48,PI / 2.0,0.0 };
    ur_main.cal_poseAll(joints_ivpf);
    ur_main.Jacobian(joints_ivpf);
    double detJ = cv::determinant(ur_main.J);
    //pose current
    ur_main.cal_poseAll(joints_ivpf);
    std::vector<double> pose_current = ur_main.pose6;
    pose_human = std::vector<std::vector<std::vector<double>>>{ {{1.0,0.15,-0.35,10.3},{1.0,0.35,-0.35,10.3},{1.0,-0.01,-0.55,10.3},{1.0,0.55,0.05,10.3},{1.0,-0.25,-0.7,10.3},{1.0,0.6,0.25,10.3}} };//{n_human,n_joints,(frame,x,y,z)}


    std::vector<std::vector<double>> save_jointsAngle;
    //save storage
    std::vector<std::vector<double>> save_dists_minimum; //sequence, num joints
    //robot pose
    std::vector<std::vector<std::vector<double>>> save_joints;//num sequence, num joints, {px,py,pz,nx,ny.nz}}
    //pose_human -> num_joints, seq,position
    std::vector<std::vector<std::vector<std::vector<double>>>> save_joints_human;//num sequence, num_human, num joints, {px,py,pz,nx,ny.nz}}
    std::vector<std::vector<cv::Mat>> velocities_ee;
    std::vector<std::vector<double>> save_virtual_obstacle;//if valid -> save virtual obstacle. {frame,x,y,z}

    //std::ofstream outputFile(file_vels);

    std::vector<std::vector<double>> target_save;
    std::vector<double> target_buffer;

    //if (!outputFile.is_open())
    //{
    //    std::cerr << "Error: Could not open the file." << std::endl;
    //}

    auto start_robot = std::chrono::high_resolution_clock::now();
    //ROBOT CONTROL - end - 

    //iterator
    int count_iteration = 1;
    int count_moving = 150;
    double t_elapsed = 0;
    int iteration = 0;
    int idx_move = 0;
    int count_finish = 0;
    double nx = 0.0;
    double ny = 0.0;
    double nz = 0.0;
    double n_norm = 0.0;

    int counterIteration = 0;
    int counterFinish = 0;
    int counterNextIteration = 0;

    while (true)
    {
        if (!q_optflow2tri_left.empty() || !q_optflow2tri_right.empty())
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    std::cout << "start calculating 3D position" << std::endl;

    while (true) // continue until finish
    {
        if (!q_endTracking.empty())
            break;

        /* new detection data available */
        if (!q_optflow2tri_left.empty() && !q_optflow2tri_right.empty())
        {
            counterNextIteration = 0;

            auto start = std::chrono::high_resolution_clock::now();
            //initialize index_remove.
            index_remove_left.clear();
            index_remove_right.clear();

            //std::cout << "s-1" << std::endl;
            //get latest tracking data.
            getData(newData_left, newData_right, index_delete_left, index_delete_right);
            //std::cout << "s-2" << std::endl;
            if (!newData_left.empty() && !newData_right.empty()) {
                //std::cout << "s-3" << std::endl;
                //organize 2d detection in each image.
                organizeData_2d(newData_left, index_delete_left, seqData_left, kfData_left, measurement_left, index_remove_left, true);
                organizeData_2d(newData_right, index_delete_right, seqData_right, kfData_right, measurement_right, index_remove_right, false);

                //std::cout << "s-4" << std::endl;
                //initialize data_3d based on the seqData_left.
                if (!seqData_left.empty() && seq_3d.empty()) {
                    std::vector<std::vector<std::vector<std::vector<double>>>> initial(seqData_left.size(), std::vector<std::vector<std::vector<double>>>(1, std::vector<std::vector<double>>(numJoint_, std::vector<double>(4, 0.0))));//{frame,x,y,z};initialize with 0.0
                    seq_3d = initial;//synchronize with seqData_left in size.
                    seq_kf_3d = initial;
                    seq_measure_3d = initial;

                    //kalmanfilter
                    std::vector<std::vector<KalmanFilter3D>> initial_kf(seqData_left.size(), std::vector<KalmanFilter3D>(6, KalmanFilter3D(INIT_X, INIT_Y, INIT_Z, INIT_VX, INIT_VY, INIT_VZ, INIT_AX, INIT_AY, INIT_AZ, NOISE_POS, NOISE_VEL, NOISE_ACC, NOISE_SENSOR)));
                    kf_3d = initial_kf;
                }

                //increase data_3d size if seqData_left.size() increase.
                if (seqData_left.size() > seq_3d.size()) {//number of seqData_left is larger than seq_3d.size(). -> add new instances.
                    while (seqData_left.size() > seq_3d.size()) {
                        seq_3d.push_back(initial_new);
                        seq_kf_3d.push_back(initial_new);
                        seq_measure_3d.push_back(initial_new);
                        kf_3d.push_back(new_kf);
                    }
                }

                //decrease data_3d size if index_remove_left isn't empty.
                if (!index_remove_left.empty()) {
                    for (int& idx : index_remove_left) {

                        //delete matching indexes lists.
                        matching.idx_match_prev.erase(matching.idx_match_prev.begin() + idx);

                        //save data in the storage
                        if (seq_3d[idx][0][0][1] > 0) {//updated at least once.
                            save_3d.push_back(seq_3d[idx]);
                            save_kf_3d.push_back(seq_kf_3d[idx]);
                            save_measure_3d.push_back(seq_measure_3d[idx]);
                        }

                        //remove data.
                        seq_3d.erase(seq_3d.begin() + idx);
                        seq_kf_3d.erase(seq_kf_3d.begin() + idx);
                        seq_measure_3d.erase(seq_measure_3d.begin() + idx);
                        kf_3d.erase(kf_3d.begin() + idx);
                    }
                }

                //if index_remove_right isn't empty
                if (!index_remove_right.empty()) {
                    int idx_delete;
                    for (int i = 0; i < index_remove_right.size(); i++) {//for each index of the right
                        idx_delete = findIndex(matching.idx_match_prev, index_remove_right[i]);//find element, index_delete_right[i], from match.idx_match_prev.
                        if (idx_delete >= 0) {//found
                            matching.idx_match_prev[idx_delete] = -1;//convert to -1.
                        }
                    }
                }
                //std::cout << "s-4" << std::endl;
                //3d triangulation 
                if (!seqData_left.empty() && !seqData_right.empty()) {
                    //matching human
                    std::vector<std::vector<int>> matchingIndexes; //{n_pairs, (idx_left, idx_right)}
                    //std::cout << "s-5" << std::endl;
                    matching.main(seqData_left, seqData_right, oY_left, oY_right, matchingIndexes);
                    //std::cout << "s-6" << std::endl;
                    if (!matchingIndexes.empty()) {
                        //std::cout << "matchingIndexes=\n";
                        //for (int i = 0; i < matchingIndexes.size(); i++)
                        //    std::cout << matchingIndexes[i][0] << ", " << matchingIndexes[i][1] << std::endl;
                        //save matchingIndexes.
                        matching_save.push_back(matchingIndexes);
                        //std::cout << "s-7" << std::endl;
                        //triangulate points based on matchingIndexes.-> seq_3d (n_human,n_seq,n_joints,(frame,x,y,z)). if not updated -> (0,0,0,0)
                        triangulation(seqData_left, seqData_right, matchingIndexes, seq_3d, seq_kf_3d, seq_measure_3d);
                        //std::cout << "s-8" << std::endl;
                        //send data to robot control.
                        pose_human.clear();//initialize
                        for (std::vector<int>& index_match : matchingIndexes) {//for each matching
                            pose_human.push_back(seq_3d[index_match[0]].back());//add {n_joints,(frame,x,y,z)}. seq_3d ; {n_human,n_seq,n_joints,(frame,x,y,z)}
                        }
                        //std::cout << "s-9" << std::endl;

                        //Update human pose for robot control
                        //initialze
                        if (!pose_human.empty()) {

                            //compensate lost data and add Head and foot from LS and RS -> n_joints: 6 -> 8
                            skeleton2robot send_data;
                            bool success = check_human_pose(send_data);//update human pose.

                            if (success) {
                                q_skeleton2robot.push(send_data);
                            }
                            //validate this, if Robot control is process in another thread.
                            //queueJointsPositions.push(pose_human);
                        }
                    }
                }

                //std::cout << "arrange data" << std::endl;
                auto stop = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                t_elapsed += duration.count();
                counterIteration++;
                if (counterIteration % 50 == 0)
                    std::cout << "++++++++++++++++++++++++++++++++  time taken by 3d positioning=" << duration.count() << " microseconds +++++++++++++++++++++++++++++++" << std::endl;
            }
        }
        /* at least one data can't be available -> delete data */
        else
        {
            //std::cout << "both data can't be availble :: left " << !queueTriangulation_left.empty() << ", right=" << queueTriangulation_right.empty() << std::endl;
            if (!q_optflow2tri_left.empty() || !q_optflow2tri_right.empty())
            {
                if (counterNextIteration == 30)
                {
                    counterNextIteration = 0;
                    //{
                    if (!q_optflow2tri_left.empty())
                        q_optflow2tri_left.pop();
                    if (!q_optflow2tri_right.empty())
                        q_optflow2tri_right.pop();
                    //}
                }
                else
                {
                    if (!bool_start_robot)
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    counterNextIteration++;
                }
            }
            if (!bool_start_robot)
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

    }
    //finish while loop
    //urCtrl->stopJ();//stop robot
    std::cout << "*** triangulation_skeleton.cpp :: process speed = " << counterIteration / t_elapsed * 1000000 << " Hz ***" << std::endl;
    //save
    if (!seqData_left.empty()) {//save remained data in a storage.
        for (int i = 0; i < seqData_left.size(); i++) {
            save_left.push_back(seqData_left[i]);
            save_kf_left.push_back(kfData_left[i]);
            save_measure_left.push_back(measurement_left[i]);
        }
    }
    std::cout << "*** LEFT *** save_left.size()=" << save_left.size() << std::endl;
    utTri.save(save_left, file_of_left);
    utTri.save(save_kf_left, file_kf_skeleton_left);
    utTri.save(save_measure_left, file_measure_skeleton_left);


    if (!seqData_right.empty()) {//save remained data in a storage.
        for (int i = 0; i < seqData_right.size(); i++) {
            save_right.push_back(seqData_right[i]);
            save_kf_right.push_back(kfData_right[i]);
            save_measure_right.push_back(measurement_right[i]);
        }
    }
    std::cout << "*** RIGHT *** save_right.size()=" << save_right.size() << std::endl;
    utTri.save(save_right, file_of_right);
    utTri.save(save_kf_right, file_kf_skeleton_right);
    utTri.save(save_measure_right, file_measure_skeleton_right);

    std::cout << "seq_3d.size()=" << seq_3d.size() << ", save_3d.size()=" << save_3d.size() << std::endl;
    if (!seq_3d.empty()) {
        for (int i = 0; i < seq_3d.size(); i++) {
            save_3d.push_back(seq_3d[i]);//(i-th human, seq,joint,(frame,x,y.z))
            save_kf_3d.push_back(seq_kf_3d[i]);
            save_measure_3d.push_back(seq_measure_3d[i]);
        }
    }
    std::cout << "***triangulation data*** save_3d.size()=" << save_3d.size() << std::endl;
    utTri.save3d(save_3d, file_3d_pose);
    utTri.save3d(save_kf_3d, file_kf_skeleton_3d);
    utTri.save3d(save_measure_3d, file_measure_skeleton_3d);

    //ROBOT CONTROL
    // close file
    /*outputFile.close();
    std::cout << "*** Robot Control :: process speed = " << iteration / t_elapsed * 1000000 << " Hz ***" << std::endl;*/
    if (!save_joints.empty()) {
        //save data into csv file
        utTri.saveDeterminant(file_determinant, ivpf_main.determinants);
        utTri.saveDeterminant(file_determinant_elbow, ivpf_main.dets_elbow);
        utTri.saveDeterminant(file_determinant_wrist, ivpf_main.dets_wrist);
        utTri.saveData(file_joints_ivpf, save_joints);
        utTri.saveData2(file_jointsAngle_ivpf, save_jointsAngle);
        utTri.saveData3(file_human_ivpf, save_joints_human);
        utTri.saveData2(file_minimumDist_ivpf, save_dists_minimum);
        utTri.saveDeterminant(file_attraction, ivpf_main.gain_attract);
        utTri.saveDeterminant(file_repulsion, ivpf_main.gain_repulsive);
        utTri.saveDeterminant(file_tangent, ivpf_main.gain_tangent);
        utTri.saveDeterminant(file_rep_global, ivpf_main.gain_repulsive_global);
        utTri.saveDeterminant(file_rep_att, ivpf_main.gain_rep_att);
        utTri.saveDeterminant(file_rep_elbow, ivpf_main.gain_rep_elbow);
        utTri.saveDeterminant(file_rep_wrist, ivpf_main.gain_rep_wrist);
        utTri.saveData2(file_lambda, ivpf_main.lambda_list);
        utTri.saveDeterminant(file_eta_repulsive, ivpf_main.etas_repulsive);
        utTri.saveDeterminant(file_eta_tangent, ivpf_main.etas_tangent);
        utTri.saveData2(file_virtual, save_virtual_obstacle);
        utTri.saveData2(file_target, target_save);
        std::cout << "velocities_ee.size()=" << velocities_ee.size() << std::endl;
    }
    //urCtrl->moveJ(config_init, 0.5, 0.5);//speed, acceleration
    //std::this_thread::sleep_for(std::chrono::seconds(1));
}
void Triangulation_skeleton::getData(std::vector<std::vector<std::vector<int>>>& data_left, std::vector<std::vector<std::vector<int>>>& data_right, std::vector<int>& index_delete_left, std::vector<int>& index_delete_right)
{
    //{
    //    std::unique_lock<std::mutex> lock_tri(mtxTri);
    if (!q_optflow2tri_left.empty()) {
        new_left = q_optflow2tri_left.front();
        q_optflow2tri_left.pop();
    }
    if (!q_optflow2tri_right.empty()) {
        new_right = q_optflow2tri_right.front();
        q_optflow2tri_right.pop();
    }
    //}
    //new data
    data_left = new_left.data;
    data_right = new_right.data;
    //index to delete
    index_delete_left = new_left.index_delete;
    index_delete_right = new_right.index_delete;
}

void Triangulation_skeleton::organizeData_2d(
    std::vector<std::vector<std::vector<int>>>& updatedPositionsHuman, std::vector<int>& index_delete,
    std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver,
    std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver_kf,
    std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver_measure,
    std::vector<int>& index_remove, bool bool_left) {

    /**
    * @param[in] updatedPositionsHuman : {(num of human),(num of joints), (frame,left,top,width,height)}
    * @param[out] PosSaver : {n_human,n_seq, n_joints, (frame, left,top,width,height)}
    * @param[out] index_remove : index remove from posSaver according to counters_notUpdate.
    * @param[in] bool_left : whether the data is from the left camera.
    */

    std::vector<int> pastData;
    double frame_latest = -1.0;

    if (!posSaver.empty())//already detected
    {
        //delete human if index_delete is not empty
        if (!index_delete.empty()) {//not empty
            //std::cout << "Triangulation :: posSaver.size()=" << posSaver.size() << std::endl;
            //std::cout << "Triangulation :: index_delete=";
            index_remove = index_delete;//pass index to delete to index_remove
            for (int& idx : index_delete) {
                //std::cout << idx << ",";
                //save posSaver[idx] in save storage.
                if (bool_left) {//left
                    save_left.push_back(posSaver[idx]);
                    save_kf_left.push_back(posSaver_kf[idx]);
                    save_measure_left.push_back(posSaver_measure[idx]);
                    kf_left.erase(kf_left.begin() + idx);//delete idx-th human data.
                }
                else {//right
                    save_right.push_back(posSaver[idx]);
                    save_kf_right.push_back(posSaver_kf[idx]);
                    save_measure_right.push_back(posSaver_measure[idx]);
                    kf_right.erase(kf_right.begin() + idx);
                }

                //remove posSaver[idx].
                posSaver.erase(posSaver.begin() + idx);
                posSaver_kf.erase(posSaver_kf.begin() + idx);
                posSaver_measure.erase(posSaver_measure.begin() + idx);
            }
            //std::cout <<"all" << std::endl;
        }

        // for each human -> if multiple humans, have to match who is the same.** TO DO **
        if (!posSaver.empty()) {
            if (!updatedPositionsHuman.empty()) {

                frame_latest = (double)updatedPositionsHuman[0][0][0];

                double dframe;
                //observation
                Eigen::Vector2d observation;
                std::vector<int> data_past;
                for (int i = 0; i < updatedPositionsHuman.size(); i++)//for each human
                {
                    std::vector<std::vector<int>> tempHuman, tempHuman_kf, tempHuman_measure;//{n_joints, (frame,left,top,width,height)}
                    /* same human */
                    if (posSaver.size() > i)//n_human's detected
                    {
                        // for each joint
                        for (int j = 0; j < updatedPositionsHuman[i].size(); j++)
                        {
                            // detected
                            if (updatedPositionsHuman[i][j][3] > 0)
                            {
                                //update Kalman filter
                                if (bool_left) {
                                    std::vector<int> newData = updatedPositionsHuman[i][j];
                                    dframe = (double)newData[0] - kf_left[i][j].frame_last;//calculate dframe from the current and last frame saved in the kalmanfilter instances.
                                    observation << ((double)newData[1] + (double)(newData[3]) / 2.0), ((double)newData[2] + (double)(newData[4]) / 2.0);//{x_center, y_center}
                                    if (kf_left[i][j].frame_last > 0) {//already updated
                                        kf_left[i][j].predict(dframe);//prediction step
                                        data_past = posSaver[i].back()[j];
                                        double x_prev = (double)data_past[1] + (double)data_past[3] / 2.0;
                                        double y_prev = (double)data_past[2] + (double)data_past[4] / 2.0;
                                        double diff = std::sqrt(std::pow((x_prev - ((double)newData[1] + (double)(newData[3]) / 2.0)), 2) + std::pow((y_prev - ((double)newData[2] + (double)(newData[4]) / 2.0)), 2));

                                        move_threshold_pix_ = dframe * max_move_per_frame_pix_ * std::pow((double)kf_left[i][j].counter_notUpdate, 2);

                                        if (diff < move_threshold_pix_) {
                                            if (kf_left[i][j].counter_notUpdate > 1) {
                                                double x_obs = (1.0 / (1.0 + (double)std::pow(kf_left[i][j].counter_notUpdate, 1))) * x_prev + (1.0 - 1.0 / (1.0 + (double)std::pow(kf_left[i][j].counter_notUpdate, 1))) * ((double)newData[1] + (double)(newData[3]) / 2.0);
                                                double y_obs = (1.0 / (1.0 + (double)std::pow(kf_left[i][j].counter_notUpdate, 1))) * y_prev + (1.0 - 1.0 / (1.0 + (double)std::pow(kf_left[i][j].counter_notUpdate, 1))) * ((double)newData[2] + (double)(newData[4]) / 2.0);
                                                observation << x_obs, y_obs;
                                            }
                                            kf_left[i][j].update(observation);//update step.
                                        }
                                        kf_left[i][j].frame_last = newData[0];//update last frame.
                                    }
                                    else {//first update.
                                        //initialize data.
                                        //delete a vacant instance.
                                        kf_left[i].erase(kf_left[i].begin() + j);
                                        //insert the initial one.
                                        kf_left[i].insert(kf_left[i].begin() + j, KalmanFilter2D_skeleton(((double)newData[1] + (double)(newData[3]) / 2.0), ((double)newData[2] + (double)(newData[4]) / 2.0), INIT_VX, INIT_VY, INIT_AX, INIT_AY, NOISE_POS, NOISE_VEL, NOISE_ACC, NOISE_SENSOR));
                                        dframe = 0.0;
                                        kf_left[i][j].predict(dframe);//prediction step
                                        kf_left[i][j].update(observation);
                                        kf_left[i][j].frame_last = newData[0];
                                    }
                                    tempHuman_measure.push_back(newData);
                                    tempHuman_kf.push_back({ newData[0],(int)(kf_left[i][j].state_(0) - (double)newData[3] / 2.0),(int)(kf_left[i][j].state_(1) - (double)newData[4] / 2.0),newData[3],newData[4] });
                                }
                                else {//right

                                    std::vector<int> newData = updatedPositionsHuman[i][j];
                                    dframe = newData[0] - kf_right[i][j].frame_last;//calculate dframe from the current and last frame saved in the kalmanfilter instances.
                                    observation << ((double)newData[1] + (double)(newData[3]) / 2.0), ((double)newData[2] + (double)(newData[4]) / 2.0);//{x_center, y_center}
                                    if (kf_right[i][j].frame_last > 0) {
                                        kf_right[i][j].predict(dframe);//prediction step
                                        data_past = posSaver[i].back()[j];
                                        double x_prev = (double)data_past[1] + (double)data_past[3] / 2.0;
                                        double y_prev = (double)data_past[2] + (double)data_past[4] / 2.0;
                                        double diff = std::sqrt(std::pow((x_prev - ((double)newData[1] + (double)(newData[3]) / 2.0)), 2) + std::pow((y_prev - ((double)newData[2] + (double)(newData[4]) / 2.0)), 2));

                                        move_threshold_pix_ = dframe * max_move_per_frame_pix_ * std::pow((double)kf_right[i][j].counter_notUpdate, 2);

                                        if (diff < move_threshold_pix_) {

                                            if (kf_right[i][j].counter_notUpdate > 1) {
                                                double x_obs = (1.0 / (1.0 + (double)std::pow(kf_right[i][j].counter_notUpdate, 1))) * x_prev + (1.0 - 1.0 / (1.0 + (double)std::pow(kf_right[i][j].counter_notUpdate, 1))) * ((double)newData[1] + (double)(newData[3]) / 2.0);
                                                double y_obs = (1.0 / (1.0 + (double)std::pow(kf_right[i][j].counter_notUpdate, 1))) * y_prev + (1.0 - 1.0 / (1.0 + (double)std::pow(kf_right[i][j].counter_notUpdate, 1))) * ((double)newData[2] + (double)(newData[4]) / 2.0);
                                                observation << x_obs, y_obs;
                                            }
                                            kf_right[i][j].update(observation);//update step.
                                        }
                                        kf_right[i][j].frame_last = newData[0];//update last frame.
                                    }
                                    else {//first update.
                                        //initialize data.
                                        //delete a vacant instance.
                                        kf_right[i].erase(kf_right[i].begin() + j);
                                        //insert the initial one.
                                        kf_right[i].insert(kf_right[i].begin() + j, KalmanFilter2D_skeleton(((double)newData[1] + (double)(newData[3]) / 2.0), ((double)newData[2] + (double)(newData[4]) / 2.0), INIT_VX, INIT_VY, INIT_AX, INIT_AY, NOISE_POS, NOISE_VEL, NOISE_ACC, NOISE_SENSOR));
                                        dframe = 0.0;
                                        kf_right[i][j].predict(dframe);//prediction step
                                        kf_right[i][j].update(observation);
                                        kf_right[i][j].frame_last = newData[0];
                                    }
                                    tempHuman_measure.push_back(newData);
                                    tempHuman_kf.push_back({ newData[0],(int)(kf_right[i][j].state_(0) - (double)newData[3] / 2.0),(int)(kf_right[i][j].state_(1) - (double)newData[4] / 2.0),newData[3],newData[4] });
                                }

                                if (bool_useKF) {//use Kalman filter data
                                    if (bool_left) {//left
                                        if (kf_left[i][j].counter_update >= 10 && kf_left[i][j].counter_notUpdate == 0) {//updated more than 5 times.
                                            updatedPositionsHuman[i][j][1] = (int)(kf_left[i][j].state_(0) - (double)updatedPositionsHuman[i][j][3] / 2.0);//update left.
                                            updatedPositionsHuman[i][j][2] = (int)(kf_left[i][j].state_(1) - (double)updatedPositionsHuman[i][j][4] / 2.0);//update top based on the kalman filter
                                        }
                                        else if (kf_left[i][j].counter_update >= 10 && kf_left[i][j].counter_notUpdate > 0) {//abnormal value is saved
                                            pastData = posSaver[i].back()[j];//get past data
                                            //updatedPositionsHuman[i][j][1] = pastData[1];
                                            //updatedPositionsHuman[i][j][2] = pastData[2];

                                            double x_center_prev = (double)pastData[1] + (double)pastData[3] / 2.0;
                                            double y_center_prev = (double)pastData[2] + (double)pastData[4] / 2.0;

                                            double x_center = (1.0 / (1.0 + std::pow((double)kf_left[i][j].counter_notUpdate, 2))) * kf_left[i][j].state_(0) + (1.0 - 1.0 / (1.0 + std::pow((double)kf_left[i][j].counter_notUpdate, 2))) * x_center_prev;
                                            double y_center = (1.0 / (1.0 + std::pow((double)kf_left[i][j].counter_notUpdate, 2))) * kf_left[i][j].state_(1) + (1.0 - 1.0 / (1.0 + std::pow((double)kf_left[i][j].counter_notUpdate, 2))) * y_center_prev;
                                            double move = std::sqrt((x_center - x_center_prev) * (x_center - x_center_prev) + (y_center - y_center_prev) * (y_center - y_center_prev));
                                            if (move > dframe * max_move_per_frame_pix_) {
                                                x_center = (x_center - x_center_prev) / std::fabs(x_center - x_center_prev) * (dframe * max_move_per_frame_pix_) + x_center_prev;
                                                y_center = (y_center - y_center_prev) / std::fabs(y_center - y_center_prev) * (dframe * max_move_per_frame_pix_) + y_center_prev;
                                            }
                                            updatedPositionsHuman[i][j][1] = (int)(x_center - (double)updatedPositionsHuman[i][j][3] / 2.0);//left
                                            updatedPositionsHuman[i][j][2] = (int)(y_center - (double)updatedPositionsHuman[i][j][4] / 2.0);//top
                                        }
                                    }
                                    else {//right
                                        if (kf_right[i][j].counter_update >= 10 && kf_right[i][j].counter_notUpdate == 0) {//updated more than 5 times.
                                            updatedPositionsHuman[i][j][1] = (int)(kf_right[i][j].state_(0) - (double)updatedPositionsHuman[i][j][3] / 2.0);//update left.
                                            updatedPositionsHuman[i][j][2] = (int)(kf_right[i][j].state_(1) - (double)updatedPositionsHuman[i][j][4] / 2.0);//update top based on the kalman filter
                                        }
                                        else if (kf_right[i][j].counter_update >= 10 && kf_right[i][j].counter_notUpdate > 0) {//abnormal value is saved
                                            pastData = posSaver[i].back()[j];//get past data
                                            //updatedPositionsHuman[i][j][1] = pastData[1];
                                            //updatedPositionsHuman[i][j][2] = pastData[2];

                                            double x_center_prev = (double)pastData[1] + (double)pastData[3] / 2.0;
                                            double y_center_prev = (double)pastData[2] + (double)pastData[4] / 2.0;

                                            double x_center = (1.0 / (1.0 + std::pow((double)kf_right[i][j].counter_notUpdate, 2))) * kf_right[i][j].state_(0) + (1.0 - (1.0 / (1.0 + std::pow((double)kf_right[i][j].counter_notUpdate, 2)))) * x_center_prev;
                                            double y_center = (1.0 / (1.0 + std::pow((double)kf_right[i][j].counter_notUpdate, 2))) * kf_right[i][j].state_(1) + (1.0 - (1.0 / (1.0 + std::pow((double)kf_right[i][j].counter_notUpdate, 2)))) * y_center_prev;
                                            double move = std::sqrt((x_center - x_center_prev) * (x_center - x_center_prev) + (y_center - y_center_prev) * (y_center - y_center_prev));
                                            if (move > dframe * max_move_per_frame_pix_) {
                                                x_center = (x_center - x_center_prev) / std::fabs(x_center - x_center_prev) * (dframe * max_move_per_frame_pix_) + x_center_prev;
                                                y_center = (y_center - y_center_prev) / std::fabs(y_center - y_center_prev) * (dframe * max_move_per_frame_pix_) + y_center_prev;
                                            }

                                            updatedPositionsHuman[i][j][1] = (int)(x_center - (double)updatedPositionsHuman[i][j][3] / 2.0);//left
                                            updatedPositionsHuman[i][j][2] = (int)(y_center - (double)updatedPositionsHuman[i][j][4] / 2.0);//top
                                        }
                                    }
                                }
                                else {//not use kalman filter data.
                                    if (bool_left) {//left
                                        if (kf_left[i][j].counter_update >= 10 && kf_left[i][j].counter_notUpdate > 0) {//abnormal value is saved
                                            pastData = posSaver[i].back()[j];//get past data
                                            //updatedPositionsHuman[i][j][1] = pastData[1];
                                            //updatedPositionsHuman[i][j][2] = pastData[2];

                                            double x_center_prev = (double)pastData[1] + (double)pastData[3] / 2.0;
                                            double y_center_prev = (double)pastData[2] + (double)pastData[4] / 2.0;

                                            double x_center = (1.0 / (1.0 + std::pow((double)kf_left[i][j].counter_notUpdate, 2))) * kf_left[i][j].state_(0) + (1.0 - 1.0 / (1.0 + std::pow((double)kf_left[i][j].counter_notUpdate, 2))) * x_center_prev;
                                            double y_center = (1.0 / (1.0 + std::pow((double)kf_left[i][j].counter_notUpdate, 2))) * kf_left[i][j].state_(1) + (1.0 - 1.0 / (1.0 + std::pow((double)kf_left[i][j].counter_notUpdate, 2))) * y_center_prev;
                                            double move = std::sqrt((x_center - x_center_prev) * (x_center - x_center_prev) + (y_center - y_center_prev) * (y_center - y_center_prev));
                                            if (move > dframe * max_move_per_frame_pix_) {
                                                x_center = (x_center - x_center_prev) / std::fabs(x_center - x_center_prev) * (dframe * max_move_per_frame_pix_) + x_center_prev;
                                                y_center = (y_center - y_center_prev) / std::fabs(y_center - y_center_prev) * (dframe * max_move_per_frame_pix_) + y_center_prev;
                                            }
                                            updatedPositionsHuman[i][j][1] = (int)(x_center - (double)updatedPositionsHuman[i][j][3] / 2.0);//left
                                            updatedPositionsHuman[i][j][2] = (int)(y_center - (double)updatedPositionsHuman[i][j][4] / 2.0);//top
                                        }
                                    }
                                    else {//right
                                        if (kf_right[i][j].counter_update >= 10 && kf_right[i][j].counter_notUpdate > 0) {//abnormal value is saved
                                            pastData = posSaver[i].back()[j];//get past data
                                            //updatedPositionsHuman[i][j][1] = pastData[1];
                                            //updatedPositionsHuman[i][j][2] = pastData[2];

                                            double x_center_prev = (double)pastData[1] + (double)pastData[3] / 2.0;
                                            double y_center_prev = (double)pastData[2] + (double)pastData[4] / 2.0;

                                            double x_center = (1.0 / (1.0 + std::pow((double)kf_right[i][j].counter_notUpdate, 2))) * kf_right[i][j].state_(0) + (1.0 - (1.0 / (1.0 + std::pow((double)kf_right[i][j].counter_notUpdate, 2)))) * x_center_prev;
                                            double y_center = (1.0 / (1.0 + std::pow((double)kf_right[i][j].counter_notUpdate, 2))) * kf_right[i][j].state_(1) + (1.0 - (1.0 / (1.0 + std::pow((double)kf_right[i][j].counter_notUpdate, 2)))) * y_center_prev;
                                            double move = std::sqrt((x_center - x_center_prev) * (x_center - x_center_prev) + (y_center - y_center_prev) * (y_center - y_center_prev));
                                            if (move > dframe * max_move_per_frame_pix_) {
                                                x_center = (x_center - x_center_prev) / std::fabs(x_center - x_center_prev) * (dframe * max_move_per_frame_pix_) + x_center_prev;
                                                y_center = (y_center - y_center_prev) / std::fabs(y_center - y_center_prev) * (dframe * max_move_per_frame_pix_) + y_center_prev;
                                            }

                                            updatedPositionsHuman[i][j][1] = (int)(x_center - (double)updatedPositionsHuman[i][j][3] / 2.0);//left
                                            updatedPositionsHuman[i][j][2] = (int)(y_center - (double)updatedPositionsHuman[i][j][4] / 2.0);//top
                                        }
                                    }
                                }

                                tempHuman.push_back(updatedPositionsHuman[i][j]);//add latest data.
                            }
                            // not detected
                            else
                            {
                                // already detected.
                                if (posSaver[i].back()[j][3] > 0)//i-th human last data: top >0
                                {
                                    if (bool_useKF) {
                                        if (bool_left) {
                                            if (kf_left[i][j].counter_update >= 10) {//use prediction data.
                                                double dframe = frame_latest - kf_left[i][j].frame_last;
                                                Eigen::Vector<double, 6> kf_prediction;
                                                kf_left[i][j].predict_only(kf_prediction, dframe);
                                                std::vector<int> newData = std::vector<int>{ (int)frame_latest,(int)(kf_prediction(0) - (double)posSaver[i].back()[j][3] / 2.0),(int)(kf_prediction(1) - (double)posSaver[i].back()[j][4] / 2.0),posSaver[i].back()[j][3],posSaver[i].back()[j][4] };
                                                tempHuman_kf.push_back({ (int)frame_latest,(int)(kf_prediction(0) - (double)posSaver[i].back()[j][3] / 2.0),(int)(kf_prediction(1) - (double)posSaver[i].back()[j][4] / 2.0),posSaver[i].back()[j][3],posSaver[i].back()[j][4] });
                                                //measurement
                                                pastData = posSaver[i].back()[j];//adopt the last data.
                                                pastData[0] = (int)frame_latest;
                                                tempHuman_measure.push_back(pastData);
                                                tempHuman.push_back(pastData); //adopt last detection
                                            }
                                            else {
                                                pastData = posSaver[i].back()[j];//adopt the last data.
                                                pastData[0] = (int)frame_latest;
                                                tempHuman.push_back(pastData); //adopt last detection
                                                tempHuman_measure.push_back(pastData);
                                                pastData = posSaver_kf[i].back()[j];//adopt the last data.
                                                pastData[0] = (int)frame_latest;
                                                tempHuman_kf.push_back(pastData); //adopt last detection

                                            }
                                        }
                                        else {
                                            if (kf_right[i][j].counter_update >= 10) {//use prediction data.
                                                double dframe = frame_latest - kf_right[i][j].frame_last;
                                                Eigen::Vector<double, 6> kf_prediction;
                                                kf_right[i][j].predict_only(kf_prediction, dframe);
                                                std::vector<int> newData = std::vector<int>{ (int)frame_latest,(int)(kf_prediction(0) - (double)posSaver[i].back()[j][3] / 2.0),(int)(kf_prediction(1) - (double)posSaver[i].back()[j][4] / 2.0),posSaver[i].back()[j][3],posSaver[i].back()[j][4] };
                                                tempHuman_kf.push_back({ (int)frame_latest,(int)(kf_prediction(0) - (double)posSaver[i].back()[j][3] / 2.0),(int)(kf_prediction(1) - (double)posSaver[i].back()[j][4] / 2.0),posSaver[i].back()[j][3],posSaver[i].back()[j][4] });
                                                //measurement
                                                pastData = posSaver[i].back()[j];//adopt the last data.
                                                pastData[0] = (int)frame_latest;
                                                tempHuman_measure.push_back(pastData);
                                                tempHuman.push_back(pastData); //adopt last detection
                                            }
                                            else {
                                                pastData = posSaver[i].back()[j];//adopt the last data.
                                                pastData[0] = (int)frame_latest;
                                                tempHuman.push_back(pastData); //adopt last detection
                                                tempHuman_measure.push_back(pastData);
                                                pastData = posSaver_kf[i].back()[j];//adopt the last data.
                                                pastData[0] = (int)frame_latest;
                                                tempHuman_kf.push_back(pastData); //adopt last detection
                                            }
                                        }
                                    }
                                    else {
                                        pastData = posSaver[i].back()[j];//adopt the last data.
                                        pastData[0] = (int)frame_latest;
                                        tempHuman.push_back(pastData); //adopt last detection
                                        tempHuman_measure.push_back(pastData);
                                        pastData = posSaver_kf[i].back()[j];//adopt the last data.
                                        pastData[0] = (int)frame_latest;
                                        tempHuman_kf.push_back(pastData); //adopt last detection
                                    }
                                }
                                // not detected yet
                                else {
                                    tempHuman.push_back(updatedPositionsHuman[i][j]); //(frameIndex,-1,-1,-1,-1)
                                    tempHuman_measure.push_back(updatedPositionsHuman[i][j]);
                                    tempHuman_kf.push_back(updatedPositionsHuman[i][j]); //adopt last detection
                                }
                            }
                        }
                        //push new data to i-th human.
                        posSaver[i].push_back(tempHuman);
                        posSaver_kf[i].push_back(tempHuman_kf);
                        posSaver_measure[i].push_back(tempHuman_measure);
                    }
                    //new human
                    else {
                        posSaver.push_back({ updatedPositionsHuman[i] });
                        posSaver_kf.push_back({ updatedPositionsHuman[i] });
                        posSaver_measure.push_back({ updatedPositionsHuman[i] });
                        //make instances of kalman vector.
                        std::vector<KalmanFilter2D_skeleton> kf_new;
                        for (int j = 0; j < updatedPositionsHuman[i].size(); j++) {//for each joint.
                            std::vector<int> newData = updatedPositionsHuman[i][j];
                            observation << ((double)newData[1] + (double)(newData[3]) / 2.0), ((double)newData[2] + (double)(newData[4]) / 2.0);//{x_center, y_center}
                            //push back the initial instance.
                            kf_new.push_back(KalmanFilter2D_skeleton(((double)newData[1] + (double)(newData[3]) / 2.0), ((double)newData[2] + (double)(newData[4]) / 2.0), INIT_VX, INIT_VY, INIT_AX, INIT_AY, NOISE_POS, NOISE_VEL, NOISE_ACC, NOISE_SENSOR));
                            if (newData[3] > 0) {//valid data
                                dframe = 0.0;
                                kf_new.back().predict(dframe);//prediction step
                                kf_new.back().update(observation);//update step
                                kf_new.back().frame_last = newData[0];//update the last frame.
                            }
                        }
                        if (bool_left)//left
                            kf_left.push_back(kf_new);
                        else//right
                            kf_right.push_back(kf_new);
                    }
                }
            }
        }
        else {//all the human data is deleted.
            if (!updatedPositionsHuman.empty()) {
                double dframe;
                //observation
                Eigen::Vector2d observation;
                for (int i = 0; i < updatedPositionsHuman.size(); i++)//fpr each human
                {
                    posSaver.push_back({ updatedPositionsHuman[i] });//{n_human,n_seq(=1),n_joints,(frame,left,top,width,height)}
                    posSaver_kf.push_back({ updatedPositionsHuman[i] });
                    posSaver_measure.push_back({ updatedPositionsHuman[i] });
                    //make instances of kalman vector.
                    std::vector<KalmanFilter2D_skeleton> kf_new;
                    for (int j = 0; j < updatedPositionsHuman[i].size(); j++) {//for each joint.
                        std::vector<int> newData = updatedPositionsHuman[i][j];
                        observation << ((double)newData[1] + (double)(newData[3]) / 2.0), ((double)newData[2] + (double)(newData[4]) / 2.0);//{x_center, y_center}
                        //push back the initial instance.
                        kf_new.push_back(KalmanFilter2D_skeleton(((double)newData[1] + (double)(newData[3]) / 2.0), ((double)newData[2] + (double)(newData[4]) / 2.0), INIT_VX, INIT_VY, INIT_AX, INIT_AY, NOISE_POS, NOISE_VEL, NOISE_ACC, NOISE_SENSOR));
                        if (newData[3] > 0) {//valid data
                            dframe = 0.0;
                            kf_new.back().predict(dframe);//prediction step
                            kf_new.back().update(observation);//update step
                            kf_new.back().frame_last = newData[0];//update the last frame.
                        }
                    }
                    if (bool_left)//left
                        kf_left.push_back(kf_new);
                    else//right
                        kf_right.push_back(kf_new);
                }
            }
        }

    }
    else// first detection
    {
        if (!updatedPositionsHuman.empty()) {
            double dframe;
            //observation
            Eigen::Vector2d observation;
            for (int i = 0; i < updatedPositionsHuman.size(); i++)//fpr each human
            {
                posSaver.push_back({ updatedPositionsHuman[i] });//{n_human,n_seq(=1),n_joints,(frame,left,top,width,height)}
                posSaver_kf.push_back({ updatedPositionsHuman[i] });
                posSaver_measure.push_back({ updatedPositionsHuman[i] });
                //make instances of kalman vector.
                std::vector<KalmanFilter2D_skeleton> kf_new;
                for (int j = 0; j < updatedPositionsHuman[i].size(); j++) {//for each joint.
                    std::vector<int> newData = updatedPositionsHuman[i][j];
                    observation << ((double)newData[1] + (double)(newData[3]) / 2.0), ((double)newData[2] + (double)(newData[4]) / 2.0);//{x_center, y_center}
                    //push back the initial instance.
                    kf_new.push_back(KalmanFilter2D_skeleton(((double)newData[1] + (double)(newData[3]) / 2.0), ((double)newData[2] + (double)(newData[4]) / 2.0), INIT_VX, INIT_VY, INIT_AX, INIT_AY, NOISE_POS, NOISE_VEL, NOISE_ACC, NOISE_SENSOR));
                    if (newData[3] > 0) {//valid data
                        dframe = 0.0;
                        kf_new.back().predict(dframe);//prediction step
                        kf_new.back().update(observation);//update step
                        kf_new.back().frame_last = newData[0];//update the last frame.
                    }
                }
                if (bool_left)//left
                    kf_left.push_back(kf_new);
                else//right
                    kf_right.push_back(kf_new);
            }
        }
    }

    if (bool_debug) {

        //std::cout << "***** posSaver_left size=" << posSaver_left.size() << ", posSaver_right size=" << posSaver_right.size() << "********" << std::endl;
        if (bool_left) {
            if (bool_debug && !queueFrame_yolopose.empty()) {
                utTri.getImagesFromQueueYoloPose(frame_show, frameIndex);
                frame_show_left = frame_show[0];
                frame_show_right = frame_show[1];
            }

            if (frame_show_left.cols > 0) {
                for (int i = 0; i < posSaver.size(); i++) {//for each human
                    for (int j = 0; j < posSaver[i].back().size(); j++) {//for each joint.
                        if (posSaver[i].back()[j][3] > 0) {
                            cv::rectangle(frame_show_left, cv::Rect(posSaver[i].back()[j][1], posSaver[i].back()[j][2], posSaver[i].back()[j][3], posSaver[i].back()[j][4]), cv::Scalar(255), 2);
                        }
                    }
                }
                cv::resize(frame_show_left, frame_show_left, cv::Size(frame_show_left.cols / 2, frame_show_left.rows / 2));
                cv::imshow(winName_left, frame_show_left);
                cv::waitKey(1);
            }
        }
        else {
            if (frame_show_right.cols > 0) {
                for (int i = 0; i < posSaver.size(); i++) {//for each human
                    for (int j = 0; j < posSaver[i].back().size(); j++) {//for each joint.
                        if (posSaver[i].back()[j][3] > 0) {
                            cv::rectangle(frame_show_right, cv::Rect(posSaver[i].back()[j][1], posSaver[i].back()[j][2], posSaver[i].back()[j][3], posSaver[i].back()[j][4]), cv::Scalar(255), 2);
                        }
                    }
                }
                cv::resize(frame_show_right, frame_show_right, cv::Size(frame_show_right.cols / 2, frame_show_right.rows / 2));
                cv::imshow(winName_right, frame_show_right);
                cv::waitKey(1);
            }
        }

    }
}

int Triangulation_skeleton::findIndex(const std::vector<int>& vec, int value)
{
    // Use std::find to get an iterator to the first occurrence of value
    auto it = std::find(vec.begin(), vec.end(), value);

    // Check if the value was found
    if (it != vec.end()) {
        // Return the index of the found element
        return std::distance(vec.begin(), it);
    }
    else {
        // Return -1 if the element was not found
        return -1;
    }
}

void Triangulation_skeleton::triangulation(std::vector<std::vector<std::vector<std::vector<int>>>>& data_left,
    std::vector<std::vector<std::vector<std::vector<int>>>>& data_right,
    std::vector<std::vector<int>>& matchingIndexes,
    std::vector<std::vector<std::vector<std::vector<double>>>>& data_3d,
    std::vector<std::vector<std::vector<std::vector<double>>>>& data_kf_3d,
    std::vector<std::vector<std::vector<std::vector<double>>>>& data_measure_3d)
{
    /**
    * @brief data_left, data_right : {n_human, n_seq, n_joints, (frame,left,top,width,height)}
    * @param[in] matchingIndexes : {n_pairs, (idx_left,idx_right)}
    * @param[out] data_3d : {n_human, n_seq, n_joints, (frame,X,Y,Z)}
    */

    //for all matching data
    int index_left;
    std::vector<std::vector<int>> new_left, new_right; //{n_joints, (frame,left,top,width,height)}
    std::vector<double> left, right;
    std::vector<cv::Point2d> ps_left, ps_right;//{n_joints*n_human,(x,y)}
    std::vector<cv::Point3d> ps_3d;//{n_joints*n_human, (x,y,z)}
    double x_left, x_right, y_left, y_right;
    std::vector<int> index_null;//non-valid data's index.
    int counter_left = 0;
    int counter_right = 0;
    for (std::vector<int>& matchIndex : matchingIndexes) //triangulate 3D points in a index-left ascending way
    {
        //calculate objects
        //left
        new_left = data_left[matchIndex[0]].back(); //{n_joints, (frameIndex, left,top,width,height)}
        for (int j = 0; j < new_left.size(); j++) {//for each joint.
            x_left = (double)new_left[j][1] + (double)new_left[j][3] / 2.0;
            y_left = (double)new_left[j][2] + (double)new_left[j][4] / 2.0;
            ps_left.push_back(cv::Point2d(x_left, y_left));
            if (new_left[j][3] < 0) {//invalid joint. not detected.
                if (std::find(index_null.begin(), index_null.end(), counter_left) == index_null.end()) {
                    index_null.push_back(counter_left);//counter_left is over multiple humans 
                }
            }
            counter_left++;
        }

        //right
        new_right = data_right[matchIndex[1]].back(); //{n_joints, (frameIndex, left,top,width,height)}
        for (int j = 0; j < new_right.size(); j++) {//for each joint.
            x_right = (double)new_right[j][1] + (double)new_right[j][3] / 2.0;
            y_right = (double)new_right[j][2] + (double)new_right[j][4] / 2.0;
            ps_right.push_back(cv::Point2d(x_right, y_right));
            if (new_right[j][3] < 0) {//invalid joint. not detected.
                if (std::find(index_null.begin(), index_null.end(), counter_right) == index_null.end()) {
                    index_null.push_back(counter_right);//counter_left is over multiple humans 
                }
            }
            counter_right++;
        }
    }

    //triangulate points by cv::TriangulatePoints()
    cal3D(ps_left, ps_right, ps_3d);//0:cv::triangulatePoints, 1:stereo triangulation. 0 is better.

    //save data in seq_3d (n_human,n_seq,n_joints,(frame,x,y,z))
    double frame;
    int counter = 0;
    int counter_joints = 0;//total joints.
    double X, Y, Z, X_prev, Y_prev, Z_prev, diff;
    //dframe.
    double dframe;
    //observation
    Eigen::Vector3d observation;
    for (std::vector<int>& matchIndex : matchingIndexes) //triangulate 3D points in the index-left ascending way
    {
        index_left = matchIndex[0];
        new_left = data_left[index_left].back(); //{n_joints, (frameIndex, left,top,width,height)}
        frame = (double)new_left[0][0];//frame
        std::vector<std::vector<double>> seq_last = data_3d[index_left].back();//last sequential data of index_left-th human. {n_joints,(frame,x,y,z)}
        int seq_update = data_3d[index_left].size();
        std::vector<std::vector<double>> joints_3d, joints_3d_kf, joints_3d_measure;//{n_joints,(frame,x,y,z)} 
        for (int j = 0; j < new_left.size(); j++) {//for each joint.
            if (std::find(index_null.begin(), index_null.end(), counter_joints) == index_null.end()) {//counter_joints isn't in index_null. -> valid joint.
                X = ps_3d[counter_joints].x;
                Y = ps_3d[counter_joints].y;
                Z = ps_3d[counter_joints].z;
                joints_3d_measure.push_back({ frame,X,Y,Z });

                //compare with the last data.
                //compare with the last data.
                if (seq_last[j][0] > 0.0) {//last frame is not 0 -> not empty
                    if (seq_update > 10.0) {//(int)(FPS / 10.0)){//(int)(FPS / 10.0)) {//finish warming up period.
                        X_prev = seq_last[j][1];//x
                        Y_prev = seq_last[j][2];//y
                        Z_prev = seq_last[j][3];//z
                        diff = std::sqrt((X - X_prev) * (X - X_prev) + (Y - Y_prev) * (Y - Y_prev) + (Z - Z_prev) * (Z - Z_prev));

                        dframe = frame - kf_3d[index_left][j].frame_last;
                        observation << X, Y, Z;

                        //prediction step
                        kf_3d[index_left][j].predict(dframe);

                        move_threshold_mm_ = dframe * max_move_per_frame_mm_ * std::pow((double)kf_3d[index_left][j].counter_notUpdate, 2);

                        if (diff <= move_threshold_mm_) {//within 1000 mm, 100 cm, 0.05sec
                            if (kf_3d[index_left][j].counter_notUpdate > 1) {//adopt the previous KF-based data.
                                X = (1.0 / (1.0 + (double)std::pow(kf_3d[index_left][j].counter_notUpdate, 1))) * X_prev + (1.0 - (1.0 / (1.0 + (double)std::pow(kf_3d[index_left][j].counter_notUpdate, 1)))) * X;
                                Y = (1.0 / (1.0 + (double)std::pow(kf_3d[index_left][j].counter_notUpdate, 1))) * Y_prev + (1.0 - (1.0 / (1.0 + (double)std::pow(kf_3d[index_left][j].counter_notUpdate, 1)))) * Y;
                                Z = (1.0 / (1.0 + (double)std::pow(kf_3d[index_left][j].counter_notUpdate, 1))) * Z_prev + (1.0 - (1.0 / (1.0 + (double)std::pow(kf_3d[index_left][j].counter_notUpdate, 1)))) * Z;
                                observation << X, Y, Z;
                            }
                            kf_3d[index_left][j].update(observation);//update step

                            //update frame_last
                            kf_3d[index_left][j].frame_last = frame;
                            if (kf_3d[index_left][j].counter_update > 10) {//valid kf model-> update with kalman filter model.
                                X = kf_3d[index_left][j].state_(0);
                                Y = kf_3d[index_left][j].state_(1);
                                Z = kf_3d[index_left][j].state_(2);
                            }
                        }
                        else {//abnormal data. -> adopt previous data.(conservative strategy)

                            //update frame_last
                            kf_3d[index_left][j].frame_last = frame;
                            //adopt previous data.
                            X = kf_3d[index_left][j].state_(0) * (1.0 / (1.0 + std::pow((double)kf_3d[index_left][j].counter_notUpdate, 2))) + (1.0 - 1.0 / (1.0 + std::pow((double)kf_3d[index_left][j].counter_notUpdate, 2))) * X_prev;
                            Y = kf_3d[index_left][j].state_(1) * (1.0 / (1.0 + std::pow((double)kf_3d[index_left][j].counter_notUpdate, 2))) + (1.0 - 1.0 / (1.0 + std::pow((double)kf_3d[index_left][j].counter_notUpdate, 2))) * Y_prev;
                            Z = kf_3d[index_left][j].state_(2) * (1.0 / (1.0 + std::pow((double)kf_3d[index_left][j].counter_notUpdate, 2))) + (1.0 - 1.0 / (1.0 + std::pow((double)kf_3d[index_left][j].counter_notUpdate, 2))) * Z_prev;
                            //X = X_prev;
                            //Y = Y_prev;
                            //Z = Z_prev;
                        }

                        joints_3d_kf.push_back({ frame,kf_3d[index_left][j].state_(0) ,kf_3d[index_left][j].state_(1),kf_3d[index_left][j].state_(2) });
                        joints_3d.push_back({ frame,X,Y,Z });//{frame,x,y,z} 

                    }
                    else {//first 40 frames is warming period.
                        //update kalmanfilter
                        dframe = frame - kf_3d[index_left][j].frame_last;
                        observation << X, Y, Z;
                        //prediction step
                        kf_3d[index_left][j].predict(dframe);
                        //update step
                        kf_3d[index_left][j].update(observation);
                        //update frame_last
                        kf_3d[index_left][j].frame_last = frame;
                        joints_3d_kf.push_back({ frame,kf_3d[index_left][j].state_(0) ,kf_3d[index_left][j].state_(1),kf_3d[index_left][j].state_(2) });
                        joints_3d.push_back({ frame,X,Y,Z });//{frame,x,y,z} 
                    }
                }
                else {//last frame is 0 -> first 3d position.
                    joints_3d.push_back({ frame,X,Y,Z });//push back latest data

                    //initialize Kalman filter
                    //delete a vacant instance.
                    kf_3d[index_left].erase(kf_3d[index_left].begin() + j);
                    //insert the initial one.
                    kf_3d[index_left].insert(kf_3d[index_left].begin() + j, KalmanFilter3D(X, Y, Z, INIT_VX, INIT_VY, INIT_VZ, INIT_AX, INIT_AY, INIT_AZ, NOISE_POS, NOISE_VEL, NOISE_ACC, NOISE_SENSOR));
                    dframe = 0.0;
                    observation << X, Y, Z;
                    kf_3d[index_left][j].predict(dframe);//prediction step
                    kf_3d[index_left][j].update(observation);//update step
                    kf_3d[index_left][j].frame_last = frame;
                    joints_3d_kf.push_back({ frame,kf_3d[index_left][j].state_(0) ,kf_3d[index_left][j].state_(1),kf_3d[index_left][j].state_(2) });
                }
            }
            else {//counter_joints isn't invalid.
                //check the last data.
                if (seq_last[j][0] > 0.0) {//last frame is not 0 and X!=-1.0 -> not empty -> adopt the last data.
                    if (kf_3d[index_left][j].counter_update >= 10) {//adopt kalmanfilter data.
                        dframe = frame - kf_3d[index_left][j].frame_last;

                        //prediction step
                        kf_3d[index_left][j].predict(dframe);

                        kf_3d[index_left][j].frame_last = frame;//update fram
                    }
                    X = seq_last[j][1];//x
                    Y = seq_last[j][2];//y
                    Z = seq_last[j][3];//z

                    joints_3d.push_back({ frame,X,Y,Z });
                    joints_3d_measure.push_back({ frame,seq_last[j][1],seq_last[j][2],seq_last[j][3] });
                    joints_3d_kf.push_back({ frame,kf_3d[index_left][j].state_(0) ,kf_3d[index_left][j].state_(1),kf_3d[index_left][j].state_(2) });
                }
                else {//last frame is 0 -> not updated yet -> add initial values
                    joints_3d.push_back({ 0.0,0.0,0.0,0.0 });//push back latest data
                    joints_3d_kf.push_back({ 0.0,0.0,0.0,0.0 });
                    joints_3d_measure.push_back({ 0.0,0.0,0.0,0.0 });
                }
            }
            counter_joints++;//increment counter_joints.
        }

        //save joints in data_3d
        data_3d[index_left].push_back(joints_3d);
        data_kf_3d[index_left].push_back(joints_3d_kf);
        data_measure_3d[index_left].push_back(joints_3d_measure);
    }
}

void Triangulation_skeleton::cal3D(std::vector<cv::Point2d>& pts_left, std::vector<cv::Point2d>& pts_right, std::vector<cv::Point3d>& results)
{
    dlt(pts_left, pts_right, results);
}

void Triangulation_skeleton::dlt(std::vector<cv::Point2d>& points_left, std::vector<cv::Point2d>& points_right, std::vector<cv::Point3d>& results)
{
    /**
    * @brief calculate 3D points with DLT method
    * @param[in] points_left, points_right {n_data,(xCenter,yCenter)}
    * @param[out] reuslts 3D points storage. shape is like (n_data, (x,y,z))
    */
    cv::Mat points_left_mat(points_left);
    cv::Mat undistorted_points_left_mat;
    cv::Mat points_right_mat(points_right);
    cv::Mat undistorted_points_right_mat;

    // Undistort the points
    cv::undistortPoints(points_left_mat, undistorted_points_left_mat, cameraMatrix_left, distCoeffs_left);
    cv::undistortPoints(points_right_mat, undistorted_points_right_mat, cameraMatrix_right, distCoeffs_right);

    // Reproject normalized coordinates to pixel coordinates
    cv::Mat normalized_points_left(undistorted_points_left_mat.rows, 1, CV_64FC2);
    cv::Mat normalized_points_right(undistorted_points_right_mat.rows, 1, CV_64FC2);

    for (int i = 0; i < undistorted_points_left_mat.rows; ++i) {
        double x, y;
        x = undistorted_points_left_mat.at<cv::Vec2d>(i, 0)[0];
        y = undistorted_points_left_mat.at<cv::Vec2d>(i, 0)[1];
        normalized_points_left.at<cv::Vec2d>(i, 0)[0] = cameraMatrix_left.at<double>(0, 0) * x + cameraMatrix_left.at<double>(0, 2);
        normalized_points_left.at<cv::Vec2d>(i, 0)[1] = cameraMatrix_left.at<double>(1, 1) * y + cameraMatrix_left.at<double>(1, 2);

        x = undistorted_points_right_mat.at<cv::Vec2d>(i, 0)[0];
        y = undistorted_points_right_mat.at<cv::Vec2d>(i, 0)[1];
        normalized_points_right.at<cv::Vec2d>(i, 0)[0] = cameraMatrix_right.at<double>(0, 0) * x + cameraMatrix_right.at<double>(0, 2);
        normalized_points_right.at<cv::Vec2d>(i, 0)[1] = cameraMatrix_right.at<double>(1, 1) * y + cameraMatrix_right.at<double>(1, 2);
    }

    // Output matrix for the 3D points
    cv::Mat triangulated_points_mat;

    // Triangulate points
    cv::triangulatePoints(projectMatrix_left, projectMatrix_right, normalized_points_left, normalized_points_right, triangulated_points_mat);
    //cv::triangulatePoints(projectMatrix_left, projectMatrix_right, undistorted_points_left_mat, undistorted_points_right_mat, triangulated_points_mat);

    // Convert homogeneous coordinates to 3D points
    triangulated_points_mat = triangulated_points_mat.t();
    cv::convertPointsFromHomogeneous(triangulated_points_mat.reshape(4), triangulated_points_mat);

    // Access triangulated 3D points
    results.clear();

    for (int i = 0; i < triangulated_points_mat.rows; i++) {
        cv::Point3d point;
        point.x = triangulated_points_mat.at<double>(i, 0);
        point.y = triangulated_points_mat.at<double>(i, 1);
        point.z = triangulated_points_mat.at<double>(i, 2);
        results.push_back(point);
    }

    // Convert from camera coordinate to robot base coordinate
    for (auto& point : results) {
        double x = point.x;
        double y = point.y;
        double z = point.z;
        point.x = transform_cam2base.at<double>(0, 0) * x + transform_cam2base.at<double>(0, 1) * y + transform_cam2base.at<double>(0, 2) * z + transform_cam2base.at<double>(0, 3);
        point.y = transform_cam2base.at<double>(1, 0) * x + transform_cam2base.at<double>(1, 1) * y + transform_cam2base.at<double>(1, 2) * z + transform_cam2base.at<double>(1, 3);
        point.z = transform_cam2base.at<double>(2, 0) * x + transform_cam2base.at<double>(2, 1) * y + transform_cam2base.at<double>(2, 2) * z + transform_cam2base.at<double>(2, 3);
    }
}

bool Triangulation_skeleton::check_human_pose(skeleton2robot& send_data) {
    /**
    * @brief check human pose and add head and foot.
    */

    //compensate lost data and add Head and foot from LS and RS -> n_joints: 6 -> 8
    bool success = false;
    int i = 0;
    double frame_human, frame_current;
    double frame = 0.0;
    //initialze
    joints_void.clear();
    while (true) {//for each human
        if (i >= pose_human.size())//i is larger than pose_human.size();
            break;
        std::vector<double> head, foot;
        std::vector<int> index_void;
        double x = 0.0; double y = 0.0;
        double counter_joint = 0.0;

        bool bool_ls = false;
        bool bool_rs = false;

        //check human tracked joints.
        for (int j = 0; j < 3; j++) {//for each joints

            //change the unit.
            //left joint
            pose_human[i][2 * j][1] /= 1000.0;//convert [mm] to [m]
            pose_human[i][2 * j][2] /= 1000.0;//convert [mm] to [m]
            pose_human[i][2 * j][3] /= 1000.0;//convert [mm] to [m]
            //right joint
            pose_human[i][2 * j + 1][1] /= 1000.0;//convert [mm] to [m]
            pose_human[i][2 * j + 1][2] /= 1000.0;//convert [mm] to [m]
            pose_human[i][2 * j + 1][3] /= 1000.0;//convert [mm] to [m]

            if (pose_human[i][2 * j][0] > 0) {//LEFT : valid

                if (!bool_ls) {
                    x += pose_human[i][2 * j][1];
                    y += pose_human[i][2 * j][2];
                    bool_ls = true;
                    counter_joint += 1.0;
                }

                if (pose_human[i][2 * j + 1][0] > 0) {//RIGHT : valid
                    if (!bool_rs) {
                        x += pose_human[i][2 * j + 1][1];
                        y += pose_human[i][2 * j + 1][2];
                        bool_rs = true;
                        counter_joint += 1.0;
                    }
                }
                else {//RIGHT : invalid -> compensate with the left joint.
                    pose_human[i][2 * j + 1][0] = pose_human[i][2 * j][0];
                    pose_human[i][2 * j + 1][1] = pose_human[i][2 * j][1] - 0.01;
                    pose_human[i][2 * j + 1][2] = pose_human[i][2 * j][2];
                    pose_human[i][2 * j + 1][3] = pose_human[i][2 * j][3];
                }
            }
            else {//LEFT is invalid.
                if (pose_human[i][2 * j + 1][0] > 0) {//RIGHT : valid
                    if (!bool_rs) {
                        x += pose_human[i][2 * j + 1][1];
                        y += pose_human[i][2 * j + 1][2];
                        bool_rs = true;
                        counter_joint += 1.0;
                    }

                    //compensate with the right joint.
                    pose_human[i][2 * j][0] = pose_human[i][2 * j + 1][0];
                    pose_human[i][2 * j][1] = pose_human[i][2 * j + 1][1] + 0.01;
                    pose_human[i][2 * j][2] = pose_human[i][2 * j + 1][2];
                    pose_human[i][2 * j][3] = pose_human[i][2 * j + 1][3];
                }
                else {//both are not valid.
                    index_void.push_back(2 * j);
                    index_void.push_back(2 * j + 1);
                }
            }
        }

        //calculate head and joints.
        if (counter_joint == 2.0) {//both LS and RS found.
            x /= counter_joint;
            y /= counter_joint;
            head = std::vector<double>{ frame,x,y,z_head_ };
            foot = std::vector<double>{ frame,x,y,z_foot_ };
        }
        else if (counter_joint == 1.0) {//Either LS or RS found
            head = std::vector<double>{ frame,x,y,z_head_ };
            foot = std::vector<double>{ frame,x,y,z_foot_ };
        }
        //others -> no detections.

        //add head and foot.
        if (index_void.size() < pose_human[i].size()) {//some joints are detected.
            pose_human[i].push_back(head);
            pose_human[i].push_back(foot);

            //add index_void to joints_void
            joints_void.push_back(index_void);
            int n_void = index_void.size() / 2;//the number of non-detected joints.

            if (n_void == 1) {//one pair of joints is failed to detect
                if (index_void[0] / 2 == 0) {//shoulder is not detected.
                    //ls<-le
                    pose_human[i][0][0] = pose_human[i][2][0];
                    pose_human[i][0][1] = pose_human[i][2][1];
                    pose_human[i][0][2] = pose_human[i][2][2];
                    pose_human[i][0][3] = pose_human[i][2][3] + 0.01;
                    //rs<-re
                    pose_human[i][1][0] = pose_human[i][3][0];
                    pose_human[i][1][1] = pose_human[i][3][1];
                    pose_human[i][1][2] = pose_human[i][3][2];
                    pose_human[i][1][3] = pose_human[i][3][3] + 0.01;
                }
                else if (index_void[0] / 2 == 1) {//elbow is not detected.
                    //le<-ls
                    pose_human[i][2][0] = pose_human[i][0][0];
                    pose_human[i][2][1] = pose_human[i][0][1];
                    pose_human[i][2][2] = pose_human[i][0][2];
                    pose_human[i][2][3] = pose_human[i][0][3] - 0.01;
                    //re<-rs
                    pose_human[i][3][0] = pose_human[i][1][0];
                    pose_human[i][3][1] = pose_human[i][1][1];
                    pose_human[i][3][2] = pose_human[i][1][2];
                    pose_human[i][3][3] = pose_human[i][1][3] - 0.01;
                }
                else if (index_void[0] / 2 == 2) {//wrist is not detected.
                    //lw<-le
                    pose_human[i][4][0] = pose_human[i][2][0];
                    pose_human[i][4][1] = pose_human[i][2][1];
                    pose_human[i][4][2] = pose_human[i][2][2];
                    pose_human[i][4][3] = pose_human[i][2][3] - 0.01;
                    //rw<-re
                    pose_human[i][5][0] = pose_human[i][3][0];
                    pose_human[i][5][1] = pose_human[i][3][1];
                    pose_human[i][5][2] = pose_human[i][3][2];
                    pose_human[i][5][3] = pose_human[i][3][3] - 0.01;
                }
            }
            else if (n_void == 2) {//two pairs of joints are failed to detect.
                if (index_void[0] / 2 == 0 && index_void[2] / 2 == 1) {//shoulder and elbow is not detected.(0,1) -> 2
                    //shoulder
                    //ls<-lw
                    pose_human[i][0][0] = pose_human[i][4][0];
                    pose_human[i][0][1] = pose_human[i][4][1];
                    pose_human[i][0][2] = pose_human[i][4][2];
                    pose_human[i][0][3] = pose_human[i][4][3] + 0.02;
                    //rs<-rw
                    pose_human[i][1][0] = pose_human[i][5][0];
                    pose_human[i][1][1] = pose_human[i][5][1];
                    pose_human[i][1][2] = pose_human[i][5][2];
                    pose_human[i][1][3] = pose_human[i][5][3] + 0.02;

                    //elbow
                    //le<-rw
                    pose_human[i][2][0] = pose_human[i][4][0];
                    pose_human[i][2][1] = pose_human[i][4][1];
                    pose_human[i][2][2] = pose_human[i][4][2];
                    pose_human[i][2][3] = pose_human[i][4][3] + 0.01;
                    //re<-rw
                    pose_human[i][3][0] = pose_human[i][5][0];
                    pose_human[i][3][1] = pose_human[i][5][1];
                    pose_human[i][3][2] = pose_human[i][5][2];
                    pose_human[i][3][3] = pose_human[i][5][3] + 0.01;
                }
                else if (index_void[0] / 2 == 0 && index_void[2] / 2 == 2) {//shoulder and wrist is not detected.(0,2) -> 1
                    //shoulder
                    //ls<-le
                    pose_human[i][0][0] = pose_human[i][2][0];
                    pose_human[i][0][1] = pose_human[i][2][1];
                    pose_human[i][0][2] = pose_human[i][2][2];
                    pose_human[i][0][3] = pose_human[i][2][3] + 0.01;
                    //rs<-re
                    pose_human[i][1][0] = pose_human[i][3][0];
                    pose_human[i][1][1] = pose_human[i][3][1];
                    pose_human[i][1][2] = pose_human[i][3][2];
                    pose_human[i][1][3] = pose_human[i][3][3] + 0.01;

                    //wrist
                    //lw<-le
                    pose_human[i][4][0] = pose_human[i][2][0];
                    pose_human[i][4][1] = pose_human[i][2][1];
                    pose_human[i][4][2] = pose_human[i][2][2];
                    pose_human[i][4][3] = pose_human[i][2][3] - 0.01;
                    //rw<-re
                    pose_human[i][5][0] = pose_human[i][3][0];
                    pose_human[i][5][1] = pose_human[i][3][1];
                    pose_human[i][5][2] = pose_human[i][3][2];
                    pose_human[i][5][3] = pose_human[i][3][3] - 0.01;
                }
                else if (index_void[0] / 2 == 1 && index_void[2] / 2 == 2) {//elbow and wrist is not detected.(1,2) -> 0
                    //elbow
                    //le<-ls
                    pose_human[i][2][0] = pose_human[i][0][0];
                    pose_human[i][2][1] = pose_human[i][0][1];
                    pose_human[i][2][2] = pose_human[i][0][2];
                    pose_human[i][2][3] = pose_human[i][0][3] - 0.01;
                    //re<-rs
                    pose_human[i][3][0] = pose_human[i][1][0];
                    pose_human[i][3][1] = pose_human[i][1][1];
                    pose_human[i][3][2] = pose_human[i][1][2];
                    pose_human[i][3][3] = pose_human[i][1][3] - 0.01;

                    //wrist
                    //lw<-ls
                    pose_human[i][4][0] = pose_human[i][0][0];
                    pose_human[i][4][1] = pose_human[i][0][1];
                    pose_human[i][4][2] = pose_human[i][0][2];
                    pose_human[i][4][3] = pose_human[i][0][3] - 0.02;
                    //rw<-rs
                    pose_human[i][5][0] = pose_human[i][1][0];
                    pose_human[i][5][1] = pose_human[i][1][1];
                    pose_human[i][5][2] = pose_human[i][1][2];
                    pose_human[i][5][3] = pose_human[i][1][3] - 0.02;
                }
            }

            i++;
            success = true;//found human.

        }
        else {//no joints detected.
            pose_human.erase(pose_human.begin() + i);//erase human data.
        }
    }

    if (success) {
        send_data.frame_current = frame_current;
        send_data.frame_human = frame_human;
        send_data.joints_void = joints_void;
        send_data.pose_human = pose_human;
        return success;
    }
    else
        return success;
}