#include "../include/mot.h"
#include "../include/detector/yolo_batch.h"

void MOT::main()
{
    auto start_whole = std::chrono::high_resolution_clock::now();
    int count_yolo = 0;
    while (true)
    {
        if (!q_yolo2buffer.empty()) {
            if (count_yolo >= 4)
                break;
            q_yolo2buffer.pop();
            count_yolo++;

        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::cout << "start saving sequential data" << std::endl;

	//make a counter.
    int counterIteration = 0;
    int counterFinish = 0;
    int counterNextIteration = 0;
    int counter_compulsiveFinish = 0;
    double t_elapsed = 0.0;

    while (true) // continue until finish
    {
        if (!q_finish_mot.empty()){
			q_finish_tracking.push(true);
			q_finish_yolo.push(true);
			q_finish_mot.pop();
            break;
		}

        counterFinish = 0;

		auto start = std::chrono::high_resolution_clock::now();
        
		//data extraction.
		//Data from Tracking module.
		if (!q_tracking2mot_left.empty() && !q_tracking2mot_right.empty()){
			trackers2mot_left = q_tracking2mot_left.front();
			trackers2mot_right = q_tracking2mot_right.front();
			q_tracking2mot_left.pop();
			q_tracking2mot_right.pop();
		}

		//Data from YOLO module.
        if (!q_yolo2mot.empty()) {//get data from YOLO. -> postprocess first.
			count_yolo++;
            //get data
            yolo2mot = q_yolo2mot.front();
            q_yolo2mot.pop();
            if (count_yolo >= 5) {
                rois = yolo2mot.rois;
                labels = yolo2mot.labels;
                scores = yolo2mot.scores;
                frameYolo = yolo2mot.frame;
                frameIndex = yolo2mot.frameIndex;

                //postprocess
                //split detections into left and right.
                //initialize
                roi_left.clear(); roi_right.clear(); class_left.clear(); class_right.clear(); scores_left.clear(); scores_right.clear();
                //classify data.
                YOLODetect_batch::roiSetting(rois, labels, roi_left, class_left, roi_right, class_right, scores_left, scores_right); //separate detection into left and right
            }

            if (!roi_right.empty() || !roi_left.empty())//Either left or right is not empty.
            {
                //left
                counterNextIteration = 0; //reset counterNextIteration
                std::vector<int> index_delete_left, index_delete_right;
                YOLODetect_batch::cvtToTrackersYOLO(roi_left, class_left,scores_left, frameIndex, frameYolo, trackersYOLO_left);
                YOLODetect_batch::cvtToTrackersYOLO(roi_right, class_right,scores_right, frameIndex, frameYolo, trackersYOLO_right);
			}
		}

		//associate data. -> trackers2mot, trackersYOLO, trackersMOT


		//newdata is avaialble
		if (!newdata_left.empty() && !newdata_right.empty()) {
			organize(newdata_left, true, seqData_left, kfData_left,
				kalmanVector_left, extrapolation_left, saveData_left, saveKFData_left, index_delete_left, q_seq2tri_left);
			//right
			organize(newdata_right, false, seqData_right, kfData_right,
				kalmanVector_right, extrapolation_right, saveData_right, saveKFData_right, index_delete_right, q_seq2tri_right);
			//thread_organize_left.join(); //wait for left data to finish
			//procedure to make data_3d.size() corresponding to seqData_left.size().
			//initialize data_3d for its size to correspond to the seqData_left.size().
			if (!seqData_left.empty() && data_3d.empty()) {//initial
				std::vector<std::vector<std::vector<double>>> initial(seqData_left.size(), std::vector<std::vector<double>>(1, std::vector<double>(5, 0.0)));//{frame,label,x,y,z};initialize with 0.0
				std::vector<std::vector<std::vector<double>>> initial_target(seqData_left.size(), std::vector<std::vector<double>>(1, std::vector<double>(8, 0.0)));//{frame,label,x,y,z,nx,ny,nz};initialize with 0.0
				std::vector<std::vector<std::vector<double>>> initial_params(seqData_left.size(), std::vector<std::vector<double>>(1, std::vector<double>(n_features, 0.0)));
				data_3d = initial;//{frame,label,x,y,z}
				targets = initial_target;//{frame,label,x,y,z,nx,ny,nz}
				params = initial_params; //frame, label, a_x, b_x, c_x, a_y, b_y, c_y, a_z, b_z, 
				if (method_prediction == 1) {//RLS method
					instances_rls = std::vector<rls>(seqData_left.size(), init_rls);
				}
			}

			//when data is deleted.
			//delete idx_match_prev according to index_delete. index_delete is in descending order.
			if (!index_delete_left.empty() || !index_delete_right.empty()) {//delete idx_match_prev.
				if (!index_delete_left.empty()) {//left list
					int idx;
					for (int i = 0; i < index_delete_left.size(); i++) {//for each index of the left.
						idx = index_delete_left[i];
						//previous match index
						match.idx_match_prev.erase(match.idx_match_prev.begin() + idx);
						//params&targets
						if (params[idx].size() > 1) {//not empty -> save in targets_save
							data_3d_save.push_back(data_3d[idx]);//save data.
							params_save.push_back(params[idx]);
							targets_save.push_back(targets[idx]);
						}

						//delete
						data_3d.erase(data_3d.begin() + idx);
						params.erase(params.begin() + idx);
						targets.erase(targets.begin() + idx);
						if (method_prediction == 1) {//RLS
							instances_rls.erase(instances_rls.begin() + idx);
						}
					}
				}

				if (!index_delete_right.empty() && !match.idx_match_prev.size() >= 1) {//right list
					int idx_delete;
					for (int i = 0; i < index_delete_right.size(); i++) {//for each index of the right
						idx_delete = findIndex(match.idx_match_prev, index_delete_right[i]);//find element, index_delete_right[i], from match.idx_match_prev.
						if (idx_delete >= 0) {//found
							match.idx_match_prev[idx_delete] = -1;//convert to -1.
						}
					}
				}
			}

			//when new data is added.
			if (seqData_left.size() > data_3d.size()) {//new data is added to the seqData_left -> prepare empty data in data_3d.

				//data_3d
				while (seqData_left.size() > data_3d.size())
					data_3d.push_back(initial_add);//add empty storage.

				//targets
				while (seqData_left.size() > targets.size())
					targets.push_back(initial_add_target);//add empty storage.

				//params
				while (seqData_left.size() > params.size()) {
					params.push_back(initial_add_params);//add empty storage.
					if (method_prediction == 1)//RLS method
						instances_rls.push_back(init_rls);
				}
			}
			//std::cout << "mot-4" << std::endl;
			////3d triangulation.
			//auto start_tri = std::chrono::high_resolution_clock::now();
			if (!seqData_left.empty() && !seqData_right.empty()) {
				//std::cout << "3" << std::endl;
				//triangulate points.
				std::vector<std::vector<int>> matchingIndexes; //list for matching indexes : {n_pairs, (idx_left,idx_right)}
				//matching when number of labels increase -> seqData_left.size()>num_obj_left || seqData_right.size()>num_obj_right;
				//matching objects in 2 images
				match.main(seqData_left, seqData_right, tri.oY_left, tri.oY_right, matchingIndexes);
				//std::cout << "mot-5" << std::endl;
				//std::cout << "0" << std::endl;
				if (!matchingIndexes.empty()) {
					//std::cout << "4" << std::endl;
					matching_save.push_back(matchingIndexes);
					//triangulate 3D points
					tri.triangulation(seqData_left, seqData_right, matchingIndexes, data_3d);//[m]
					//std::cout << "mot-6" << std::endl;
					//predict targets.
					double frame_latest, label;
					std::vector<double> param_tmp;
					std::vector<Seq2robot> params_send;//predicted trajectory parameters to send to RobotControl.cpp.
					//std::cout << "1" << std::endl;
					for (int i = 0; i < data_3d.size(); i++) {//for each objects
						frame_latest = data_3d[i].back()[0];
						label = data_3d[i].back()[1];
						if (method_prediction == 0) {//ordinary Least squares method
							if (data_3d[i].size() >= 3) {//sufficient data. minimum nubmer of data is 3.
								Seq2robot seq2robot;
								prediction.predictTargets(i, depth_target, data_3d[i], targets);
								//push params for trajectory prediction. quadratic fitting.
								param_tmp = std::vector<double>{ frame_latest,label,
									prediction.coefX(1),prediction.coefX(2),
									prediction.coefY(1),prediction.coefY(2),
									prediction.coefZ(0),prediction.coefZ(1),prediction.coefZ(2)
								};

								if (!params.empty()) {

									if (params[i][0][0] == 0.0)//first data.
										params[i][0] = param_tmp;
									else//after first.
										params[i].push_back(param_tmp);

									if (params[i].size() >= counter_update_params_) {//valid prediction. params:{#(objects),#(seq),(frame_target,label,coef_x,coef_y,coef_z)}
										seq2robot.frame_current = frame_latest;
										seq2robot.label = label;
										//seq2robot.param_x = Eigen::Vector2d(prediction.coefX(1), prediction.coefX(2));
										//seq2robot.param_y = Eigen::Vector2d(prediction.coefY(1), prediction.coefY(2));
										//seq2robot.param_z = Eigen::Vector3d(prediction.coefZ(0), prediction.coefZ(1), prediction.coefZ(2));
										seq2robot.param_x = Eigen::Vector2d(params[i].back()[2], params[i].back()[3]);
										seq2robot.param_y = Eigen::Vector2d(params[i].back()[4], params[i].back()[5]);
										seq2robot.param_z = Eigen::Vector3d(params[i].back()[6], params[i].back()[7], params[i].back()[8]);
										seq2robot.pos_current = data_3d[i].back();//latest data.
										params_send.push_back(seq2robot);
									}
								}
							}
						}
						else if (method_prediction == 1) {//recursive least squares method.

							if (data_3d[i].back()[0] > 0) {
								Seq2robot seq2robot;
								std::vector<Eigen::VectorXd> coeffs = prediction.predictTargets_rls(i, depth_target, data_3d[i], instances_rls, targets);
								//save data.
								param_tmp = std::vector<double>{ frame_latest,label };
								//coeff_x
								for (int i_para = 0; i_para < coeffs.size(); i_para++) {
									for (int i_ele = 0; i_ele < coeffs[i_para].size(); i_ele++) {
										param_tmp.push_back(coeffs[i_para](i_ele));
									}
								}

								if (!params.empty()) {
									if (params[i][0][0] == 0.0)//first data.
										params[i][0] = param_tmp;
									else//after first.
										params[i].push_back(param_tmp);

									if (params[i].size() >= counter_update_params_) {//valid prediction. params:{#(objects),#(seq),(frame_target,label,coef_x,coef_y,coef_z)}
										seq2robot.frame_current = frame_latest;
										seq2robot.label = label;
										seq2robot.param_x = coeffs[0];
										seq2robot.param_y = coeffs[1];
										seq2robot.param_z = coeffs[2];
										seq2robot.pos_current = data_3d[i].back();//latest data.
										params_send.push_back(seq2robot);
									}
								}
							}
						}
					}
					//std::cout << "mot-7" << std::endl;
					//send predicted trajectory parameters.
					if (!params_send.empty()) {
						//determine target pose.
						std::vector<double> pose_target(6, 0.0);
						InfoParams param_target;
						double frame_target;
						bool bool_back = false;
						double frame_target_return = decide_target(params_send, frame_latest, pose_target, param_target, frame_target, bool_back);

						//std::cout << "bool_back=" << bool_back << std::endl;
						if (bool_backward_ && bool_back) {//make the robot to move backward
							/***Preprocess***/
							boolImgs = ut_robot.getFrameFromQueueRobot(frameIndex);
							if (boolImgs) {
								if (frame_latest < frameIndex)
									frame_latest = frameIndex;
							}
							double time_current = frame_latest / (double)FPS;//time[sec]
							double time_target = frame_target / (double)FPS;//
							double dt_interval_ = 0.02;//check target pose from (time_target - 0.1) sec to (time_target+0.04) sec.
							double time_candidate;
							double speed;
							double speed_min_ = speed_max_ * (double)FPS;//[m/frame]*[frame/sec]->[m/sec]
							//trajectory parameters.
							Eigen::Vector2d px = param_target.param_x;
							Eigen::Vector2d py = param_target.param_y;
							Eigen::Vector3d pz = param_target.param_z;

							std::vector<double> pose_robot = ee_current_;//get current robot end-effector pose.
							double x_ee = pose_robot[0];
							double y_ee = pose_robot[1];
							double z_ee = pose_robot[2];
							double xT, yT, zT, nxT, nyT, nzT, norm_rot;
							double time_target_tmp = time_target;
							/******/

							/***Check the close targets' position.***/
							//std::cout << "time_current=" << time_current << ", time_target=" << time_target <<", x_target="<<pose_target[0]<<", y_target="<<pose_target[1]<<", z_target="<<pose_target[2] << std::endl;
							for (int i = -3; i < 11; i++) {//t-0.4 sec ~ t+0.2 sec
								time_candidate = time_target + (double)i * dt_interval_;//candidate catching time
								xT = px(0) * time_candidate + px(1);
								yT = py(0) * time_candidate + py(1);
								zT = pz(0) * time_candidate * time_candidate + pz(1) * time_candidate + pz(2);
								nxT = px(0);
								nyT = py(0);
								nzT = 2.0 * pz(0) * time_candidate + pz(1);
								norm_rot = std::sqrt(nxT * nxT + nyT * nyT + nzT * nzT);
								if (norm_rot > 0.0) {
									nxT = -1.0 * nxT / norm_rot;
									nyT = -1.0 * nyT / norm_rot;
									nzT = -1.0 * nzT / norm_rot;
								}
								//consider cup's size
								xT = xT - std::max(-1.0 * h_cup_, std::min(h_cup_, nxT * h_cup_));
								yT = yT - std::max(-1.0 * h_cup_, std::min(h_cup_, nyT * h_cup_));
								zT = zT - std::max(-1.0 * h_cup_, std::min(h_cup_, nzT * h_cup_));
								//std::cout << "time_current=" << time_current << ", time_target=" << time_candidate <<",xT="<<xT<<", yT="<<yT<<",zT="<<zT << std::endl;
								if ((x_min_ <= xT && xT <= x_max_) && (y_min_ <= yT && yT <= y_max_) && (z_min_ <= zT && zT <= z_max_)) {
									speed = std::sqrt((x_ee - xT) * (x_ee - xT) + (y_ee - yT) * (y_ee - yT) + (z_ee - zT) * (z_ee - zT));//[m]
									speed /= (time_candidate - time_current);//[m/sec]
									//std::cout << "speed=" << speed<<", z_target="<<zT<< std::endl;
									if (0.0 < speed && speed < speed_min_) {//lower speed -> good -> change the target position
										pose_target = std::vector<double>{ xT,yT,zT,nxT,nyT,nzT };
										speed_min_ = speed;
										time_target_tmp = time_candidate;
										//std::cout << "Updated minimum speed=" << speed_min_<<", target_time="<<time_target_tmp << std::endl;
									}
								}
							}
							frame_target = (int)(time_target_tmp * (double)FPS);
							infoTarget_.delta_frame = frame_target - frame_latest;
							infoTarget_.p_target = pose_target;
						}
						/***End fine search***/

						/***Send target position to robot_control.cpp***/
						if (frame_target_return > 0.0) {
							if ((x_min_ <= pose_target[0] && pose_target[0] <= x_max_) && (y_min_ <= pose_target[1] && pose_target[1] <= y_max_) && (z_min_ <= pose_target[2] && pose_target[2] <= z_max_)) {
								Seq2robot_send new_target;
								new_target.frame_current = frame_latest;
								new_target.frame_target = frame_target;
								new_target.pose_target = pose_target;
								new_target.param_target = param_target;
								new_target.bool_back = bool_back;
								q_seq2robot.push(new_target);
								//std::cout << "SEND DATA :: x_target=" << pose_target[0] << ", y_target=" << pose_target[1] << ", z_target=" << pose_target[2] << std::endl;
							}
							else {//reset target information
								infoTarget_.p_target.clear();
							}
						}
						//q_trajectory_params.push(params_send);
					}

					//comparison of data_3d and data_3d_save. based on the predicted params. time-sequential-based comparison.
					if (!params.empty() && !params_save.empty()) {
						//std::cout << "7" << std::endl;
						std::vector<double> idxes_current, idxes_prev;
						std::vector<std::vector<double>> costMatrix;
						int counter_iteration = 0;
						for (int i_cur = 0; i_cur < params.size(); i_cur++) {//for each object. 
							if (params[i_cur][0][0] > 0.0) {//current parameters were calculated. <- (frame,laebel,params)
								std::vector<double> cost_row;
								idxes_current.push_back(i_cur);
								for (int i_prev = 0; i_prev < params_save.size(); i_prev++) {//for each object in save storage.
									if (params_save[i_prev][0][0] > 0.0) {
										if (counter_iteration == 0)
											idxes_prev.push_back(i_prev);
										//tracker id
										label_latest = params[i_cur].back()[1];
										label_prev = params_save[i_prev].back()[1];
										cost_label = compareID(label_latest, label_prev);
										//parameters
										params_latest = params[i_cur].back();
										params_prev_latest = params_save[i_prev].back();
										cost_params = compareParams(params_latest, params_prev_latest);
										//total
										cost_total = cost_label + cost_params;
										cost_row.push_back(cost_total);
									}
								}
								counter_iteration++;
								if (!cost_row.empty())
									costMatrix.push_back(cost_row);
							}
						}
						//std::cout << "8" << std::endl;
						//match predictions based on hungarian algorithm.
						if (!costMatrix.empty()) {
							//std::cout << "3-0" << std::endl;
							std::vector<int> assignment;
							double cost = HungAlgo.Solve(costMatrix, assignment);
							std::vector<int> index_delete;//index list of the YOLO detections to be deleted.
							double frame_prev;
							int index_current, index_storage;
							//assign data according to indexList_tm, indexList_yolo and assign
							for (unsigned int x = 0; x < assignment.size(); x++) {//for each candidate from 0 to the end of the candidates. ascending way.
								index_current = idxes_current[x];//current data index
								//std::cout << "index_current=" << index_current << std::endl;
								if (assignment[x] >= 0) {//matching tracker is found.
									if (costMatrix[x][assignment[x]] < Cost_params_max) {//under max cost
										index_storage = idxes_prev[assignment[x]];//storage index
										index_delete.push_back(index_storage);
										//merge data_3d_save with data_3d.
										concatenateVectors(data_3d_save[index_storage], data_3d[index_current]);
										data_3d[index_current] = data_3d_save[index_storage];
										//std::cout << "*" << std::endl;
										//targets
										concatenateVectors(targets_save[index_storage], targets[index_current]);
										targets[index_current] = targets_save[index_storage];
										//std::cout << "**" << std::endl;
										//params
										concatenateVectors(params_save[index_storage], params[index_current]);
										params[index_current] = params_save[index_storage];
										//std::cout << "***" << std::endl;
									}
								}
							}
							//std::cout << "3-1" << std::endl;
							if (!index_delete.empty()) {
								//sort index_delete in descending way.
								//delete data from data_3d_save,params_save,tarets_save.
								std::sort(index_delete.rbegin(), index_delete.rend());
								for (int& k : index_delete) {
									data_3d_save.erase(data_3d_save.begin() + k);
									targets_save.erase(targets_save.begin() + k);
									params_save.erase(params_save.begin() + k);
								}
							}

						}
					}
				}
			}
		}

		auto stop_tri = std::chrono::high_resolution_clock::now();
		auto stop = std::chrono::high_resolution_clock::now();
		//auto duration_tri = std::chrono::duration_cast<std::chrono::microseconds>(stop_tri - start_tri);
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

		t_elapsed += duration.count();
		counterIteration++;
		//std::cout << "time taken by 3d positioning=" << duration_tri.count() << " microseconds" << std::endl;
		if (counterIteration % 50 == 0)
			std::cout << " !!!!!! Time taken by Sequence and KF in both frames : " << duration.count() << " microseconds !!!!!!" << std::endl;
            }
        }
        else
        {
            //through
        }

    }
    std::cout << "*** Sequence.cpp (MOT) :: process speed = " << counterIteration / t_elapsed * 1000000.0 << " Hz ***" << std::endl;
    if (!seqData_left.empty()) {
        //save data
        for (int i = 0; i < seqData_left.size(); i++) {
            saveData_left.push_back(seqData_left[i]);
            saveKFData_left.push_back(kfData_left[i]);
        }
    }

    if (!seqData_right.empty()) {
        //save data
        for (int i = 0; i < seqData_right.size(); i++) {
            saveData_right.push_back(seqData_right[i]);
            saveKFData_right.push_back(kfData_right[i]);
        }
    }

    std::cout << "sequential data" << std::endl;
    std::cout << "LEFT :: saveData " << std::endl;
    utSeq.checkSeqData(saveData_left, file_seq_left);
    std::cout << "LEFT :: kfData " << std::endl;
    utSeq.checkSeqData(saveKFData_left, file_kf_left);
    std::cout << "RIGHT :: saveData " << std::endl;
    utSeq.checkSeqData(saveData_right, file_seq_right);
    std::cout << "RIGHT :: kfData " << std::endl;
    utSeq.checkSeqData(saveKFData_right, file_kf_right);
    std::cout << "RIGHT ;; Ball " << std::endl;

    std::cout << "Matching" << std::endl;
    if (!matching_save.empty()) utSeq.saveMatching(matching_save, file_match);
    //triangulation data
    if (!data_3d.empty()) {
        for (int i = 0; i < data_3d.size(); i++) {//for each object
            if (data_3d[i].size() > 1) {//there is a data.
                data_3d_save.push_back(data_3d[i]);
            }
        }
        data_3d.clear();//delete all the data
    }
    if (!data_3d_save.empty()) {
        std::cout << "***triangulation data*** data_3d_save.size()=" << data_3d_save.size() << std::endl;
        utSeq.save3d_mot(data_3d_save, file_3d);
    }

    //target prediction.
    if (!targets.empty()) {
        for (int i = 0; i < targets.size(); i++) {//for each object
            if (targets[i].size() > 1) {//there is a data.
                targets_save.push_back(targets[i]);
            }
        }
        targets.clear();//delete all the data
    }
    if (!targets_save.empty()) {
        std::cout << "***targets***" << std::endl;
        utSeq.save3d_mot(targets_save, file_target);
    }

    //prediction params.
    if (!params.empty()) {
        for (int i = 0; i < params.size(); i++) {//for each object
            if (params[i].size() > 1) {//there is a data.
                params_save.push_back(params[i]);
            }
        }
        params.clear();//delete all the data
    }
    if (!params_save.empty()) {
        std::cout << "***paramss***" << std::endl;
        utSeq.save_params(params_save, file_params);
    }
}


double Sequence::decide_target(std::vector<Seq2robot>& params_trajectory, double& frame_current, std::vector<double>& pose_target, InfoParams& param_target, double& frame_target_current, bool& bool_back) {

    /***Variable setting***/
    if (!wrists.empty())
        wrists_human = wrists;//get wrists

    std::vector<std::vector<double>> pose_targets;//target poses:{#(objects),{frame,label,x,y,z,nx,ny,nz}}
    std::vector<std::vector<double>> poses_current;//{#(objects),(frame,label,x,y,z)}
    double delta_x, delta_frame;
    double x_target, y_target, z_target, nx_target, ny_target, nz_target, frame_target, dist_rob2target;
    double speed;
    double speed_min, dist_candidate;
    double frame_target_return = -1.0;
    //std::vector<double> ee_current = urDI->getActualTCPPose();//current end-effector pose
    std::vector<double> ee_current = ee_current_;
    //calculate target poses.
    std::vector<double> speeds_rob2targets;//{#(targets),speed}
    std::vector<double>dists_rob2targets;
    std::vector<double> frames_longest;
    std::vector<double> min_dists_target2human;
    std::vector<double> frames_target;
    idx_human_catch.clear();//initialize
    min_dists_human_objects.clear();//initialize.distance between wrist and target.
    //robot
    idx_robot_catch.clear();
    params_candidates.clear();

    double delta_time, r_catch;
    /**************/

    //calculate parameters for target position.
    if (bool_dynamic_targetAdjustment) {//dynamically adjust target paramters.
        if (!infoTarget_.p_target.empty()) {//not first
            double dframe_target = infoTarget_.delta_frame;
            dframe_target = std::max(threshold_dframe_min_, std::min(threshold_dframe_max_, dframe_target));//clip dframe
            //within the same object's candidates.
            const double grad_dframe_ = (lambda_max_dframe_ - lambda_min_dframe_) / (threshold_dframe_max_ - threshold_dframe_min_);//gradient for dframe
            const double grad_dist_ = -1.0 * (lambda_max_dist_ - lambda_min_dist_) / (threshold_dframe_max_ - threshold_dframe_min_);//gradient for distance
            const double grad_human_ = (lambda_max_human_ - lambda_min_human_) / (threshold_dframe_max_ - threshold_dframe_min_);//gradient for distance
            lambda_dist = grad_dist_ * (dframe_target - threshold_dframe_min_) + lambda_max_dist_;
            lambda_dframe = grad_dframe_ * (dframe_target - threshold_dframe_min_) + lambda_min_dframe_;
            lambda_human = grad_human_ * (dframe_target - threshold_dframe_min_) + lambda_min_human_;
            //intra-object selection
            const double grad_dframe_intra_ = (lambda_max_dframe_intra_ - lambda_min_dframe_intra_) / (threshold_dframe_max_ - threshold_dframe_min_);//gradient for dframe
            const double grad_dist_intra_ = -1.0 * (lambda_max_dist_intra_ - lambda_min_dist_intra_) / (threshold_dframe_max_ - threshold_dframe_min_);//gradient for distance
            lambda_dist_intra = grad_dist_intra_ * (dframe_target - threshold_dframe_min_) + lambda_max_dist_intra_;
            lambda_dframe_intra = grad_dframe_intra_ * (dframe_target - threshold_dframe_min_) + lambda_min_dframe_intra_;
        }
        else {//first -> time until catching will be prioritized.
            lambda_dist = lambda_min_dist_;
            lambda_dframe = lambda_max_dframe_;
            lambda_human = lambda_max_human_;
            lambda_dist_intra = lambda_min_dist_intra_;
            lambda_dframe_intra = lambda_max_dframe_intra_;
        }
    }
    else {
        lambda_dist = lambda_min_dist_;
        lambda_dframe = lambda_max_dframe_;
        lambda_human = lambda_max_human_;
        lambda_dist_intra = lambda_min_dist_intra_;
        lambda_dframe_intra = lambda_max_dframe_intra_;
    }
    /*****************/

    /**Determine target position within the same object**/
    bool bool_human_catch = false;
    std::vector<double> p_w;
    for (int n = 0; n < params_trajectory.size(); n++) {//for each object
        if (params_trajectory[n].param_z(0) < -0.20) {//comply with physics law.
            // whether human will catch
            bool_human_catch = false;

            //check human catching candidates.
            if (!wrists_human.empty()) {
                double dist_wrist2target;
                for (int h = 0; h < wrists_human.size(); h++) {//for each wrist
                    std::vector<double> target_human;//{frameTarget,label,x,y,z,nx,ny,nz}
                    p_w = wrists_human[h];//(frame,x,y,z)
                    prediction_.calculate_target(p_w[1], params_trajectory[n], target_human);//human's wrist x value is the target depth.
                    if (!target_human.empty()) {
                        if (target_human[0] > frame_current) {//catching candidate
                            dist_wrist2target = std::sqrt((p_w[1] - target_human[2]) * (p_w[1] - target_human[2]) + (p_w[2] - target_human[3]) * (p_w[2] - target_human[3]) + (p_w[3] - target_human[4]) * (p_w[3] - target_human[4]));

                            //determine catching radius.
                            delta_time = (target_human[0] - frame_current) / (double)FPS;//time[sec]
                            if (delta_time >= t_upper_)
                                r_catch = r_catch_candidate_;
                            else if (t_lower_ <= delta_time && delta_time <= t_upper_)
                                r_catch = r_catch_candidate_ * (1.0 / 2.0) + (r_catch_candidate_ * (1.0 / 2.0)) * ((delta_time - t_lower_) / (t_upper_ - t_lower_));
                            else
                                r_catch = r_catch_candidate_ / 2.0;

                            if (dist_wrist2target < r_catch) {//candidate
                                if (std::find(idx_human_catch.begin(), idx_human_catch.end(), n) == idx_human_catch.end()) {//not updated
                                    idx_human_catch.push_back(n);//push object index.
                                    min_dists_human_objects.push_back(dist_wrist2target);
                                }
                                else {//already updated
                                    if (min_dists_human_objects.back() > dist_wrist2target)//update minimum distance.
                                        min_dists_human_objects.back() = dist_wrist2target;
                                }
                                bool_human_catch = true;
                            }
                        }
                    }
                }
            }

            std::vector<double> candidate_target;//{frame,label,x,y,z,nx,ny,nz}
            double delta_frame_longest = 0;
            double x_target;
            double cost_tmp, frame_target_selected;
            double dmin_target2human = 100.0;
            double cost_min = 100;
            for (int j = 0; j < (int)n_candidates; j++) {//for each target steps.
                //std::cout << "1-2-2"<<j << std::endl;

                x_target = x_candidates[j];
                std::vector<double> target;//{frameTarget,label,x,y,znx,ny,nz}
                prediction_.calculate_target(x_target, params_trajectory[n], target);
                if (!target.empty()) {//valid target point is available
                    //target position is for the tip of a cup. calculate the UR end-effector position
                    y_target = target[3];
                    z_target = target[4];
                    nx_target = target[5];
                    ny_target = target[6];
                    nz_target = target[7];
                    //convert to UR end effector position.
                    x_target = x_target - nx_target * h_cup_;
                    y_target = y_target - ny_target * h_cup_;
                    z_target = z_target - nz_target * h_cup_;
                    target[2] = x_target;//transform target for the cup to catch the ball
                    target[3] = y_target;
                    target[4] = z_target;
                    if ((x_min_ <= x_target && x_target <= x_max_) && (y_min_ <= y_target && y_target <= y_max_) && (z_min_ <= z_target && z_target <= z_max_)) {//withing the working space.
                        //check robot speed to the target pose.
                        speed = std::sqrt((ee_current[0] - x_target) * (ee_current[0] - x_target) + (ee_current[1] - y_target) * (ee_current[1] - y_target) + (ee_current[2] - z_target) * (ee_current[2] - z_target));
                        frame_target = target[0];
                        delta_frame = frame_target - frame_current;
                        speed /= delta_frame;
                        if (speed < 0)
                            speed = speed_max_ + 0.01;
                        double dmin = 100.0;

                        if (!wrists_human.empty()) {
                            double d_target2wrist;
                            std::vector<double> p_w;
                            for (int h = 0; h < wrists_human.size(); h++) {//for each joint
                                p_w = wrists_human[h];//(frame,x,y,z)
                                d_target2wrist = std::sqrt((p_w[1] - x_target) * (p_w[1] - x_target) + (p_w[2] - y_target) * (p_w[2] - y_target) + (p_w[3] - z_target) * (p_w[3] - z_target));
                                if (d_target2wrist < dmin)
                                    dmin = d_target2wrist;
                            }
                        }
                        //std::cout << n << "-th object :: " << "speed=" << speed <<", delta_frame="<<delta_frame<< std::endl;
                        if (speed <= speed_max_ && delta_frame > 0.0)
                            cost_tmp = std::max(0.0, std::min(1.0, (speed * delta_frame) / dist_thresh_)) * lambda_dist + std::max(0.0, std::min(1.0, (1.0 - delta_frame / dframe_thresh_))) * lambda_dframe + lambda_human * std::max(0.0, std::min(1.0, (1.0 - dmin / dist_target2human_thresh_)));
                        else
                            cost_tmp = 1000.0;

                        if (cost_tmp < cost_min) {//higher probability target pose.->add to candidate_target
                            candidate_target = target;
                            speed_min = speed;//update minimum speed.
                            dist_candidate = speed * delta_frame;
                            delta_frame_longest = delta_frame;
                            cost_min = cost_tmp;

                            //minimum distance between the target and human wrist.
                            if (dmin < dmin_target2human)
                                dmin_target2human = dmin;
                            frame_target_selected = frame_target;
                        }
                    }
                }
            }

            //add candidate to pose_targets
            if (!candidate_target.empty()) {
                idx_robot_catch.push_back(n);//push object index.
                pose_targets.push_back(candidate_target);
                speeds_rob2targets.push_back(speed_min);
                dists_rob2targets.push_back(dist_candidate);
                frames_longest.push_back(delta_frame_longest);
                min_dists_target2human.push_back(dmin_target2human);
                frames_target.push_back(frame_target_selected);
                poses_current.push_back(params_trajectory[n].pos_current);

                //save parameters for move robot backward.
                param_candidate.param_x = params_trajectory[n].param_x;
                param_candidate.param_y = params_trajectory[n].param_y;
                param_candidate.param_z = params_trajectory[n].param_z;
                params_candidates.push_back(param_candidate);
            }
        }
    }
    /*********************/

    //std::cout << "pose_targets.size()=" << pose_targets.size() << ", idx_human_catch.size()=" << idx_human_catch.size() << ", idx_robot_catch.size()=" << idx_robot_catch.size() << std::endl;
    //if (!idx_human_catch.empty()) {
    //    std::cout << "idx_human_catch=";
    //    for (int k = 0; k < idx_human_catch.size(); k++)
    //        std::cout << idx_human_catch[k] << ",";
    //    std::cout << std::endl;
    //}
    //if (!idx_robot_catch.empty()) {
    //    std::cout << "idx_robot_catch=";
    //    for (int k = 0; k < idx_robot_catch.size(); k++)
    //        std::cout << idx_robot_catch[k] << ",";
    //    std::cout << std::endl;
    //}

    /***Select Targets among multiple objects***/
    if (!pose_targets.empty()) {
        int idx_target = -1;

        //pose_targets:{#(objects),{frame,label,x,y,z,nx,ny,nz}}
        std::vector<double> target_opt;
        if (pose_targets.size() >= 2) {//choose based on the speed and distance

            if (idx_human_catch.size() < idx_robot_catch.size()) {//robot catching candidate's number is larger than human's catching candidate -> choose the one which is out of human wrist region.
                //check whether there are duplicated ones.
                if (!idx_human_catch.empty()) {
                    int j = 0;
                    while (j < idx_robot_catch.size()) {
                        if (std::find(idx_human_catch.begin(), idx_human_catch.end(), idx_robot_catch[j]) != idx_human_catch.end()) {//find index in the human candidate. -> delte data.
                            idx_robot_catch.erase(idx_robot_catch.begin() + j);
                            pose_targets.erase(pose_targets.begin() + j);
                            speeds_rob2targets.erase(speeds_rob2targets.begin() + j);
                            dists_rob2targets.erase(dists_rob2targets.begin() + j);
                            frames_longest.erase(frames_longest.begin() + j);
                            min_dists_target2human.erase(min_dists_target2human.begin() + j);
                            frames_target.erase(frames_target.begin() + j);
                            poses_current.erase(poses_current.begin() + j);
                            params_candidates.erase(params_candidates.begin() + j);
                        }
                        else
                            j++;
                    }
                }

                double cost_tmp;
                double cost_min = 100.0;
                idx_target = -1;
                for (int k = 0; k < pose_targets.size(); k++) {//for each target
                    if ((frames_target[k] - frame_current) > 0) {
                        //cost_tmp = std::max(0.0, std::min(1.0, dists_rob2targets[j] / dist_thresh_)) * lambda_dist_ + std::max(0.0, std::min(1.0, (frames_target[j] - frame_current) / dframe_thresh_)) * lambda_dframe_;
                        cost_tmp = std::max(0.0, std::min(1.0, dists_rob2targets[k] / dist_thresh_)) * lambda_dist_intra + std::max(0.0, std::min(1.0, (1.0 - (frames_target[k] - frame_current) / dframe_thresh_))) * lambda_dframe_intra;
                        //cost_tmp = std::max(0.0, std::min(1.0, dists_rob2targets[j] / dist_thresh_)) + std::max(0.0, std::min(1.0, (frames_target[j] - frame_current) / dframe_thresh_));
                        if (cost_tmp < cost_min && frames_target[k] < (frame_current + dframe_thresh_)) {
                            cost_min = cost_tmp;
                            idx_target = k;
                        }
                    }
                }

                if (idx_target >= 0) {
                    //update target position
                    target_opt = pose_targets[idx_target];
                    //update delta_frame
                    frame_target_robot = frames_target[idx_target];

                    pose_target = std::vector<double>{ target_opt[2],target_opt[3],target_opt[4],target_opt[5],target_opt[6],target_opt[7] };//{0:frameTarget,1:label,2:x,3:y,4:z,5:nx,6:ny,7:nz}
                    bool_back = true;

                    param_target = params_candidates[idx_target];//param_x,param_y,param_z
                    frame_target_current = frames_target[idx_target];//frame

                    if (bool_use_actual_data) {
                        if ((frames_target[idx_target] - poses_current[idx_target][0]) > 0 && (frames_target[idx_target] - poses_current[idx_target][0]) <= frame_use_tracking_) {//use actual tracking data
                            pose_target[0] = poses_current[idx_target][2] - pose_target[3] * h_cup_;//x. {0:frame,1:label,2:x,3:y,4:z} 
                            pose_target[1] = poses_current[idx_target][3] - pose_target[4] * h_cup_;//y
                            pose_target[2] = poses_current[idx_target][4] - pose_target[5] * h_cup_;//z
                            //convert from the tip of a cup to UR end effector position.
                        }
                    }

                    infoTarget_.p_target = pose_target;
                    infoTarget_.delta_frame = frames_target[idx_target] - frame_current;

                    return 1.0;
                }
                else {
                    bool_back = false;
                    infoTarget_.p_target.clear();
                    return -1.0;
                }
            }
            else if (idx_human_catch.size() == idx_robot_catch.size()) {//same number
                //check whether there are duplicated ones.
                int counter = 0;
                std::vector<int> indexes_duplicated;
                for (int j = 0; j < idx_robot_catch.size(); j++) {
                    if (std::find(idx_human_catch.begin(), idx_human_catch.end(), idx_robot_catch[j]) != idx_human_catch.end()) {//find index in the human candidate. -> delte data.
                        counter++;
                        indexes_duplicated.push_back(j);//save duplicated index of robot's candidates.
                    }
                }

                if (counter == idx_robot_catch.size()) {//human's candidate and robot candidate's number is equal -> pose target is the middle points.
                    double x_cand = 0.0;
                    double y_cand = 0.0;
                    double z_cand = 0.0;
                    double nx_cand = 0.0;
                    double ny_cand = 0.0;
                    double nz_cand = 0.0;
                    double counter = 0.0;
                    double frame_target_cand = 1000.0;
                    double weight = 0.0;
                    double max_dist = 0.0;
                    for (int j = 0; j < idx_robot_catch.size(); j++) {//passive catching.for each target.
                        x_cand += poses_target[j][2] * min_dists_human_objects[j];
                        y_cand += poses_target[j][3] * min_dists_human_objects[j];
                        z_cand += poses_target[j][4] * min_dists_human_objects[j];
                        if (max_dist < min_dists_human_objects[j]) {
                            nx_cand = poses_target[j][5];
                            ny_cand = poses_target[j][6];
                            nz_cand = poses_target[j][7];
                            max_dist = min_dists_human_objects[j];
                        }
                        weight += min_dists_human_objects[j];

                        if (frames_target[j] < frame_target_cand)
                            frame_target_cand = frames_target[j];
                    }
                    x_cand /= weight;
                    y_cand /= weight;
                    z_cand /= weight;
                    pose_target = std::vector<double>{ x_cand,y_cand,z_cand,nx_cand,ny_cand,nz_cand };//{0:frameTarget,1:label,2:x,3:y,4:z,5:nx,6:ny,7:nz}
                    bool_back = false;

                    infoTarget_.p_target = pose_target;
                    infoTarget_.delta_frame = frame_target_cand - frame_current;

                    return 1.0;

                }
                else {//another object is chosen.
                    //check whether there are duplicated ones.
                    if (!indexes_duplicated.empty()) {
                        std::sort(indexes_duplicated.rbegin(), indexes_duplicated.rend());//sort indexes_duplicated in a descending order to maintain the order.
                        for (int& j : indexes_duplicated) {//erase from the large to the small
                            idx_robot_catch.erase(idx_robot_catch.begin() + j);
                            pose_targets.erase(pose_targets.begin() + j);
                            speeds_rob2targets.erase(speeds_rob2targets.begin() + j);
                            dists_rob2targets.erase(dists_rob2targets.begin() + j);
                            frames_longest.erase(frames_longest.begin() + j);
                            min_dists_target2human.erase(min_dists_target2human.begin() + j);
                            frames_target.erase(frames_target.begin() + j);
                            poses_current.erase(poses_current.begin() + j);
                            params_candidates.erase(params_candidates.begin() + j);
                        }
                    }

                    if (idx_robot_catch.size() > 0) {//choose the furthest one.
                        double cost_tmp;
                        double cost_min = 100.0;
                        idx_target = -1;
                        for (int k = 0; k < pose_targets.size(); k++) {//for each target
                            if ((frames_target[k] - frame_current) > 0)
                            {
                                //cost_tmp = std::max(0.0, std::min(1.0, dists_rob2targets[j] / dist_thresh_)) * lambda_dist_ + std::max(0.0, std::min(1.0, (frames_target[j] - frame_current) / dframe_thresh_)) * lambda_dframe_;
                                cost_tmp = std::max(0.0, std::min(1.0, dists_rob2targets[k] / dist_thresh_)) * lambda_dist_intra + std::max(0.0, std::min(1.0, (1.0 - (frames_target[k] - frame_current) / dframe_thresh_))) * lambda_dframe_intra;
                                //cost_tmp = std::max(0.0, std::min(1.0, dists_rob2targets[j] / dist_thresh_)) + std::max(0.0, std::min(1.0, (frames_target[j] - frame_current) / dframe_thresh_));
                                if (cost_tmp < cost_min && frames_target[k] < (frame_current + dframe_thresh_)) {
                                    cost_min = cost_tmp;
                                    idx_target = k;
                                }
                            }
                        }

                        if (idx_target >= 0) {
                            target_opt = pose_targets[idx_target];
                            //update delta_frame
                            frame_target_robot = frames_target[idx_target];

                            //retain target parameter
                            bool_back = true;//should move 
                            param_target = params_candidates[idx_target];//param_x,param_y,param_z
                            frame_target_current = frames_target[idx_target];//frame

                            pose_target = std::vector<double>{ target_opt[2],target_opt[3],target_opt[4],target_opt[5],target_opt[6],target_opt[7] };//{0:frameTarget,1:label,2:x,3:y,4:z,5:nx,6:ny,7:nz}

                            if (bool_use_actual_data) {
                                if ((frames_target[idx_target] - poses_current[idx_target][0]) > 0 && (frames_target[idx_target] - poses_current[idx_target][0]) <= frame_use_tracking_) {//use actual tracking data
                                    pose_target[0] = poses_current[idx_target][2] - pose_target[3] * h_cup_;//x. {0:frame,1:label,2:x,3:y,4:z} 
                                    pose_target[1] = poses_current[idx_target][3] - pose_target[4] * h_cup_;//y
                                    pose_target[2] = poses_current[idx_target][4] - pose_target[5] * h_cup_;//z
                                    //convert from the tip of a cup to UR end effector position.
                                }
                            }

                            infoTarget_.p_target = pose_target;
                            infoTarget_.delta_frame = frames_target[idx_target] - frame_current;

                            return 1.0;
                        }
                        else {
                            bool_back = false;
                            infoTarget_.p_target.clear();
                            return -1.0;
                        }
                    }
                    else {
                        std::cout << ":: WARNING :: Seems to be awkward cases. Check the sequence.cpp, line. 1312" << std::endl;
                        bool_back = false;
                        infoTarget_.p_target.clear();
                        return -1.0;
                    }
                }
            }
            else {//robot catch is smaller number
                //check whether there are duplicated ones.
                int counter = 0;
                std::vector<int> indexes_duplicated;
                for (int j = 0; j < idx_robot_catch.size(); j++) {
                    if (std::find(idx_human_catch.begin(), idx_human_catch.end(), idx_robot_catch[j]) != idx_human_catch.end()) {//find index in the human candidate. -> delte data.
                        counter++;
                        indexes_duplicated.push_back(j);//save duplicated index of robot's candidates.
                    }
                }

                if (counter == idx_robot_catch.size()) {//human's candidate and robot candidate's number is equal -> pose target is the middle points.
                    //std::cout << "robot-11-1" << std::endl;
                    double x_cand = 0.0;
                    double y_cand = 0.0;
                    double z_cand = 0.0;
                    double nx_cand = 0.0;
                    double ny_cand = 0.0;
                    double nz_cand = 0.0;
                    double counter = 0.0;
                    double weight = 0.0;
                    double frame_target_cand = 1000.0;
                    double max_dist = 0.0;
                    for (int j = 0; j < idx_robot_catch.size(); j++) {//passive catching.for each target.
                        x_cand += poses_target[j][2] * min_dists_human_objects[j];
                        y_cand += poses_target[j][3] * min_dists_human_objects[j];
                        z_cand += poses_target[j][4] * min_dists_human_objects[j];
                        if (max_dist < min_dists_human_objects[j]) {
                            nx_cand = poses_target[j][5];
                            ny_cand = poses_target[j][6];
                            nz_cand = poses_target[j][7];
                            max_dist = min_dists_human_objects[j];
                        }
                        weight += min_dists_human_objects[j];
                        if (frames_target[j] < frame_target_cand)
                            frame_target_cand = frames_target[j];
                    }
                    x_cand /= weight;
                    y_cand /= weight;
                    z_cand /= weight;
                    pose_target = std::vector<double>{ x_cand,y_cand,z_cand,nx_cand,ny_cand,nz_cand };//{0:frameTarget,1:label,2:x,3:y,4:z,5:nx,6:ny,7:nz}
                    bool_back = false;

                    infoTarget_.p_target = pose_target;
                    infoTarget_.delta_frame = frame_target_cand - frame_current;

                    return 1.0;
                }
                else {//another objects are chosen.
                    if (!indexes_duplicated.empty()) {//delete duplicated candidates.
                        std::sort(indexes_duplicated.rbegin(), indexes_duplicated.rend());//sort indexes_duplicated in a descending order to maintain the order.
                        for (int& j : indexes_duplicated) {//erase from the large to the small
                            idx_robot_catch.erase(idx_robot_catch.begin() + j);
                            pose_targets.erase(pose_targets.begin() + j);
                            speeds_rob2targets.erase(speeds_rob2targets.begin() + j);
                            dists_rob2targets.erase(dists_rob2targets.begin() + j);
                            frames_longest.erase(frames_longest.begin() + j);
                            min_dists_target2human.erase(min_dists_target2human.begin() + j);
                            frames_target.erase(frames_target.begin() + j);
                            poses_current.erase(poses_current.begin() + j);
                            params_candidates.erase(params_candidates.begin() + j);
                        }
                    }

                    double cost_tmp;
                    double cost_min = 100.0;
                    idx_target = -1;
                    for (int k = 0; k < pose_targets.size(); k++) {//for each target
                        if ((frames_target[k] - frame_current) > 0) {
                            //cost_tmp = std::max(0.0, std::min(1.0, dists_rob2targets[j] / dist_thresh_)) * lambda_dist_ + std::max(0.0, std::min(1.0, (frames_target[j] - frame_current) / dframe_thresh_)) * lambda_dframe_;
                            cost_tmp = std::max(0.0, std::min(1.0, dists_rob2targets[k] / dist_thresh_)) * lambda_dist_intra + std::max(0.0, std::min(1.0, (1.0 - (frames_target[k] - frame_current) / dframe_thresh_))) * lambda_dframe_intra;
                            //cost_tmp = std::max(0.0, std::min(1.0, dists_rob2targets[j] / dist_thresh_)) + std::max(0.0, std::min(1.0, (frames_target[j] - frame_current) / dframe_thresh_));
                            if (cost_tmp < cost_min && frames_target[k] < (frame_current + dframe_thresh_)) {
                                cost_min = cost_tmp;
                                idx_target = k;
                            }
                        }
                    }

                    if (idx_target >= 0) {
                        target_opt = pose_targets[idx_target];
                        //update delta_frame
                        frame_target_robot = frames_target[idx_target];

                        //retain target parameter
                        bool_back = true;//should move 
                        param_target = params_candidates[idx_target];//param_x,param_y,param_z
                        frame_target_current = frames_target[idx_target];//frame

                        pose_target = std::vector<double>{ target_opt[2],target_opt[3],target_opt[4],target_opt[5],target_opt[6],target_opt[7] };//{0:frameTarget,1:label,2:x,3:y,4:z,5:nx,6:ny,7:nz}
                        if (bool_use_actual_data) {
                            if ((frames_target[idx_target] - poses_current[idx_target][0]) > 0 && (frames_target[idx_target] - poses_current[idx_target][0]) <= frame_use_tracking_) {//use actual tracking data
                                pose_target[0] = poses_current[idx_target][2] - pose_target[3] * h_cup_;//x. {0:frame,1:label,2:x,3:y,4:z} 
                                pose_target[1] = poses_current[idx_target][3] - pose_target[4] * h_cup_;//y
                                pose_target[2] = poses_current[idx_target][4] - pose_target[5] * h_cup_;//z
                                //convert from the tip of a cup to UR end effector position.
                            }
                        }

                        infoTarget_.p_target = pose_target;
                        infoTarget_.delta_frame = frames_target[idx_target] - frame_current;

                        return 1.0;
                    }
                    else {
                        bool_back = false;
                        infoTarget_.p_target.clear();
                        return -1.0;
                    }
                }

            }
        }
        else {//pose target is one.
            if (!idx_human_catch.empty()) {//human's candidate is included.
                int counter = 0;
                for (int j = 0; j < idx_robot_catch.size(); j++) {
                    if (std::find(idx_human_catch.begin(), idx_human_catch.end(), idx_robot_catch[j]) != idx_human_catch.end()) {//find index in the human candidate. -> delte data.
                        counter++;
                    }
                }

                if (counter == idx_robot_catch.size()) {//human will catch the ball.        
                    bool_back = false;
                    infoTarget_.p_target.clear();
                    return -1.0;

                }
                else {//human's candidate is different from the robot's one.
                    double cost_tmp;
                    double cost_min = 100.0;
                    idx_target = -1;
                    for (int k = 0; k < pose_targets.size(); k++) {//for each target
                        if ((frames_target[k] - frame_current) > 0) {
                            //cost_tmp = std::max(0.0, std::min(1.0, dists_rob2targets[j] / dist_thresh_)) * lambda_dist_ + std::max(0.0, std::min(1.0, (frames_target[j] - frame_current) / dframe_thresh_)) * lambda_dframe_;
                            cost_tmp = std::max(0.0, std::min(1.0, dists_rob2targets[k] / dist_thresh_)) * lambda_dist_intra + std::max(0.0, std::min(1.0, (1.0 - (frames_target[k] - frame_current) / dframe_thresh_))) * lambda_dframe_intra;
                            //cost_tmp = std::max(0.0, std::min(1.0, dists_rob2targets[j] / dist_thresh_)) + std::max(0.0, std::min(1.0, (frames_target[j] - frame_current) / dframe_thresh_));
                            if (cost_tmp < cost_min && frames_target[k] < (frame_current + dframe_thresh_)) {
                                cost_min = cost_tmp;
                                idx_target = k;
                            }
                        }
                    }

                    if (idx_target >= 0) {
                        target_opt = pose_targets[idx_target];
                        //update delta_frame
                        frame_target_robot = frames_target[idx_target];

                        //retain target parameter
                        bool_back = true;//should move 
                        param_target = params_candidates[idx_target];//param_x,param_y,param_z
                        frame_target_current = frames_target[idx_target];//frame

                        pose_target = std::vector<double>{ target_opt[2],target_opt[3],target_opt[4],target_opt[5],target_opt[6],target_opt[7] };//{0:frameTarget,1:label,2:x,3:y,4:z,5:nx,6:ny,7:nz}
                        if (bool_use_actual_data) {
                            if ((frames_target[idx_target] - poses_current[idx_target][0]) > 0 && (frames_target[idx_target] - poses_current[idx_target][0]) <= frame_use_tracking_) {//use actual tracking data
                                pose_target[0] = poses_current[idx_target][2] - pose_target[3] * h_cup_;//x. {0:frame,1:label,2:x,3:y,4:z} 
                                pose_target[1] = poses_current[idx_target][3] - pose_target[4] * h_cup_;//y
                                pose_target[2] = poses_current[idx_target][4] - pose_target[5] * h_cup_;//z
                                //convert from the tip of a cup to UR end effector position.
                            }
                        }

                        infoTarget_.p_target = pose_target;
                        infoTarget_.delta_frame = frames_target[idx_target] - frame_current;

                        return 1.0;
                    }
                    else {
                        bool_back = false;
                        infoTarget_.p_target.clear();
                        return -1.0;
                    }
                }
            }
            else {//human's target doesn't exist.-> robot will catch the one.
                double cost_tmp;
                double cost_min = 100.0;
                idx_target = -1;
                for (int k = 0; k < pose_targets.size(); k++) {//for each target
                    if ((frames_target[k] - frame_current) > 0) {
                        //cost_tmp = std::max(0.0, std::min(1.0, dists_rob2targets[j] / dist_thresh_)) * lambda_dist_ + std::max(0.0, std::min(1.0, (frames_target[j] - frame_current) / dframe_thresh_)) * lambda_dframe_;
                        cost_tmp = std::max(0.0, std::min(1.0, dists_rob2targets[k] / dist_thresh_)) * lambda_dist_intra + std::max(0.0, std::min(1.0, (1.0 - (frames_target[k] - frame_current) / dframe_thresh_))) * lambda_dframe_intra;
                        //cost_tmp = std::max(0.0, std::min(1.0, dists_rob2targets[j] / dist_thresh_)) + std::max(0.0, std::min(1.0, (frames_target[j] - frame_current) / dframe_thresh_));
                        if (cost_tmp < cost_min && frames_target[k] < (frame_current + dframe_thresh_)) {
                            cost_min = cost_tmp;
                            idx_target = k;
                        }
                    }
                }

                if (idx_target >= 0) {
                    target_opt = pose_targets[idx_target];
                    //update delta_frame
                    frame_target_robot = frames_target[idx_target];

                    //retain target parameter
                    bool_back = true;//should move 
                    param_target = params_candidates[idx_target];//param_x,param_y,param_z
                    frame_target_current = frames_target[idx_target];//frame

                    pose_target = std::vector<double>{ target_opt[2],target_opt[3],target_opt[4],target_opt[5],target_opt[6],target_opt[7] };//{0:frameTarget,1:label,2:x,3:y,4:z,5:nx,6:ny,7:nz}
                    if (bool_use_actual_data) {
                        if ((frames_target[idx_target] - poses_current[idx_target][0]) > 0 && (frames_target[idx_target] - poses_current[idx_target][0]) <= frame_use_tracking_) {//use actual tracking data
                            pose_target[0] = poses_current[idx_target][2] - pose_target[3] * h_cup_;//x. {0:frame,1:label,2:x,3:y,4:z} 
                            pose_target[1] = poses_current[idx_target][3] - pose_target[4] * h_cup_;//y
                            pose_target[2] = poses_current[idx_target][4] - pose_target[5] * h_cup_;//z
                            //convert from the tip of a cup to UR end effector position.
                        }
                    }

                    infoTarget_.p_target = pose_target;
                    infoTarget_.delta_frame = frames_target[idx_target] - frame_current;

                    return 1.0;
                }
                else {
                    bool_back = false;
                    infoTarget_.p_target.clear();
                    return -1.0;
                }
            }
        }
    }
    else {//no target
        bool_back = false;
        infoTarget_.p_target.clear();
        return -1.0;
    }
}

double Sequence::calculateIoU_Rect2d(const cv::Rect2d& box1, const cv::Rect2d& box2)
{
    double left = std::max(box1.x, box2.x);
    double top = std::max(box1.y, box2.y);
    double right = std::min((box1.x + box1.width), (box2.x + box2.width));
    double bottom = std::min((box1.y + box1.height), (box2.y + box2.height));

    if (left < right && top < bottom)
    {
        double intersection = (right - left) * (bottom - top);
        double area1 = box1.width * box1.height;
        double area2 = box2.width * box2.height;
        double unionArea = area1 + area2 - intersection;

        return intersection / unionArea;
    }

    return 0.0; // No overlap
}

double Sequence::calculateRMSE_Rect2d(const cv::Rect2d& box1, const cv::Rect2d& box2)
{
    double centerX_1 = box1.x + box1.width / 2;
    double centerY_1 = box1.y + box1.height / 2;
    double centerX_2 = box2.x + box2.width / 2;
    double centerY_2 = box2.y + box2.height / 2;
    double dx = std::pow(centerX_1 - centerX_2, 2);
    double dy = std::pow(centerY_1 - centerY_2, 2);
    double rmse = std::sqrt(dx + dy);
    //std::cout << "BOX1 :: left=" << box1.x << ", top=" << box1.y << ", w=" << box1.width << ", h=" << box1.height << std::endl;
    //std::cout << "BOX2 :: left=" << box2.x << ", top=" << box2.y << ", w=" << box2.width << ", h=" << box2.height << std::endl;
    //std::cout << "rmse=" << rmse << std::endl;
    if (rmse >= Rmse_identity || std::isnan(rmse))
        rmse = Cost_max;
    //rmse = std::max(1.0, rmse / Rmse_identity);
    //if (rmse == 1.0)
    //    rmse = Cost_max;
    return rmse;
}

double Sequence::compareID(int label1, int label2) {
    /**
    * @brief compare 2 labels and return cost
    * @param[in] label1, label2 labels (detection label)
    */

    if (label1 == label2)
        return 0.0;
    else
        return Cost_max;
}

double Sequence::sizeDiff(cv::Rect2d& roi1, cv::Rect2d& roi2) {
    double h_left = roi1.height;
    double w_left = roi1.width;
    double h_right = roi2.height;
    double w_right = roi2.width;
    double gamma_left = std::max(1.0, h_left) / std::max(1.0, w_left);
    double gamma_right = std::max(1.0, h_right) / std::max(1.0, w_right);
    double gamma_rate = gamma_left / gamma_right;
    if (gamma_rate < 1.0)
        gamma_rate = 1.0 / gamma_rate;


    //diffenrence in size
    double delta_size;
    if (h_right * w_right > 0) {
        delta_size = (h_left * w_left) / (h_right * w_right);
        if (0.0 < delta_size && delta_size < 1.0)//delta_size>=1,0
            delta_size = 1.0 / delta_size;
        if (delta_size > 0.0) {
            delta_size = delta_size * gamma_rate;
        }
        else
            delta_size = Cost_max;
    }
    else
        delta_size = Cost_max;

    //std::cout << "delta_size=" << delta_size <<", gamma_rate="<<gamma_rate << std::endl;
    if (std::isnan(delta_size))
        delta_size = Cost_max;
    return delta_size;
}

double Sequence::compareParams(std::vector<double>& data1, std::vector<double>& data2) {
    double cost = 0.0;
    if (method_prediction == 0) {//ordinary least squares method
        for (int i = 2; i < data1.size(); i++) {//{0:frame,1:label,2:ax,3:bx,4,cx,5:ay,6:by,7:cy,8:az,9:bz,10:cz}
            if (i == 4 || i == 7 || i == 10)//constant
                continue;
            else {
                cost += std::abs(data1[i] - data2[i]);
            }
        }
    }
    else if (method_prediction == 1) {//recursive least squares method
        for (int i = 2; i < data1.size(); i++) {
            //{0:frame,1:label,(2~2+dim_poly_x-1:theta_x),(2+dim_poly_x~2+dim_poly_y-1):theta_y),
            // (2+dim_poly_x+dim_poly_y~2+dim_poly_x+dim_poly_y+dim_poly_z-1):theta_z)}//from higher order to the lower order.

            //exclude constant variables.
            if (i == (2 + dim_poly_x - 1) || i == (2 + dim_poly_x + dim_poly_y - 1) || i == (2 + dim_poly_x + dim_poly_y + dim_poly_z - 1))//constant
                continue;
            else {
                cost += std::abs(data1[i] - data2[i]);
            }
        }
    }
    return cost;
}


int Sequence::findIndex(const std::vector<int>& vec, int value)
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

void Sequence::concatenateVectors(std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b) {
    a.insert(a.end(), b.begin(), b.end());
}