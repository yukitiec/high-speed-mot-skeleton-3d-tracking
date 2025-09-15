#include "../../include/utils/associate.h"

/*
*@param[in] Trackers2MOT {
	std::vector<int> success_flags;
	Trackers trackers;
}
* @param[in] TrackersYOLO {
    int frameIndex;
	cv::Mat1b frame;
    std::vector<int> classIndex;
    std::vector<cv::Rect2d> bbox;
};
* @param[in] TrackersMOT {
    int frameIndex;
    std::vector<int> classIndex;
    std::vector<cv::Rect2d> bbox;
	std::vector<KalmanFilter> kalmanFilter;
};
* @param[in] Trackers_sequence{
	std::vector<KalmanFilter> kalmanFilter;
	std::vector<TrackersMOT> trackersMOT;
}
*/
void Associate::organize(
	Trackers2MOT& trackers2mot,
	TrackersYOLO& trackersYOLO,
	TrackersMOT& trackersMOT
)
{
	//Step1 :: Merge trackers2mot to trackers_sequence, and create trackersMOT.
	mergeTracking_MOT(trackers2mot, trackersMOT);

	//Step2 :: Update TrackersMOT by comparing trackersYOLO with trackersMOT using success_flags if exists.
	if (!trackersYOLO.bbox.empty()) {//YOLO data is avaiable.
		mergeTracking_YOLO(trackersYOLO, trackersMOT);
	}

	//Step3 ::Add trackersMOT to trackers_sequence.

	//Step4 :: Send trackersMOT to tracking module.

    //organize data
    std::vector<cv::Rect2d> newRoi = newData.bbox;
    std::vector<int> newLabel = newData.classIndex;
    double frame = newData.frame;

    //left data
    if (!newRoi.empty()) {
        matching(newRoi, newLabel, frame, seqData, kfData, kalmanVector, extrapolation); //matching data

        //if seqData[i].back()[0]<0 -> lost data -> move to saveData and delete from seqData
        int num_objects = seqData.size();
        if (num_objects >= 1) {
            for (int i = 0; i < num_objects; i++) {
                if (seqData[i].back()[0] < 0) {//lost data -> save index first.
                    index_delete.push_back(i);
                }
            }
            if (!index_delete.empty()) {//delete data.
                std::sort(index_delete.rbegin(), index_delete.rend());//sort index_delete in descending way 

                for (int& index : index_delete) {//large to small. maintain the index identity.
                    int n_seq = seqData[index].size();
                    if (n_seq > 1) {
                        //delete the last -1 data.
                        seqData[index].erase(seqData[index].begin() + n_seq - 1);
                        //save data
                        saveData.push_back(seqData[index]);
                        saveKFData.push_back(kfData[index]);
                    }
                    //delete data
                    seqData.erase(seqData.begin() + index);
                    kfData.erase(kfData.begin() + index);
                    if (idx_compensation == 0)//kalman filter
                        kalmanVector.erase(kalmanVector.begin() + index);
                    else if (idx_compensation == 1)//linear extrapolation
                        extrapolation.erase(extrapolation.begin() + index);
                }
            }
        }
        //push latest data for match objects between 2 cameras.
        //q_seq2tri.push(seqData);
    }
}

/*
* @brief merge trackers2mot to trackersMOT.
*@param[in] Trackers2MOT {
	int frameIndex;
	std::vector<int> success_flags;
	Trackers trackers;
}

* @param[out] TrackersMOT {//for the latest data.
    int frameIndex;
    std::vector<int> classIndex;
    std::vector<cv::Rect2d> bbox;//for each object.
	std::vector<KalmanFilter> kalmanFilter;
};
*/
void Associate::mergeTracking_MOT(
	Trackers2MOT& trackers2mot,
	TrackersMOT& trackersMOT
)
{
	if (!trackers2mot.success_flags.empty()) {
		//struct Trackers {
		//	std::vector<int> classIndex;
		//	std::vector<int> index_highspeed; //index for high speed tracking
		//	std::vector<TrackerInfo> trackerInfo;
		//	cv::Mat1b previousImg; 
		//};
		//struct TrackerInfo{
		//	cv::Ptr<cv::mytracker::TrackerMOSSE> mosse;
		//	cv::Ptr<TemplateMatching> template_matching;//have a template image internally.
		//	cv::Rect2d bbox;
		//	double scale_searcharea; //scale ratio of search area against bbox
		//	cv::Point2d vel; //previous velocity
		//	int n_notMove; //number of not move
		//};
		int frameIndex_tracking = trackers2mot.frameIndex;//extract the frame index of the tracking result.
		for (int i = 0; i < trackers2mot.success_flags.size(); i++) {//for each high-speed tracking result.
			if (trackers2mot.success_flags[i]) {//succeeded.
				Trackers trackers = trackers2mot.trackers;
				int index_object = trackers.index_highspeed[i];//get the index of the object.
				TrackerInfo tracker = trackers.trackerInfo[index_object];//extract the latest tracker information..
                //extract the current bbox.
				cv::Rect2d bbox = tracker.bbox;
				cv::Point2d vel = tracker.vel;
				//update the bbox.
				trackersMOT.bbox[index_object] = bbox;
				//update Kalman filter with the latest bbox and time.
				//prediction step.
				bool success = trackersMOT.kalmanFilter[index_object].predict(frameIndex_tracking); //predict kalman filter
				//update step.
				Eigen::VectorXd measurement = Eigen::VectorXd::Zero(4);
				measurement(0) = bbox.x + bbox.width / 2.0;
				measurement(1) = bbox.y + bbox.height / 2.0;
				measurement(2) = vel.x;
				measurement(3) = vel.y;
				trackersMOT.kalmanFilter[index_object].update(measurement);
			}
			else{//tracking is failed. -> if Kalman filter is working, 
				Trackers trackers = trackers2mot.trackers;
				int index_object = trackers.index_highspeed[i];//get the index of the object.
				TrackerInfo tracker = trackers.trackerInfo[index_object];//extract the latest tracker information..
				//predict Kalman filter.
				if (trackersMOT.kalmanFilter[index_object].counter_update >= GP::COUNTER_VALID) {//valid trackers.
					Eigen::VectorXd prediction = trackersMOT.kalmanFilter[index_object].predict_only(frameIndex_tracking); //predict kalman filter
					//update the bbox's left and top with prediction.
					cv::Rect2d bbox = tracker.bbox;
					bbox.x = prediction(0) - bbox.width / 2.0;
					bbox.y = prediction(1) - bbox.height / 2.0;
					//update the bbox.
					trackers2mot.trackers.trackerInfo[index_object].bbox = bbox;
					//update the velocity.
					trackers2mot.trackers.trackerInfo[index_object].vel = cv::Point2d(prediction(2), prediction(3));				
				}
			}
		}
		trackersMOT.frameIndex = frameIndex_tracking;//update the frame index.
	}
	else{//high-speed tracking is not working.
		//skinp merging.
	}
}

/* @param[in] TrackersYOLO {
    int frameIndex;
	cv::Mat1b frame;
    std::vector<int> classIndex;
    std::vector<cv::Rect2d> bbox;
    std::vector<double> scores;
};
* @param[out] TrackersMOT {//for the latest data.
    int frameIndex;
    std::vector<int> classIndex;
    std::vector<cv::Rect2d> bbox;//for each object.
	std::vector<KalmanFilter> kalmanFilter;
};
*/
void Associate::mergeTracking_YOLO(
	TrackersYOLO& trackersYOLO,
	TrackersMOT& trackersMOT
)
{
	if (!trackersYOLO.bbox.empty()) {//YOLO data is avaiable.
		int frameIndex_tracking = trackersYOLO.frameIndex;//extract the frame index of the tracking result.
	}
}

void Associate::matching(
    std::vector<cv::Rect2d>& newRoi, std::vector<int>& newLabel, double& frameIndex,
    std::vector<std::vector<std::vector<double>>>& seqData, std::vector<std::vector<std::vector<double>>>& kfData,
    std::vector<KalmanFilter2D>& kalmanVector, std::vector<LinearExtrapolation2D>& extrapolation
)
{
    /**
    * matching data between Yolo and existed data with Kalmanfilter
    */

    //sequential data exists -> compare with new detections
    double dframe = 1.0;//interval of inference for kalmanfilter prediction
    Eigen::Vector<double, 6> kf_prediction;
    cv::Mat p_prediction;
    std::vector<double> newData;//{frameIndex,label,left,top,width,height}
    //latest frame
    double frame_new = frameIndex;
    if (!seqData.empty())//previous data is available.
    {
        std::vector<std::vector<double>> candidates; //candidates from sequence data.
        std::vector<double> candidate;
        int counter_update;
        double label, left, top, width, height, x_center, y_center;
        std::vector<double> data_latest;
        //iterate for each existed tracker
        for (int i = 0; i < seqData.size(); i++) //for each data. {0:frameIndex, 1:label, 2:left, 3:top, 4:width, 5:height }
        {
            counter_update = seqData[i].size();//number of updates
            if (counter_update >= COUNTER_VALID) {//valid trackers.-> adopt kalman prediction values.
                data_latest = seqData[i].back();
                label = data_latest[1];
                //compensation
                if (idx_compensation == 0) {//kalmanfilter
                    dframe = frame_new - kfData[i].back()[0];
                    kalmanVector[i].predict_only(kf_prediction, dframe);//predict current position
                    left = kf_prediction(0) - data_latest[4] / 2.0;
                    top = kf_prediction(1) - data_latest[5] / 2.0;
                    width = data_latest[4];
                    height = data_latest[5];
                    candidate = std::vector<double>{ label,left,top,width,height };
                }
                else if (idx_compensation == 1) {//linear extrapolation
                    dframe = frame_new - data_latest[0];
                    p_prediction = extrapolation[i].calculateNextPosition_temp(dframe);//{left,top,width,height}
                    x_center = p_prediction.at<double>(0);
                    y_center = p_prediction.at<double>(1);
                    width = p_prediction.at<double>(2);
                    height = p_prediction.at<double>(3);
                    left = x_center - width / 2.0;
                    top = y_center - height / 2.0;
                    candidate = std::vector<double>{ label,left,top,width,height };
                }
            }
            else {//not official trackers.
                candidate = seqData[i].back();
                candidate.erase(candidate.begin() + 0);//erase frameIndex.
            }
            //save candidate in candidates.
            candidates.push_back(candidate);
        }

        //compare existed and new trackers.
        std::vector<std::vector<double>> costMatrix;
        double cost_id, cost_rmse, cost_size, cost_total;
        cv::Rect2d roi_prev;
        for (std::vector<double>& prev : candidates) {//for each candidate {0:label,1:left,2:top,3:width,4:height}
            std::vector<double> cost_row;
            for (int idx_new = 0; idx_new < newRoi.size(); idx_new++) {//for each new roi.
                //tracker id
                cost_id = compareID((int)prev[0], newLabel[idx_new]);
                //RMSE
                roi_prev.x = prev[1]; roi_prev.y = prev[2]; roi_prev.width = prev[3]; roi_prev.height = prev[4];
                cost_rmse = calculateRMSE_Rect2d(roi_prev, newRoi[idx_new]);
                //size
                cost_size = sizeDiff(roi_prev, newRoi[idx_new]);
                //total
                cost_total = cost_id + cost_rmse * lambda_rmse_ + cost_size;
                cost_row.push_back(cost_total);
            }
            costMatrix.push_back(cost_row);
        }

        //match tracker based on hungarian algorithm.
        if (!costMatrix.empty()) {
            std::vector<int> assignment;
            double cost = HungAlgo.Solve(costMatrix, assignment);
            cv::Rect2d roi_match;
            std::vector<int> index_delete;//index list of the YOLO detections to be deleted.
            double frame_prev;
            //assign data according to indexList_tm, indexList_yolo and assign
            for (unsigned int x = 0; x < assignment.size(); x++) {//for each candidate from 0 to the end of the candidates. ascending way.
                int index_match = assignment[x];
                if (index_match >= 0) {//matching tracker is found.
                    if (costMatrix[x][index_match] < Cost_max) {//good candidate
                        index_delete.push_back(index_match);
                        //add ROI to sequence data and update kalman filter.
                        //extract data
                        roi_match = newRoi[index_match];
                        label = newLabel[index_match];
                        newData = std::vector<double>{ frame_new,label,roi_match.x,roi_match.y,roi_match.width,roi_match.height };
                        //save previous frame in advance.
                        frame_prev = seqData[x].back()[0];
                        //save data in seqData.
                        seqData[x].push_back(newData);

                        //update compensation module
                        dframe = frame_new - frame_prev;
                        if (idx_compensation == 0) {//kalman filter
                            observation << (roi_match.x + roi_match.width / 2.0), (roi_match.y + roi_match.height / 2.0);//{x_center, y_center}
                            dframe = frame_new - kfData[x].back()[0];
                            //predict and update
                            kalmanVector[x].predict(kf_predict, dframe, seqData[x]);
                            kalmanVector[x].update(observation);
                            Eigen::Vector<double, 6> filtered_data = kalmanVector[x].getState();
                            kfData[x].push_back({ frame_new,label,(filtered_data(0) - roi_match.width / 2.0),(filtered_data(1) - roi_match.height / 2.0),roi_match.width,roi_match.height });//{frameIndex, label, left, top, width,height}
                        }
                        else if (idx_compensation == 1) {//linear extrapolation
                            extrapolation[x].update(newData, dframe);
                            p_prediction = extrapolation[x].calculateNextPosition(dframe, seqData[x]);//{left,top,width,height}
                            x_center = p_prediction.at<double>(0);
                            y_center = p_prediction.at<double>(1);
                            width = p_prediction.at<double>(2);
                            height = p_prediction.at<double>(3);
                            left = x_center - width / 2.0;
                            top = y_center - height / 2.0;
                            kfData[x].push_back({ frame_new,label,left,top,width,height });//{frameIndex, label, left, top, width,height}
                        }
                    }
                    else {//if new tracker wasn't found.->kalman prediction and save in the seqData
                        if (idx_compensation == 0) {//kalman filter
                            dframe = frame_new - kfData[x].back()[0];
                            kalmanVector[x].predict(kf_prediction, dframe, seqData[x]); //predict kalman filter
                            if (seqData[x].back()[0] >= 0) {//still alive.
                                std::vector<double> temp = seqData[x].back();//frame,label,left,top,width,height
                                kfData[x].push_back({ (double)frame_new,temp[1],(kf_prediction(0) - temp[4] / 2.0),(kf_prediction(1) - temp[5] / 2.0),temp[4],temp[5] }); //add kalman prediction data
                            }
                        }
                        else if (idx_compensation == 1) {//linear extrapolation.
                            frame_prev = seqData[x].back()[0];
                            dframe = frame_new - frame_prev;
                            p_prediction = extrapolation[x].calculateNextPosition(dframe, seqData[x]); //left,top,width,height
                            if (seqData[x].back()[0] >= 0) {//still alive.
                                x_center = p_prediction.at<double>(0);
                                y_center = p_prediction.at<double>(1);
                                width = p_prediction.at<double>(2);
                                height = p_prediction.at<double>(3);
                                left = x_center - width / 2.0;
                                top = y_center - height / 2.0;
                                std::vector<double> temp = seqData[x].back();//frame,label,left,top,width,height
                                kfData[x].push_back({ (double)frame_new,temp[1],left,top,width,height }); //add kalman prediction data
                            }
                        }
                    }
                }
                else {//if new tracker wasn't found.->kalman prediction and save in the seqData
                    if (idx_compensation == 0) {//kalman prediction
                        frame_prev = kfData[x].back()[0];
                        dframe = frame_new - frame_prev;
                        kalmanVector[x].predict(kf_prediction, dframe, seqData[x]); //predict kalman filter
                        if (seqData[x].back()[0] >= 0) {//still alive.
                            std::vector<double> temp = seqData[x].back();//frame,label,left,top,width,height
                            kfData[x].push_back({ (double)frame_new,temp[1],(kf_prediction(0) - temp[4] / 2.0),(kf_prediction(1) - temp[5] / 2.0),temp[4],temp[5] }); //add kalman prediction data
                        }
                    }
                    else if (idx_compensation == 1) {//linear extrapolation.
                        frame_prev = seqData[x].back()[0];
                        dframe = frame_new - frame_prev;
                        p_prediction = extrapolation[x].calculateNextPosition(dframe, seqData[x]); //left,top,width,height
                        if (seqData[x].back()[0] >= 0) {//still alive.
                            x_center = p_prediction.at<double>(0);
                            y_center = p_prediction.at<double>(1);
                            width = p_prediction.at<double>(2);
                            height = p_prediction.at<double>(3);
                            left = x_center - width / 2.0;
                            top = y_center - height / 2.0;
                            std::vector<double> temp = seqData[x].back();//frame,label,left,top,width,height
                            kfData[x].push_back({ (double)frame_new,temp[1],left,top,width,height }); //add kalman prediction data
                        }
                    }
                }
            }
            //delete YOLO detections
            if (!index_delete.empty()) {
                // Sort the idx_delete vector in descending order
                std::sort(index_delete.rbegin(), index_delete.rend());
                for (int& idx : index_delete) {
                    newRoi.erase(newRoi.begin() + idx);
                    newLabel.erase(newLabel.begin() + idx);
                }
            }
        }
    }
    //newDetection is available.
    if (!newRoi.empty()) //new detection 
    {
        double label, left, top, width, height;
        double xCenter, yCenter;
        for (int idx = 0; idx < newRoi.size(); idx++)
        {
            label = newLabel[idx];
            left = newRoi[idx].x; top = newRoi[idx].y; width = newRoi[idx].width; height = newRoi[idx].height;
            seqData.push_back({ {frame_new,label,left,top,width,height} }); //frameIndex,label,left,top,width,height.
            //make kalmanfilter model
            xCenter = left + width / 2.0;
            yCenter = top + height / 2.0;

            if (idx_compensation == 0) {//kalman filter
                kalmanVector.push_back(KalmanFilter2D(xCenter, yCenter, INIT_VX, INIT_VY, INIT_AX, INIT_AY, NOISE_POS, NOISE_VEL, NOISE_ACC, NOISE_SENSOR));
                dframe = 1.0;
                kalmanVector.back().predict(kf_prediction, dframe, defaultVector);
                observation << xCenter, yCenter;
                kalmanVector.back().update(observation);
                kfData.push_back({ { (double)frame_new,label,(xCenter - width / 2.0),(yCenter - height / 2.0),width,height } }); //add kalman prediction to kfData
            }
            else if (idx_compensation == 1) {//linear extrapolation
                newData = std::vector<double>{ frame_new,label,left,top,width,height };
                extrapolation.push_back(LinearExtrapolation2D(newData));
                kfData.push_back({ { frame_new,label,left,top,width,height } });
            }
        }
    }//end if (!newData.empty())
}


void TemplateMatching::combineYoloTMData(cv::Mat1b& frame, std::vector<int>& classIndexesYolo, std::vector<int>& classIndexTM,
    std::vector<cv::Rect2d>& bboxesYolo, std::vector<cv::Rect2d>& bboxesTM,
    std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& trackersYolo, std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& trackers_mosse,
    std::vector<cv::Mat1b>& templatesYolo, std::vector<cv::Mat1b>& templatesTM,
    std::vector<std::vector<int>>& previousMove, cv::Mat1b& previousImg,
    std::vector<int>& updatedClasses, std::vector<cv::Rect2d>& updatedBboxes,
    std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& updatedTrackers, std::vector<cv::Mat1b>& updatedTemplates,
    std::vector<bool>& boolScalesTM, std::vector<std::vector<int>>& updatedMove, const int& numTrackersTM,std::vector<int>& num_notMove, std::vector<int>& updated_num_notMove)
{
    int counterYolo = 0;
    int counterTM = 0;      // for counting TM adaptations
    int counterClassTM = 0; // for counting TM class counter
    int counter_notMove = 0; //number of notMove -> if previous tracker exist or add new tracker -> add 1
    // organize current situation : determine if tracker is updated with Yolo or TM, and is deleted
    // think about tracker continuity : tracker survival : (not Yolo Tracker) and (not TM tracker)
    int numPastLabels = classIndexTM.size();
    /* should check carefully -> compare num of detection */
    for (const int& classIndex : classIndexesYolo)
    {
        ///if (!classIndexTM.empty()) //when comment out comment out this line, too!
        //   std::cout <<"numPastLabels="<<numPastLabels<<", classIndexTM.size()="<<classIndexTM.size()<<", bboxesTM.size()="<<bboxesTM.size()<<"classIndexTM="<<classIndexTM[counterClassTM]<< ",classIndexesYolo.size()=" << classIndexesYolo.size() << ", classYolo:" << classIndex << ", bboxesYolo.size()=" << bboxesYolo.size() << "counterClassTM=" << counterClassTM << ", counterTM=" << counterTM << std::endl;
        //std::cout << "num_notMove=" << updated_num_notMove.size() << ", updatedBboxes.size()=" << updatedBboxes.size() << std::endl;
        /* after 2nd time from YOLO data */
        if (numPastLabels > 0)
        {
            //std::cout << "TM tracker already exist" << std::endl;
            /* first numTrackersTM is existed Templates -> if same label, update else, unless tracking was successful lost tracker */
            if (counterClassTM < numPastLabels) // numTrackersTM : num ber of class indexes
            {
                /*update tracker*/
                if (classIndex >= 0)
                {
                    /* if classIndex != -1, update tracker. and if tracker of TM is successful, search aream can be limited */
                    if (classIndex == classIndexTM[counterClassTM])
                    {
                        //std::cout<<"classIndex="<<classIndexTM[counterClassTM]<<", tracker addess"<<trackers_mosse[counterTM]<<", template img size="<<
                        if (bool_check_psr) //check which tracker to adopt
                        {
                            //std::cout << "tracker psr=" << trackers_mosse[counterTM]->previous_psr << std::endl;
                            if (classIndex == 0) //circular objects
                            {
                                if (bool_comparePSR) //compare PSR
                                {
                                    double psr_yolo = check_tracker(previousImg, bboxesTM[counterTM], trackersYolo[counterYolo]); //calculate PSR of yolo tracker
                                    //adopt current tracker
                                    if ((trackers_mosse[counterTM]->previous_psr) > psr_yolo && (trackers_mosse[counterTM]->counter_skip <= 0)) {
                                        //std::cout << "keep rameined tracker" << std::endl;
                                        updatedTrackers.push_back(trackers_mosse[counterTM]); // update template to YOLO's one
                                        updatedTemplates.push_back(templatesTM[counterTM]); // update template to YOLO's one
                                    }
                                    //adopt yolo tracker
                                    else {
                                        updatedTrackers.push_back(trackersYolo[counterYolo]); // update template to YOLO's one
                                        updatedTemplates.push_back(templatesYolo[counterYolo]); // update template to YOLO's one
                                    }
                                }
                                else {
                                    if ((trackers_mosse[counterTM]->previous_psr) >= min_keep_psr)
                                    {
                                        //std::cout << "keep ramained tracker" << std::endl;
                                        updatedTrackers.push_back(trackers_mosse[counterTM]); // update template to YOLO's one
                                        updatedTemplates.push_back(templatesTM[counterTM]); // update template to YOLO's one
                                    }
                                    else
                                    {
                                        updatedTrackers.push_back(trackersYolo[counterYolo]); // update template to YOLO's one
                                        updatedTemplates.push_back(templatesYolo[counterYolo]); // update template to YOLO's one
                                    }
                                }
                            }
                            else if (classIndex == 1) //non-circular objects -> update every time
                            {
                                //std::cout << "keep rameined tracker" << std::endl;
                                updatedTrackers.push_back(trackers_mosse[counterYolo]); // update template to YOLO's one
                                updatedTemplates.push_back(templatesTM[counterYolo]); // update template to YOLO's one
                            }
                        }
                        else
                        {
                            updatedTrackers.push_back(trackersYolo[counterYolo]); // update template to YOLO's one
                            updatedTemplates.push_back(templatesYolo[counterYolo]); // update template to YOLO's one
                        }
                        //updatedTrackers.push_back(trackersYolo[counterYolo]); // update template to YOLO's one
                        if (bool_TBD) updatedBboxes.push_back(bboxesTM[counterYolo]);// update bbox with Yolo data
                        else updatedBboxes.push_back(bboxesTM[counterTM]);          // update bbox with TM one
                        updatedClasses.push_back(classIndex);                       // update class
                        boolScalesTM.push_back(true);                               // scale is set to TM
                        updatedMove.push_back(previousMove[counterTM]);
                        updated_num_notMove.push_back(num_notMove[counterTM]);
                        counterTM++;
                        counterYolo++;
                        counterClassTM++;
                    }
                    /* trakcer of TM was failed */
                    else
                    {
                        updatedTrackers.push_back(trackersYolo[counterYolo]); // update template to YOLO's one
                        updatedTemplates.push_back(templatesYolo[counterYolo]); // update template to YOLO's one
                        updatedBboxes.push_back(bboxesYolo[counterYolo]);       // update bbox to YOLO's one
                        updatedClasses.push_back(classIndex);                       // update class to YOLO's one
                        boolScalesTM.push_back(false);                              // scale is set to Yolo
                        updatedMove.push_back(defaultMove);
                        updated_num_notMove.push_back(0); //add not_move count
                        counterYolo++;
                        counterClassTM++;
                    }
                }
                /* tracker not found in YOLO */
                else
                {
                    /* template matching was successful -> keep tracking */
                    if (classIndexTM[counterClassTM] >= 0)
                    {
                        updatedTrackers.push_back(trackers_mosse[counterTM]); // update tracker to TM's one
                        updatedTemplates.push_back(templatesTM[counterTM]); // update tracker to TM's one
                        updatedBboxes.push_back(bboxesTM[counterTM]);       // update bbox to TM's one
                        updatedClasses.push_back(classIndexTM[counterClassTM]);
                        boolScalesTM.push_back(true); // scale is set to TM
                        updatedMove.push_back(previousMove[counterTM]);
                        updated_num_notMove.push_back(num_notMove[counterTM]);
                        counterTM++;
                        counterClassTM++;
                    }
                    /* both tracking was failed -> lost */
                    else
                    {
                        updatedClasses.push_back(classIndex);
                        counterClassTM++;
                    }
                }
            }
            /* new tracker -> add new templates * maybe in this case all calss labels should be positive, not -1 */
            else
            {
                if (classIndex >= 0)
                {
                    //std::cout << "add new tracker" << std::endl;
                    updatedTrackers.push_back(trackersYolo[counterYolo]); // update template to YOLO's one
                    updatedTemplates.push_back(templatesYolo[counterYolo]); // update template to YOLO's one
                    updatedBboxes.push_back(bboxesYolo[counterYolo]);       // update bbox to YOLO's one
                    updatedClasses.push_back(classIndex);                       // update class to YOLO's one
                    boolScalesTM.push_back(false);                              // scale is set to Yolo
                    updatedMove.push_back(defaultMove);
                    updated_num_notMove.push_back(0);
                    counterYolo++;
                }
                /* this is for exception, but prepare for emergency*/
                else
                {
                    //std::cout << "this is exception:: even if new tracker, class label is -1. Should revise code " << std::endl;
                    updatedClasses.push_back(classIndex);
                }
            }
        }
        /* for the first time from YOLO data */
        else
        {
            //std::cout << "first time of TM" << std::endl;
            /* tracker was successful */
            if (classIndex >= 0)
            {
                updatedTrackers.push_back(trackersYolo[counterYolo]); // update template to YOLO's one
                updatedTemplates.push_back(templatesYolo[counterYolo]); // update template to YOLO's one
                updatedBboxes.push_back(bboxesYolo[counterYolo]);       // update bbox to YOLO's one
                updatedClasses.push_back(classIndex);                       // update class to YOLO's one
                boolScalesTM.push_back(false);                              // scale is set to Yolo
                updatedMove.push_back(defaultMove);
                updated_num_notMove.push_back(0);
                counterYolo++;
            }
            /* tracker was not found in YOLO */
            else
            {
                updatedClasses.push_back(classIndex);
            }
        }
    }
    std::cout << "num_notMove=" << updated_num_notMove.size() << ", updatedBboxes.size()=" << updatedBboxes.size() << std::endl;
    //IoU check -> delete duplicated trackers
    if (bool_iouCheck)
    {
        if (updatedBboxes.size() >= 2)
        {
            std::cout << " /////////////////////////// check Duplicated trackers" << std::endl;
            //std::cout << "num_notMove=" << num_notMove.size() << ", trackers="<<updatedTrackers.size()<<", bboxes=" << updatedBboxes.size() << std::endl;
            int counterLabels = 0;
            std::vector<int> labels_on; //successful tracker index
            for (int& label : updatedClasses) //gatcher successful trackers index
            {
                if (label >= 0)
                {
                    labels_on.push_back(counterLabels);
                }
                counterLabels++;//increment counter of templates
            }
            int counter_template = 0;
            cv::Rect2d roi_template;
            while (true)
            {
                if (counter_template >= updatedBboxes.size() - 1) break;
                roi_template = updatedBboxes[counter_template]; //base template
                int i = counter_template + 1;
                while (true)
                {
                    if (i >= updatedBboxes.size()) break; //terminate condition
                    if (updatedClasses[labels_on[counter_template]] == updatedClasses[labels_on[i]]) //same label
                    {
                        float iou = calculateIoU_Rect2d(roi_template, updatedBboxes[i]);
                        //std::cout << "iou=" << iou << std::endl;
                        if (iou >= IoU_overlapped) //duplicated tracker -> delete template,roi and scales and convert class label to -1
                        {
                            std::cout << " //////////////////////////////// overlapped tracker: iou=" << iou << std::endl;
                            if (bool_augment)
                            {
                                //augment tracker and delete new tracker
                                double left = std::min(updatedBboxes[i].x, updatedBboxes[counter_template].x);
                                double right = std::max((updatedBboxes[i].x + updatedBboxes[i].width), (updatedBboxes[counter_template].x + updatedBboxes[counter_template].width));
                                double top = std::min(updatedBboxes[i].y, updatedBboxes[counter_template].y);
                                double bottom = std::max((updatedBboxes[i].y + updatedBboxes[i].height), (updatedBboxes[counter_template].y + updatedBboxes[counter_template].height));
                                if ((0 < left && left < right && right < frame.cols) && (0 < top && top < bottom && bottom < frame.rows))
                                {
                                    cv::Rect2d newRoi(left, top, (right - left), (bottom - top));
                                    updatedBboxes[counter_template] = newRoi;
                                    updatedTemplates[counter_template] = previousImg(newRoi); //change to previousFrame(newRoi);
                                    cv::Ptr<cv::mytracker::TrackerMOSSE> tracker = cv::mytracker::TrackerMOSSE::create();
                                    tracker->init(previousImg, newRoi); //change to tracker->init(previousFrame,newRoi);
                                    updatedTrackers[counter_template] = tracker;
                                    updatedMove[counter_template][0] = (int)((updatedMove[counter_template][0] + updatedMove[i][0]) / 2);
                                    updatedMove[counter_template][1] = (int)((updatedMove[counter_template][1] + updatedMove[i][1]) / 2);
                                    updated_num_notMove[counter_template] = std::min(updated_num_notMove[counter_template], updated_num_notMove[i]);
                                    //delete tracker
                                    updatedTrackers.erase(updatedTrackers.begin() + i);
                                    updatedTemplates.erase(updatedTemplates.begin() + i);
                                    updatedBboxes.erase(updatedBboxes.begin() + i);
                                    boolScalesTM.erase(boolScalesTM.begin() + i);
                                    updatedClasses[labels_on[i]] = -2;
                                    updatedMove.erase(updatedMove.begin() + i);
                                    updated_num_notMove.erase(updated_num_notMove.begin() + i);
                                    labels_on.erase(labels_on.begin() + i);
                                }
                                else
                                {
                                    if ((updatedTrackers[i]->previous_psr) >= (updatedTrackers[counter_template]->previous_psr))
                                    {
                                        //exchange tracker
                                        updatedTrackers[counter_template] = updatedTrackers[i];
                                        updatedTemplates[counter_template] = updatedTemplates[i];
                                        updatedBboxes[counter_template] = updatedBboxes[i];
                                        updatedMove[counter_template] = updatedMove[i];
                                        updated_num_notMove[counter_template] = std::min(updated_num_notMove[counter_template], updated_num_notMove[i]);
                                    }
                                    updatedTrackers.erase(updatedTrackers.begin() + i);
                                    updatedTemplates.erase(updatedTemplates.begin() + i);
                                    updatedBboxes.erase(updatedBboxes.begin() + i);
                                    boolScalesTM.erase(boolScalesTM.begin() + i);
                                    updatedClasses[labels_on[i]] = -2;
                                    updatedMove.erase(updatedMove.begin() + i);
                                    updated_num_notMove.erase(updated_num_notMove.begin() + i);
                                    labels_on.erase(labels_on.begin() + i);
                                }
                            }
                            else if (bool_checkVel)
                            {
                                int vx_base = updatedMove[counter_template][0]; int vy_base = updatedMove[counter_template][1];
                                int vx_cand = updatedMove[i][0]; int vy_cand = updatedMove[i][1];
                                float norm_base = std::pow((std::pow(vx_base, 2) + std::pow(vy_base, 2)), 0.5);
                                float norm_cand = std::pow((std::pow(vx_cand, 2) + std::pow(vy_cand, 2)), 0.5);
                                float cos = ((vx_base * vx_cand) + (vy_base * vy_cand)) / (norm_base * norm_cand);//check direction
                                if (cos <= thresh_cos_dup) //judge as another objects -> move ROI of another things
                                {
                                    int dx_cand = (int)(vx_cand / norm_cand * delta_move);
                                    int dy_cand = (int)(vy_cand / norm_cand * delta_move);
                                    //move duplicated bboxes position
                                    updatedBboxes[i].x += dx_cand;
                                    updatedBboxes[i].y += dy_cand;
                                }
                                else //same objects -> delete duplicated one
                                {
                                    //augment tracker and delete new tracker
                                    double left = std::min(updatedBboxes[i].x, updatedBboxes[counter_template].x);
                                    double right = std::max((updatedBboxes[i].x + updatedBboxes[i].width), (updatedBboxes[counter_template].x + updatedBboxes[counter_template].width));
                                    double top = std::min(updatedBboxes[i].y, updatedBboxes[counter_template].y);
                                    double bottom = std::max((updatedBboxes[i].y + updatedBboxes[i].height), (updatedBboxes[counter_template].y + updatedBboxes[counter_template].height));
                                    if ((0 < left && left < right && right < frame.cols && (right - left)>10) && (0 < top && top < bottom && bottom < frame.rows && (bottom - top)>10))
                                    {
                                        cv::Rect2d newRoi(left, top, (right - left), (bottom - top));
                                        updatedBboxes[counter_template] = newRoi;
                                        updatedTemplates[counter_template] = previousImg(newRoi);//change to previousFrame(newRoi);
                                        cv::Ptr<cv::mytracker::TrackerMOSSE> tracker = cv::mytracker::TrackerMOSSE::create();
                                        tracker->init(previousImg, newRoi); //change to tracker->init(previousFrame,newRoi);
                                        updatedTrackers[counter_template] = tracker;
                                        updatedMove[counter_template][0] = (int)((updatedMove[counter_template][0] + updatedMove[i][0]) / 2);
                                        updatedMove[counter_template][1] = (int)((updatedMove[counter_template][1] + updatedMove[i][1]) / 2);
                                        updated_num_notMove[counter_template] = std::min(updated_num_notMove[counter_template], updated_num_notMove[i]);
                                        //delete tracker
                                        updatedTrackers.erase(updatedTrackers.begin() + i);
                                        updatedTemplates.erase(updatedTemplates.begin() + i);
                                        updatedBboxes.erase(updatedBboxes.begin() + i);
                                        boolScalesTM.erase(boolScalesTM.begin() + i);
                                        updatedClasses[labels_on[i]] = -2;
                                        updatedMove.erase(updatedMove.begin() + i);
                                        updated_num_notMove.erase(updated_num_notMove.begin() + i);
                                        labels_on.erase(labels_on.begin() + i);
                                    }
                                    else
                                    {
                                        if ((updatedTrackers[i]->previous_psr) > (updatedTrackers[counter_template]->previous_psr))
                                        {
                                            //exchange trackere
                                            updatedTrackers[counter_template] = updatedTrackers[i];
                                            updatedTemplates[counter_template] = updatedTemplates[i];
                                            updatedBboxes[counter_template] = updatedBboxes[i];
                                            updatedMove[counter_template] = updatedMove[i];
                                            updated_num_notMove[counter_template] = std::min(updated_num_notMove[counter_template], updated_num_notMove[i]);
                                        }
                                        updatedTrackers.erase(updatedTrackers.begin() + i);
                                        updatedTemplates.erase(updatedTemplates.begin() + i);
                                        updatedBboxes.erase(updatedBboxes.begin() + i);
                                        boolScalesTM.erase(boolScalesTM.begin() + i);
                                        updatedClasses[labels_on[i]] = -2;
                                        updatedMove.erase(updatedMove.begin() + i);
                                        updated_num_notMove.erase(updated_num_notMove.begin() + i);
                                        labels_on.erase(labels_on.begin() + i);
                                    }
                                }
                            }
                            else
                            {
                                if ((updatedTrackers[i]->previous_psr) > (updatedTrackers[counter_template]->previous_psr))
                                {
                                    //exchange trackere
                                    updatedTrackers[counter_template] = updatedTrackers[i];
                                    updatedTemplates[counter_template] = updatedTemplates[i];
                                    updatedBboxes[counter_template] = updatedBboxes[i];
                                    updatedMove[counter_template] = updatedMove[i];
                                    updated_num_notMove[counter_template] = std::min(updated_num_notMove[counter_template], updated_num_notMove[i]);
                                }
                                updatedTrackers.erase(updatedTrackers.begin() + i);
                                updatedTemplates.erase(updatedTemplates.begin() + i);
                                updatedBboxes.erase(updatedBboxes.begin() + i);
                                boolScalesTM.erase(boolScalesTM.begin() + i);
                                updatedClasses[labels_on[i]] = -2;
                                updatedMove.erase(updatedMove.begin() + i);
                                updated_num_notMove.erase(updated_num_notMove.begin() + i);
                                labels_on.erase(labels_on.begin() + i);
                            }
                        }
                        else i++;
                    }
                    else i++; //other labels
                }
                counter_template++;
            }
        }
    }
}

double TemplateMatching::check_tracker(cv::Mat1b& previousImg, cv::Rect2d& roi, cv::Ptr<cv::mytracker::TrackerMOSSE>& tracker)
{
    double scale_x = scaleXTM;
    double scale_y = scaleYTM;
    int leftSearch = std::min(previousImg.cols, std::max(0, static_cast<int>(roi.x - (scale_x - 1) * roi.width / 2)));
    int topSearch = std::min(previousImg.rows, std::max(0, static_cast<int>(roi.y - (scale_y - 1) * roi.height / 2)));
    int rightSearch = std::max(0, std::min(previousImg.cols, static_cast<int>(roi.x + (scale_x + 1) * roi.width / 2)));
    int bottomSearch = std::max(0, std::min(previousImg.rows, static_cast<int>(roi.y + (scale_y + 1) * roi.height / 2)));
    if ((rightSearch - leftSearch) > 0 && (bottomSearch - topSearch) > 0)
    {
        cv::Rect2d searchArea(leftSearch, topSearch, (rightSearch - leftSearch), (bottomSearch - topSearch));
        cv::Mat1b croppedImg = previousImg.clone();
        croppedImg = croppedImg(searchArea); // crop img
        //convert roi from image coordinate to local search area coordinate
        cv::Rect2d croppedRoi;
        croppedRoi.x = roi.x - searchArea.x;
        croppedRoi.y = roi.y - searchArea.y;
        croppedRoi.width = roi.width;
        croppedRoi.height = roi.height;
        // MOSSE Tracker
        double psr = tracker->check_quality(croppedImg, croppedRoi, true);
        return psr;
    }
    else
        return 0;
}
