#include "../include/tracking.h"


void Tracking::main(std::queue<bool>& q_startTracker)
{
    //constructor
    int countIteration = 0;
    int counterFinish = 0;
    int counterStart = 0;
    while (true)
    {
        if (counterStart == 3)
            break;
        if (!q_startTracker.empty()) {
            q_startTracker.pop();
            counterStart++;
            std::cout << "Tracker :: by starting " << 3 - counterStart << std::endl;
        }
    }

    std::cout << "start tracking" << std::endl;
    while (true) // continue until finish
    {
        if (queueFrame.empty())
        {
            if (counterFinish == 30)
                break;
            counterFinish++;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            std::cout << "Tracker :: By finish : remain count is " << (30 - counterFinish) << std::endl;
            continue;
        }
        else if (!queueFrame.empty())
        {
            if (!q_mot2tracking_left.empty() && !q_mot2tracking_right.empty()) //Trackers are detected in both cameras. -> required for 3D triangulation.
            {
				//get data from MOT queue.
				Trackers trackers_left = q_mot2tracking_left.front();
				Trackers trackers_right = q_mot2tracking_right.front();
				q_mot2tracking_left.pop();
				q_mot2tracking_right.pop();
                //std::cout << !queueTrackerYolo_left.empty() << !queueTrackerYolo_right.empty() << !queueTrackerMOSSE_left.empty() << !queueTrackerMOSSE_right.empty() << !queueKfPredictLeft.empty() << !queueKfPredictRight.empty() << std::endl;
                counterFinish = 0; // reset
                std::array<cv::Mat1b, 2> frames;
                int frameIndex;
                bool boolImgs = Utility::getImagesFromQueueMot(frames, frameIndex);
                cv::Mat1b frame_left = frames[LEFT_CAMERA];
                cv::Mat1b frame_right = frames[RIGHT_CAMERA];
                if ((frame_left.rows > 0 && frame_right.rows > 0))
                {
                    /*start template matching process */
                    auto start = std::chrono::high_resolution_clock::now();
                    std::thread thread_left(&Tracking::track, this, std::ref(frame_left), std::ref(frameIndex),std::ref(trackers_left));
                    //right
                    track(frame_right, frameIndex,trackers_right);
                    thread_left.join();
                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                    float time_iteration = static_cast<float>(duration.count());
                    t_elapsed = t_elapsed + time_iteration;
                    if (time_iteration < 2500)
                    {
                        std::cout << "Tracker :: " << time_iteration << " microseconds" << std::endl;
                    }
                    else
                    {
                        int frame_delete = static_cast<int>((time_iteration / 2500) - 1);
                        if (frame_delete >= 1)
                        {
                            std::cout << "Tracker :: " << time_iteration << " microseconds and " << frame_delete << " frames will be deleted" << std::endl;
                            for (int i = 0; i < frame_delete; i++)
                            {
                                if (!queueFrame.empty()) queueFrame.pop();
                                if (!queueFrameIndex.empty()) queueFrameIndex.pop();
                            }
                        }
                    }
                    countIteration++;
                }
            }
        }
    }
    if (countIteration != 0) 
		std::cout << "Tracking process speed :: " << static_cast<int>(countIteration / t_elapsed * 1000000) << " Hz for" << countIteration << "cycles" << std::endl;
    // check data
}

/**
 * Track only chosen trackers to manage the number of threads.
 * struct TrackerInfo{
	cv::Ptr<cv::mytracker::TrackerMOSSE> mosse;
	cv::Ptr<TemplateMatching> template_matching;//have a template image internally.
	cv::Rect2d bbox;
	double scale_searcharea; //scale ratio of search area against bbox
    cv::Point2d vel; //previous velocity
	int n_notMove; //number of not move
};

struct Trackers {
    std::vector<int> classIndex;
	std::vector<int> index_highspeed;//index for high speed tracking
	std::vector<TrackerInfo> trackerInfo;
    cv::Mat1b previousImg;
};
 */
*
*/
std::vector<bool> Tracking::track(cv::Mat1b& img, const int& frameIndex,Trackers& trackers)
{
    //prepare containers
    cv::Mat1b previousImg = trackers.previousImg; //previous image
	std::vector<bool> success_flags;

	//Trackg object within index_highspeed.
	for (int i = 0; i < trackers.index_highspeed.size(); i++)
	{
		//get tracker for high speed tracking.
		int index = trackers.index_highspeed[i];
		TrackerInfo tracker = trackers.trackerInfo[index];

		//Search area setting.
		cv::Rect2d bbox = tracker.bbox;//get the previous tracker.
		// Clip speed with std::clamp instead.
		cv::Point2d vel_object = cv::Point2d(std::clamp(tracker.vel.x, -1.0 * _max_move, _max_move), std::clamp(tracker.vel.y, -1.0 * _max_move, _max_move)); //get the previous object's movement.
		//set the search area. -> move the previous center by previous velocity.
		cv::Point2d center_search(bbox.x + bbox.width / 2 + vel_object.x, bbox.y + bbox.height / 2 + vel_object.y);
		//width and height setting.
		double width_search = bbox.width * tracker.scale_searcharea;
		double height_search = bbox.height * tracker.scale_searcharea;
		//roi_search area setting with integer and image size.setting.
		int leftSearch = std::max(0, std::min(img.cols, static_cast<int>(center_search.x - width_search / 2)));
		int topSearch = std::max(0, std::min(img.rows, static_cast<int>(center_search.y - height_search / 2)));
		int rightSearch = std::min(img.cols, static_cast<int>(center_search.x + width_search / 2));
		int bottomSearch = std::min(img.rows, static_cast<int>(center_search.y + height_search / 2));
		cv::Rect2 roi_search_area(leftSearch, topSearch, rightSearch - leftSearch, bottomSearch - topSearch);
		//Crop an image..
		cv::Mat1b croppedImg = img.clone();
		//Crop an image.
		croppedImg = croppedImg(roi_search_area);
		//transform the coordinate of the bbox to the cropped image.
		cv::Rect2d bbox_in_search = cv::Rect2d(bbox.x - leftSearch, bbox.y - topSearch, bbox.width, bbox.height);

		bool success_tracking = false;
		//update tracker with simultaneous update with thread.
		if (_mode_tracking >= 2)
		{
			cv::Rect2d bbox_in_search_tm = bbox_in_search;//for template matching.
			cv::Rect2d bbox_in_search_mosse = bbox_in_search;
			bool success_tm = false;
			bool success_mosse = false;
			if (_mode_tracking >= 3){//multithread
				std::thread threadTM([&](){
					success_tm = this->track_template_matching(croppedImg, tracker.template_matching, bbox_in_search_tm);
				});
				success_mosse = track_mosse(croppedImg, tracker.mosse, bbox_in_search_mosse);
				threadTM.join();
			}
			else{//single thread.
				success_tm = track_template_matching(croppedImg, tracker.template_matching, bbox_in_search_tm);
				success_mosse = track_mosse(croppedImg, tracker.mosse, bbox_in_search_mosse);
			}


			if (success_tm && success_mosse)//both are successful.
			{
				success_tracking = true;
				//update bounding box. -> average.
				tracker.bbox.x = (bbox_in_search_mosse.x + bbox_in_search_tm.x)/2.0;
				tracker.bbox.y = (bbox_in_search_mosse.y + bbox_in_search_tm.y)/2.0;
				tracker.bbox.width = (bbox_in_search_mosse.width + bbox_in_search_tm.width)/2.0;
				tracker.bbox.height = (bbox_in_search_mosse.height + bbox_in_search_tm.height)/2.0;
				//if necessary -> update correlation filter and template image.
				if (bbox_in_search_mosse.width != tracker.bbox.width || bbox_ini_search_mosse.height != tracker.bbox.height){
					//size has been chnaged. -> update correlation filter.
					cv::Point3d scores_tmp= tracker.mosse->_scores;//save scores.
					tracker.mosse->init(croppedImg,tracker.bbox);//init tracker.
					tracker.mosse->_scores = scores_tmp;//succeed _scores.
				}
				if (bbox_in_search_tm.width != tracker.bbox.width || bbox_ini_search_tm.height != tracker.bbox.height){
					//size has been chnaged. -> update correlation filter.
					cv::Point3d scores_tmp= tracker.template_matching->_scores;//save scores.
					tracker.template_matching->init(croppedImg,tracker.bbox);//init tracker.
					tracker.template_matching->_scores = scores_tmp;//succeed _scores.
				}
			}
			else if (success_tm && !success_mosse){//Only TM succeeded. -> init MOSSE.]
				success_tracking = true;
				//update bounding box.
				tracker.bbox = bbox_in_search_tm;
				tracker.mosse->init(croppedImg,tracker.bbox);//init tracker.
			}
			else if (!success_tm && success_mosse){//Only MOsse succeeded. -> Init. TM.
				success_tracking = true;
				//update bounding box.
				tracker.bbox = bbox_in_search_mosse;
				tracker.template_matching->init(croppedImg,tracker.bbox);//init tracker.
			}
		}
		else if (_mode_tracking == 1){//Template matching
		{
			success_tracking = track_template_matching(croppedImg, tracker.template_matching, bbox_in_search);
			tracker.bbox = bbox_in_search;//eve
		}
		else if (_mode_tracking == 0){
			success_tracking = track_mosse(img, tracker.mosse, bbox_in_search);
			tracker.bbox = bbox_in_search;
		}

		//transform bbox to global image frame.
		tracker.bbox.x += leftSearch;
		tracker.bbox.y += topSearch;

		//update velocity.
		tracker.vel = cv::Point2d(tracker.bbox.x - bbox.x, tracker.bbox.y - bbox.y);
		
		//update scaler_esarch_area.
		tracker.scale_searcharea = cv::norm(tracker.vel)/(std::sqrt(2.0)*_max_move)*(_scale_max-_scale_min)+_scale_min;
		tracker.scale_searcharea = std::clamp(tracker.scale_searcharea, _scale_min, _scale_max);

		//update trackers.
		trackers.trackerInfo[index] = tracker;//update tracker information.
	}
    
}

/* Return seccess flag. */
bool Tracking::track_mosse(cv::Mat1b img, cv::Ptr<cv::mytracker::TrackerMOSSE> tracker, cv::Rect2d bbox){
	// MOSSE Tracker
	bool success = tracker->update(img, bbox); //bbox and correlation filter are updated internally if psr>_psrThreshold
	
	if (success)//tracking is successful.
		return true;
	else
		return false;
}

bool Tracking::track_template_matching(cv::Mat1b img, cv::Ptr<TemplateMatching> template_matching, cv::Rect2d bbox){
	bool success = template_matching->update(img, bbox);//uppdate bounding box and template image internally.
	if (success)
		return true;
	else
		return false;
}


void TemplateMatching::getData(cv::Mat1b& frame, bool& boolTrackerTM, std::vector<int>& classIndexTM, std::vector<cv::Rect2d>& bboxesTM, std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& trackers_mosse,
    std::vector<cv::Mat1b>& templatesTM, std::vector<bool>& boolScalesTM, std::vector<std::vector<int>>& previousMove,
    int& numTrackersTM, cv::Mat1b& previousImg, std::vector<int>& num_notMove, std::queue<Tracker2tracker>& q_tracker2tracker, std::queue<std::vector<std::vector<double>>>& q_seq2tracker)
{
    if (!q_tracker2tracker.empty())
    {
        //make a instance
        Tracker2tracker newData = q_tracker2tracker.front();
        q_tracker2tracker.pop();
        //classIndex
        if (!newData.classIndex.empty())
        {
            //classIndex
            classIndexTM = newData.classIndex;
            numTrackersTM = classIndexTM.size();
            //previousImg
            previousImg = newData.previousImg;
        }
        //bbox
        if (!newData.bbox.empty()) {
            boolTrackerTM = true;
            //bbox
            bboxesTM = newData.bbox;
            //tracker
            trackers_mosse = newData.tracker;
            //template
            templatesTM = newData.templateImg;
            //scale
            boolScalesTM = newData.scale;
            //previous velocity
            previousMove = newData.vel;
            //num not move
            num_notMove = newData.num_notMove;
            //std::cout << "number of notMove ";
            //for (int i = 0; i < num_notMove.size(); i++)
            //    std::cout << num_notMove[i] << " ";
            //std::cout << std::endl;
            //std::cout <<"num of classes="<<classIndexTM<< ", num of notMove=" << num_notMove.size() << ", bboxesTM.size()=" << bboxesTM.size() << std::endl;
        }
        else
            boolTrackerTM = false;
    }
    //Kalman filter data compensation
    if (bool_kf)
    {
        if (!q_seq2tracker.empty())
        {
            std::vector<std::vector<double>> kf_predictions = q_seq2tracker.front();
            q_seq2tracker.pop();
            int counter_label = 0;
            int counter_tracker = 0;
            for (std::vector<double>& kf_predict : kf_predictions)
            {
                if (!kf_predict.empty() && classIndexTM[counter_label] < 0 && classIndexTM[counter_label] != -2) //revival
                {
                    classIndexTM[counter_label] = (int)kf_predict[0]; //update label
                    cv::Rect2d newRoi((double)std::min(std::max((int)kf_predict[1], 0), (frame.cols - (int)kf_predict[3] - 1)), (double)std::min(std::max((int)kf_predict[2], 0), (frame.rows - (int)kf_predict[4] - 1)), (double)kf_predict[3], (double)kf_predict[4]);
                    bboxesTM.insert(bboxesTM.begin() + counter_tracker, newRoi);
                    cv::Ptr<cv::mytracker::TrackerMOSSE> tracker = cv::mytracker::TrackerMOSSE::create();
                    tracker->init(frame, newRoi);//update tracker with current frame
                    trackers_mosse.insert(trackers_mosse.begin() + counter_tracker, tracker);
                    templatesTM.insert(templatesTM.begin() + counter_tracker, frame(newRoi)); //template
                    boolScalesTM.insert(boolScalesTM.begin() + counter_tracker, false); //scale
                    previousMove.insert(previousMove.begin() + counter_tracker, defaultMove); //previous move velocity
                    num_notMove.insert(num_notMove.begin() + counter_tracker, 0); //number of not moving times
                    boolTrackerTM = true;
                    //std::cout << "compensate with KF data" << std::endl;
                }
                if (classIndexTM[counter_label] >= 0) counter_tracker++;
                counter_label++;
            }
        }
    }
}

void TemplateMatching::organizeData(cv::Mat1b& frame, std::vector<int>& classIndexTM, std::vector<cv::Rect2d>& bboxesTM,
    std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& trackers_mosse, std::vector<cv::Mat1b>& templatesTM,
    std::vector<bool>& boolScalesTM, std::vector<std::vector<int>>& previousMove, cv::Mat1b& previousImg,
    bool& boolTrackerYolo,
    std::vector<int>& updatedClasses, std::vector<cv::Rect2d>& updatedBboxes,
    std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& updatedTrackers, std::vector<cv::Mat1b>& updatedTemplates,
    std::vector<std::vector<int>>& updatedMove, int& numTrackersTM,std::vector<int>& num_notMove, std::vector<int>& updated_num_notMove,
    std::queue<Yolo2tracker>& q_yolo2tracker)
{
    //std::unique_lock<std::mutex> lock(mtxYolo); // Lock the mutex
    //std::cout << "TM :: Yolo data is available" << std::endl;
    boolTrackerYolo = true;
    if (!boolScalesTM.empty())
    {
        boolScalesTM.clear(); // clear all elements of scales
    }
    // get Yolo data
    std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>> trackersYolo;
    trackersYolo.reserve(10); // get new data
    std::vector<cv::Mat1b> templatesYolo;
    templatesYolo.reserve(10); // get new data
    std::vector<cv::Rect2d> bboxesYolo;
    bboxesYolo.reserve(10); // get current frame data
    std::vector<int> classIndexesYolo;
    classIndexesYolo.reserve(150);
    //get Yolo data
    getYoloData(trackersYolo, templatesYolo, bboxesYolo, classIndexesYolo, q_yolo2tracker); // get new frame
    // combine Yolo and TM data, and update latest data
    combineYoloTMData(frame, classIndexesYolo, classIndexTM, bboxesYolo, bboxesTM, trackersYolo, trackers_mosse, templatesYolo, templatesTM, previousMove, previousImg,
        updatedClasses, updatedBboxes, updatedTrackers, updatedTemplates, boolScalesTM, updatedMove, numTrackersTM,num_notMove,updated_num_notMove);
}

void TemplateMatching::getYoloData(std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& newTrackers, std::vector<cv::Mat1b>& newTemplates, std::vector<cv::Rect2d>& newBboxes, std::vector<int>& newClassIndexes,
    std::queue<Yolo2tracker>& q_yolo2tracker)
{
    Yolo2tracker newData;
    newData = q_yolo2tracker.front();
    q_yolo2tracker.pop();
    //classIndex
    newClassIndexes = newData.classIndex;
    //bbox
    if (!newData.bbox.empty()) newBboxes = newData.bbox;
    //tracker
    if (!newData.tracker.empty()) newTrackers = newData.tracker;
    //template
    if (!newData.templateImg.empty()) newTemplates = newData.templateImg;
}

float TemplateMatching::calculateIoU_Rect2d(const cv::Rect2d& box1, const cv::Rect2d& box2)
{
    float left = std::max(box1.x, box2.x);
    float top = std::max(box1.y, box2.y);
    float right = std::min((box1.x + box1.width), (box2.x + box2.width));
    float bottom = std::min((box1.y + box1.height), (box2.y + box2.height));

    if (left < right && top < bottom)
    {
        float intersection = (right - left) * (bottom - top);
        float area1 = box1.width * box1.height;
        float area2 = box2.width * box2.height;
        float unionArea = area1 + area2 - intersection;

        return intersection / unionArea;
    }

    return 0.0f; // No overlap
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
