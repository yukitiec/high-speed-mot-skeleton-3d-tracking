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
	auto time_last_process = std::chrono::high_resolution_clock::now();
	//Prevent overprocessing.
	double time_to_wait = 1e6/static_cast<double>(GP::FPS);//in microseconds
    while (true) // continue until finish
    {
        if (!q_finish_tracking.empty()){
			q_finish_tracking.pop();
            break;
		}

		auto time_current = std::chrono::high_resolution_clock::now();
		auto duration_wait = std::chrono::duration_cast<std::chrono::microseconds>(time_current - time_last_process);
		if (duration_wait.count() >= time_to_wait && !queueFrame.empty())
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
                    auto start = std::chrono::high_resolution_clock::now();

					//Tracking process.
                    std::vector<bool> success_flags_left;
                    std::thread thread_left([&](){
                        success_flags_left = track(frame_left, frameIndex, trackers_left);
                    });
                    //right
                    std::vector<bool> success_flags_right = track(frame_right, frameIndex,trackers_right);
                    thread_left.join();
					//Finish 

					//send updated trackers to MOT.
					Trackers2MOT trackers2mot_left, trackers2mot_right;
					trackers2mot_left.success_flags = success_flags_left;
					trackers2mot_left.trackers = trackers_left;
					trackers2mot_right.success_flags = success_flags_right;
					trackers2mot_right.trackers = trackers_right;
					//Pop before push.
					if (!q_tracking2mot_left.empty())
						q_tracking2mot_left.pop();
					if (!q_tracking2mot_right.empty())
						q_tracking2mot_right.pop();
					q_tracking2mot_left.push(trackers2mot_left);
					q_tracking2mot_right.push(trackers2mot_right);

                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                    float time_iteration = static_cast<float>(duration.count());
                    t_elapsed = t_elapsed + time_iteration;
                    
					if (countIteration % 50 == 0)
						std::cout << "Tracker :: " << time_iteration << " microseconds and " << frame_delete << " frames will be deleted" << std::endl;
				
					countIteration++;
					time_last_process = std::chrono::high_resolution_clock::now();
                }
            }
        }
    }

    if (countIteration > 0) 
		std::cout << "Tracking process speed :: " << static_cast<int>(countIteration / t_elapsed * 1000000) << " Hz for" << countIteration << "cycles" << std::endl;
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
		else if (_mode_tracking == 1)//Template matching
		{
			success_tracking = track_template_matching(croppedImg, tracker.template_matching, bbox_in_search);
			tracker.bbox = bbox_in_search;//eve
		}
		else if (_mode_tracking == 0)//MOSSE
		{
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

		//add success flags.
		success_flags.push_back(success_tracking);
	}
	return success_flags;
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