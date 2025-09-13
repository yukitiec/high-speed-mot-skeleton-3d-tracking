#include "../../include/tracker/template_matching.h"

bool TemplateMatching::init(const cv::Mat& image, cv::Rect2d& boundingBox)
{
	//Initialize scores.
	_scores = cv::Point3d(0.0,0.0,0.0);

	//Initialize template image. boundingBox is expressed in the image coordinate system.
	// cv::Rect2d is acceptable for ROI extraction as long as its coordinates and size are within image bounds.
	// However, OpenCV's Mat::operator()(Rect) expects cv::Rect (int), so we need to convert.
	cv::Rect roi(
		static_cast<int>(std::round(boundingBox.x)),
		static_cast<int>(std::round(boundingBox.y)),
		static_cast<int>(std::round(boundingBox.width)),
		static_cast<int>(std::round(boundingBox.height))
	);
	// Ensure ROI is within image bounds
	roi = roi & cv::Rect(0, 0, image.cols, image.rows);
	_templateImg = image(roi);

	return true;
}

double TemplateMatching::update(const cv::Mat& image, cv::Rect2d& boundingBox, cv::Point2d& previous_move, bool transport)
{

	//create result image.
	cv::Mat result; // for saving template matching results
	int width_image = image.cols;
	int height_image = image.rows;
	int width_template = _templateImg.cols;
	int height_template = _templateImg.rows;
	int result_cols = width_image - width_template + 1;
	int result_rows = height_image - height_template + 1;
	result.create(result_rows, result_cols, CV_32FC1); // create result array for matching quality+
    //Finish creating result image.
	
	// template Matching
	cv::matchTemplate(image, _templateImg, result, MATCHINGMETHOD); // template Matching
	double minVal;    // minimum score
	double maxVal;    // max score
	cv::Point minLoc; // minimum score left-top points
	cv::Point maxLoc; // max score left-top points
	cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat()); // In C++, we should prepare type-defined box for returns, which is usually pointer
	//Finish finding matching object.
	
	//Extract matching object.
	if ((MATCHINGMETHOD == cv::TM_SQDIFF_NORMED && minVal <= _scoreThreshold) || 
	((MATCHINGMETHOD == cv::TM_CCOEFF_NORMED || MATCHINGMETHOD == cv::TM_CCORR_NORMED) && maxVal >= _scoreThreshold))
	{
		double score;
		if (MATCHINGMETHOD == cv::TM_SQDIFF_NORMED)
			score = minVal;
		else
			score = maxVal;
			
		//Check occlusion.
		//update scores.
		double mu_previous = _scores.x;
		double var_previous = _scores.y;
		_scores.x = (_scores.z*_scores.x+score)/(_scores.z+1);//update mean. 1/(N_samples+1)*(N_samples*mean+PSR)
		_scores.y = (_scores.z*(_scores.y+mu_previous*mu_previous)+score*score)/(_scores.z+1)-_scores.x*_scores.x;//update variance. 1/(N_samples+1)*(N_samples*std+std(PSR-mean))
		_scores.z++;//increment N_samples

		//Check occlusion.
		if (_bool_skip && _scores.z >= _N_WARMUP)//_N_WARMUP consecutive update for checking skip condition
		{
			//check the current PSR is higher than the lower bound of k*std region.
			double lower_bound = _scores.x - std::sqrt(_scores.y)*_K_SIGMA;//mu-k*std
			double upper_bound = _scores.x + std::sqrt(_scores.y)*_K_SIGMA;//mu+k*std
			if ((score >= upper_bound && MATCHINGMETHOD == cv::TM_SQDIFF_NORMED) ||
				(score <= lower_bound && MATCHINGMETHOD == cv::TM_CCOEFF_NORMED || MATCHINGMETHOD == cv::TM_CCORR_NORMED))//PSR is lower than the lower bound.
			{
				std::cout << "TEMPLATE MATCHING  : : skip updating" << std::endl;
				//Reset scores with the previous scores.
				_scores.x = mu_previous;
				_scores.y = var_previous;
				_scores.z -= 1;
				
				if (MATCHINGMETHOD == cv::TM_SQDIFF_NORMED)//diff. lower is better.
					return _scoreThreshold+1.0;//return failure to update.
				else//correlation coefficient or cross correlation. higher is better.
					return _scoreThreshold-1.0;//return failure to update.
			}
		}

		if (MATCHINGMETHOD == cv::TM_SQDIFF_NORMED)
		{
			cv::Point matchLoc = minLoc;
			//Update bounding box.
			boundingBox = cv::Rect2d(minLoc.x, minLoc.y, width_template, height_template);

			//Update template image.
			cv::Rect roi(
				static_cast<int>(std::round(boundingBox.x)),
				static_cast<int>(std::round(boundingBox.y)),
				static_cast<int>(std::round(boundingBox.width)),
				static_cast<int>(std::round(boundingBox.height))
			);
			// Ensure ROI is within image bounds
			roi = roi & cv::Rect(0, 0, image.cols, image.rows);
			_templateImg = image(roi);

			return score;
		}
		else
		{
			cv::Point matchLoc = maxLoc;
			//Update bounding box.
			boundingBox = cv::Rect2d(maxLoc.x, maxLoc.y, width_template, height_template);

			//Update template image.
			cv::Rect roi(
				static_cast<int>(std::round(boundingBox.x)),
				static_cast<int>(std::round(boundingBox.y)),
				static_cast<int>(std::round(boundingBox.width)),
				static_cast<int>(std::round(boundingBox.height))
			);
			// Ensure ROI is within image bounds
			roi = roi & cv::Rect(0, 0, image.cols, image.rows);
			_templateImg = image(roi);

			return score;
		}
	}
}