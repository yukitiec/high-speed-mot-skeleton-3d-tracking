#pragma once

#ifndef TEMPLATE_MATCHING_H
#define TEMPLATE_MATCHING_H

#include "../stdafx.h"

class TemplateMatching
{
private:
    
    int MATCHINGMETHOD = cv::TM_SQDIFF_NORMED;
	double _scoreThreshold = 0.7;
	//cv::TM_SQDIFF_NORMED -> unique background
	//cv::TM_CCOEFF_NORMED :: Correlation coefficient, cv::TM_CCORR_NORMED -> patterned background 
	// TM_SQDIFF_NORMED is good for small template
    const double MoveThreshold = 0.0;                 // move threshold of objects

	bool _bool_skip = true; //skip updating for occlusion and switching prevention
    double _K_SIGMA = 2.0;//threshold for skipping condition. MU-_K_SIGMA*std region.
    double _N_WARMUP = 10;
	int _MAX_SKIP = 10;
public:
	cv::Mat _templateImg;//save template image internally.
	cv::Point3d _scores = cv::Point3d(0.0,0.0,0.0);//mean,std,N_samples

	/// Constructor
	TemplateMatching(double scoreThreshold=0.7, double K_SIGMA=1.0, double N_WARMUP=10,int MAX_SKIP=10,std::string method="sqdiff",bool bool_skip=false) 
	: _scoreThreshold(scoreThreshold), _K_SIGMA(K_SIGMA), _N_WARMUP(N_WARMUP), _MAX_SKIP(MAX_SKIP), _bool_skip(bool_skip) 
	{

		if(method == "sqdiff")
		{
			MATCHINGMETHOD = cv::TM_SQDIFF_NORMED;//cv::TM_SQDIFF_NORMED -> unique background
		}
		else if(method == "ccoeff")
		{
			MATCHINGMETHOD = cv::TM_CCOEFF_NORMED;//cv::TM_CCOEFF_NORMED :: Correlation Coefficient->patterned background 
		}
		else if(method == "ccorr")
		{
			MATCHINGMETHOD = cv::TM_CCORR_NORMED;//cv::TM_CCORR_NORMED -> Cross Correlation -> unique background
		}
	};
	/// Destructor
	~TemplateMatching() {};

	/**
	* @brief Initialize tracking window
	* @param[in] image Source image
	* @param[in] boundingBox Bounding box of target object. Window size depends on this bounding box size.
	* @return True
	*/
	bool init(const cv::Mat& image, cv::Rect2d& boundingBox);

	/**
	* @brief Update Correlation filter.
	* @param[in] image Source image
	* @param[in] boundingBox Bounding box of target object. Window size depends on this bounding box size.
	* @param[in] previous_move Previous move
	* @param[in] transport If true, tracker seraches around the "boundingBox" argument, otherwise searches around the previous bounding box.
	* @param[in] psrThreshold If PSR is smaller than this, do not update filter
	* @return PSR value. If PSR is very small, it failed detection.
	*/
	bool update(const cv::Mat& image, cv::Rect2d& boundingBox);

	/**
	* @brief Make TrackerMOSSE object
	* @return A shared pointer of TrackerMOSSE object
	*/
	static cv::Ptr<TemplateMatching> create() { return cv::makePtr<TemplateMatching>(); };

};

#endif