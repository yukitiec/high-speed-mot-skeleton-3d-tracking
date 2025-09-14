#pragma once

/**
 * @file MOSSETracker.h
 * @brief MOSSE tracker class, based on OpenCV library.
 * @author Akira Matsuo (M2 student at GSII)
 * @date 2022-09-12
 */

#include "../stdafx.h"

extern const double threshold_mosse;

namespace cv {
    namespace mytracker {

        /// MOSSE tracker class
        class TrackerMOSSE {
        private:
			double _psrThreshold = 5.7;

            const double eps = 0.00001;  // for normalization
            const double rate = 0.2;     // learning rate
            // const double psrThreshold = 5.7;  // no detection, if PSR is smaller than this
            Point2d center;  // center of the bounding box
            Size size;       // size of the bounding box
            Mat hanWin;
            Mat G;        // goal
            Mat H, A, B;  // state
            Mat divDFTs(const Mat& src1, const Mat& src2) const;
            void preProcess(Mat& window) const;
            double correlate(const Mat& image_sub, Point& delta_xy) const;
            Mat randWarp(const Mat& a) const;

            const double move_threshold = 30;//accelerate abruptly
            const double angle_threshold = 0.5; //change direction abruptly :: 90 degree
            const double delta_psr_threshold = 5; //psr delta threshold
            const double MIN_MOVE_MOSSE = 1; //min move
            const double MIN_PSR_SKIP = 5.0;//threshold_mosse; //for psr skipping condition
            bool _bool_skip = true; //skip updating for occlusion and switching prevention
			double _N_WARMUP = 10;
			double _K_SIGMA = 2.0;//threshold for skipping condition. MU-_K_SIGMA*std region.
			bool _bool_skip = false;
        public:
			Point3d _scores = Point3d(0.0,0.0,0.0);//mean,std,N_samples
            
            /// Constructor
            TrackerMOSSE(double psrThreshold=5.7, double K_SIGMA=1.0, double N_WARMUP=10,bool bool_skip=false) 
			: _psrThreshold(psrThreshold), _K_SIGMA(K_SIGMA), _N_WARMUP(N_WARMUP), _bool_skip(bool_skip) {};
            /// Destructor
            ~TrackerMOSSE() {};

            /**
            * @brief Initialize tracking window
            * @param[in] image Source image
            * @param[in] boundingBox Bounding box of target object. Window size depends on this bounding box size.
            * @return True
            */
            bool init(const Mat& image, Rect2d& boundingBox);

            /**
            * @brief Update Correlation filter.
            * @param[in] image Source image
            * @param[in,out] boundingBox Output Bounding box. If "transport" argument is true, the search position moves to the "boundingBox" position instead of the previous detected point
            * @return Success flag. If PSR is very small, it failed detection.
            */
            bool update(const Mat& image, Rect2d& boundingBox);

            /**
            * @brief Make TrackerMOSSE object
            * @return A shared pointer of TrackerMOSSE object
            */
            static Ptr<TrackerMOSSE> create() { return makePtr<TrackerMOSSE>(); };
        };

    }  // namespace mytracker
}  // namespace cv
