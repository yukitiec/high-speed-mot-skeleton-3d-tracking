#pragma once

/**
 * @file MOSSETracker.h
 * @brief MOSSE tracker class, based on OpenCV library.
 * @author Akira Matsuo (M2 student at GSII)
 * @date 2022-09-12
 */

#include "stdafx.h"

extern const double threshold_mosse;

namespace cv {
    namespace mytracker {

        /// MOSSE tracker class
        class TrackerMOSSE {
        private:
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
            const int MAX_SKIP = 3; //max number of skip
        public:
            double previous_psr = 0.0;
            double vel_current, vel_previous; //current and previous velocity
            double delta_psr; //change of psr
            int counter_skip = 0; //num of skip

            /// Constructor
            TrackerMOSSE() {};
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
            * @param[in] transport If true, tracker seraches around the "boundingBox" argument, otherwise searches around the previous bounding box.
            * @param[in] psrThreshold If PSR is smaller than this, do not update filter
            * @return PSR value. If PSR is very small, it failed detection.
            */
            double update(const Mat& image, Rect2d& boundingBox, std::vector<int>& previous_move, bool transport = false, bool bool_skip = false, double psrThreshold = 5.7);

            /**
            * @brief Make TrackerMOSSE object
            * @return A shared pointer of TrackerMOSSE object
            */
            static Ptr<TrackerMOSSE> create() { return makePtr<TrackerMOSSE>(); };
        };

    }  // namespace mytracker
}  // namespace cv
