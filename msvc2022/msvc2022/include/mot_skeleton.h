#pragma once

#include "stdafx.h"
#include "yolo_batch.h"
#include "utility.h"
#include "sequence.h"
#include "global_parameters.h"
#include "robot_control.h"

#include "yolo_batch_skeleton.h"
#include "optflow.h"
#include "triangulation_skeleton.h"
#include "buffer_skeleton.h"

class Mot_skeleton {
private:
	Utility ut_mot_skeleton;
public:
	Mot_skeleton() {
		std::cout << "Construct MOT and Skeleton instance." << std::endl;
	};

	void main();
	void yoloDetect();
	void yolo(bool bool_ground_truth);
	void denseOpticalFlow();
};