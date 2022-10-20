#pragma once
#include "pch.h"
#include "util.hpp"
#include "face_detector.hpp"

class LandmarksDetector : InferEngine {
public:
	LandmarksDetector(
		InferenceEngine::Core& core,
		std::string model_path,
		std::string cache_path,
		std::string infer_device);

	void start_sync(Mat& frame, std::vector<FaceDetector::Result>& rois, vector<vector<float>>& results);	
	std::vector<std::vector<float>> inputs;

private:
	void preprocess(Mat& frame, std::vector<FaceDetector::Result>& rois);

private:
	int POINTS_NUMBER = 5;

	float width;
	float height;

	size_t model_input_n;
	size_t model_input_c;
	size_t model_input_w;
	size_t model_input_h;
	size_t model_input_size;

	cv::Size input_size;
};
