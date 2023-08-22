#pragma once
#include "pch.h"

class FaceDetector : public InferEngine{
public:
	class Result {
	public:
		Result(vector<float>& tmp);

		int label; // predicted class ID (1-face)
		float confidence; // confidence for the predict class
		float x, y, w, h; // since the output of the model is ratio
		cv::Rect location; // (x,y,w,h)

		void rescale_roi(float roi_scale_factor);
		void resize_roi(int frame_width, int frame_height);
		void clip(float width, float height);
		void set_rect();
	};

public:
	FaceDetector(
		InferenceEngine::Core& core,
		std::string model_path,
		std::string cache_path,
		std::string infer_device,
		float prob_threshold,
		float roi_scale_factor);
	void start_sync(Mat& blob_image);
	void fetchresults(vector<FaceDetector::Result>& results);

private:
	int OUTPUT_SIZE = 7;

	float confidence_threshold;
	float roi_scale_factor;

	float width;
	float height;

	size_t model_input_n;
	size_t model_input_c;
	size_t model_input_w;
	size_t model_input_h;
	size_t model_input_size;

	cv::Size input_size;

private:
	void preprocess(Mat& frame, std::vector<FaceDetector::Result>& rois);
	void preprocess(Mat& frame);
};
