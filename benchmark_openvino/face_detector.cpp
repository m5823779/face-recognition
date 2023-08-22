#include "face_detector.hpp"

namespace { constexpr size_t ndetections = 200; }

FaceDetector::Result::Result(vector<float>& outputs) {
	this->label = static_cast<int>(outputs[1]);
	this->confidence = outputs[2];
	this->x = outputs[3];
	this->y = outputs[4];
	this->w = outputs[5];
	this->h = outputs[6];
	this->set_rect();
}

void FaceDetector::Result::rescale_roi(float roi_scale_factor) {
	this->x -= (this->w * 0.5 * (roi_scale_factor - 1.0));
	this->y -= (this->h * 0.5 * (roi_scale_factor - 1.0));
	this->w = (this->w * roi_scale_factor);
	this->h = (this->h * roi_scale_factor);
}

void FaceDetector::Result::resize_roi(int frame_width, int frame_height) {
	this->x *= frame_width;
	this->y *= frame_height;
	this->w = (this->w * frame_width) - this->x;
	this->h = (this->h * frame_height) - this->y;
}

void FaceDetector::Result::clip(float width, float height) {
	this->x = std::clamp(this->x, 0.f, width);
	this->y = std::clamp(this->y, 0.f, height);
	this->w = std::clamp(this->w, 0.f, width);
	this->h = std::clamp(this->h, 0.f, height);
}

void FaceDetector::Result::set_rect() {
	this->location.x = static_cast<int>(this->x);
	this->location.y = static_cast<int>(this->y);
	this->location.width = static_cast<int>(this->w);
	this->location.height = static_cast<int>(this->h);
}

// constructor
FaceDetector::FaceDetector(
	InferenceEngine::Core& core,
	std::string model_path,
	std::string cache_path,
	std::string infer_device,
	float prob_threshold, 
	float roi_scale_factor) {

	this->Loadmodel(core, model_path, cache_path, infer_device, "face detection");

	// input_shape = (n,c,h,w)
	this->model_input_n = this->input_shape[0];
	this->model_input_c = this->input_shape[1];
	this->model_input_h = this->input_shape[2];
	this->model_input_w = this->input_shape[3];
	this->model_input_size = this->model_input_c * this->model_input_w * this->model_input_h;

	this->height = this->width = 0.;

	if (this->output_shape.size() != 4 || this->output_shape[3] != 7)
		throw "The model expects output shape with 7 outputs";

	this->confidence_threshold = prob_threshold;
	this->roi_scale_factor = roi_scale_factor;
}

void FaceDetector::preprocess(Mat& blob_image) {
	this->input_size = blob_image.size();
	
	// resize (1 x 3 x 384 x 672)
	cv::resize(blob_image, blob_image, cv::Size(this->model_input_w, this->model_input_h));
	blob_image.convertTo(blob_image, CV_32F);

	// opencv mat to array (HWC -> NCHW)
	float* pixels = (float*)(blob_image.data);
	float* input_tensor = this->Get_InputBlob();
	for (int i = 0; i < this->model_input_size; i+=this->model_input_c) {
		UINT32 index = i / this->model_input_c;
		for (int j = 0; j < this->model_input_c; ++j) {
			input_tensor[(this->model_input_w * this->model_input_h * j) + index] = pixels[i + j];
		}
	}
}

void FaceDetector::start_sync(Mat& blob_image) {
	this->preprocess(blob_image);
	this->infer();
}

void FaceDetector::fetchresults(vector<FaceDetector::Result>& results) {
	// get output
	float* output_tensor = this->Get_OutputBlob();

	for (size_t i = 0; i < ndetections; i++) {
		float* output = output_tensor + i * this->OUTPUT_SIZE;
		vector<float> tmp(output, output + 7);
		auto r = FaceDetector::Result(tmp);

		if (r.confidence < this->confidence_threshold) 
			break;
		else {
			// postprocessing
			r.resize_roi(this->input_size.width, this->input_size.height);
			r.rescale_roi(this->roi_scale_factor);
			r.clip(static_cast<float>(this->input_size.width), static_cast<float>(this->input_size.height));
			r.set_rect();
			results.push_back(r);
		}
	}
}