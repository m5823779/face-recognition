#include "landmarks_detector.hpp"

LandmarksDetector::LandmarksDetector(
	InferenceEngine::Core& core,
	std::string model_path,
	std::string cache_path,
	std::string infer_device) {

	this->Loadmodel(core, model_path, cache_path, infer_device, "landmarks detection");

	this->model_input_n = this->input_shape[0];
	this->model_input_c = this->input_shape[1];
	this->model_input_h = this->input_shape[2];
	this->model_input_w = this->input_shape[3];
	this->model_input_size = this->model_input_c * this->model_input_w * this->model_input_h;

	this->height = this->width = 0.;
	if (this->output_shape.size() != 4 || this->output_shape[1] != this->POINTS_NUMBER * 2)
		throw "The model expects output shape with 10 outputs";
}

void LandmarksDetector::preprocess(Mat& frame, std::vector<FaceDetector::Result>& rois) {
	this->inputs.clear();
	Mat img = frame.clone();
	std::vector<Mat> cropped_frames;
	cut_rois(img, cropped_frames, rois);

	// resize (1 x 3 x 48 x 48)
	resize_input(cropped_frames, this->input_shape);

	for (size_t i = 0; i < cropped_frames.size(); ++i) {
		cropped_frames[i].convertTo(cropped_frames[i], CV_32F);

		// opencv mat to array (HWC -> NCHW)
		float* pixels = (float*)(cropped_frames[i].data);
		auto input_vec = std::vector<float>(this->model_input_size);
		for (int i = 0; i < this->model_input_size; i += this->model_input_c) {
			UINT32 index = i / this->model_input_c;
			for (int j = 0; j < this->model_input_c; ++j) {
				input_vec[(this->model_input_w * this->model_input_h * j) + index] = pixels[i + j];
			}
		}
		this->inputs.push_back(std::move(input_vec));
	}
}

void LandmarksDetector::start_sync(Mat& frame, std::vector<FaceDetector::Result>& rois, vector<vector<float>>& results) {
	this->preprocess(frame, rois);
	for (size_t i = 0; i < this->inputs.size(); i++) {
		auto input_tensor = this->Get_InputBlob();

		for (size_t j = 0; j < this->model_input_size; j++) {
			input_tensor[j] = this->inputs[i][j];
		}

		// inference
		this->infer();

		// get output
		auto output_tensor = this->Get_OutputBlob();
		vector<float> output_vector(output_tensor, output_tensor + 2 * this->POINTS_NUMBER);
		results.push_back(output_vector);
	}
}