#pragma once
#include <queue>
#include <chrono>
#include <Windows.h>

#include <opencv2/opencv.hpp>

#undef min
#undef max
#include <ie/inference_engine.hpp>

using namespace std;
using namespace cv;
using namespace InferenceEngine;

class InferEngine {
public:
	InferEngine(){};
	~InferEngine();

	HRESULT Loadmodel(
		InferenceEngine::Core& core,
		std::string model_path,
		std::string cache_path,
		std::string infer_device,
		std::string model_type);

	void infer();

	float* Get_InputBlob(); // copy input buffer to the input tensor
	float* Get_OutputBlob(); 

public:
	InferenceEngine::Core* ie;

	std::string model_path;
	std::string cache_path;
	std::string infer_device;
	std::string model_type;
	std::string input_name;
	std::string output_name;

	InferRequest infer_request;

	ExecutableNetwork executable_network;

	InferenceEngine::SizeVector input_shape;
	InferenceEngine::SizeVector output_shape;

	Blob::Ptr input_blob;
	Blob::Ptr output_blob;

private:
	bool CheckFileExist(const std::string& file_path);
};



