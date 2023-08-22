#pragma once
#include "pch.h"
#include "util.hpp"
#include "face_detector.hpp"

class FaceIdentifier : public InferEngine {
public:
    class Result {
    public:
        Result(int id, float distance, vector<float>& desc) {
            this->id = id;
            this->distance = distance;
            for (size_t i = 0; i < this->descriptor.size(); ++i) {
                this->descriptor[i] = desc[i];
            }
        };
        Result(std::string unknown) {
            this->id = -1;
            this->distance = 1;
        }
        int id;
        float distance;
        std::vector<float> descriptor = std::vector<float>(256); // row vector
    };

public:
    FaceIdentifier(
        InferenceEngine::Core& core,
        std::string model_path,
        std::string cache_path,
        std::string infer_device,
        float match_threshold);
    void preprocess(Mat& frame, vector<FaceDetector::Result>& rois, vector<vector<float>>& landmarks);
    void start_sync(Mat& frame, std::vector<FaceDetector::Result>& rois, vector<vector<float>>& landmarks, vector<vector<float>>& results);
    float get_threshold();

private:
    void align_rois(vector<Mat>& face_images, vector<vector<float>>& face_landmarks);
    std::vector<std::vector<float>> get_transform(vector<float> desired_landmarks, vector<float>& original_landmarks);

private:
    std::vector<std::vector<float>> inputs;

    size_t model_input_n;
    size_t model_input_c;
    size_t model_input_h;
    size_t model_input_w;
    size_t model_input_size;
    size_t model_output_size;

    float match_threshold;

    std::vector<float> REFENCE_LANDMARKS = { 
        30.2946 / 96.0, 51.6963 / 112.0, // left eye
        65.5318 / 96.0, 51.5014 / 112.0, // right eye
        48.0252 / 96.0, 71.7366 / 112.0, // nose tip
        33.5493 / 96.0, 92.3655 / 112.0, // left lip corner
        62.7299 / 96.0, 92.2041 / 112.0  // right lip corner
    }; 
};