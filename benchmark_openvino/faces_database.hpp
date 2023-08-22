#pragma once
#include <Windows.h>
#include <chrono>
#include <queue>
#include <vector>
#include <cmath>
#undef min
#undef max
#include <ie/inference_engine.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <filesystem>
#include "infer_engine.h"
#include "faces_database.hpp"

using namespace cv;
using namespace InferenceEngine;
using namespace std;

const string IMAGE_EXTENSIONS[2] = { "jpg", "png" };

class Identity {
public:
	Identity(
		string label,
		float descriptors[256],
		string company,
		string position);
	inline float cosine_dist(const float x[256], const float y[256]);

	string label;
	float descriptors[256];
	string company;
	string position;
};


class FacesDatabase {
public:
	FacesDatabase();
	FacesDatabase(string path, string face_identifier, string landmarks_detector); // TODO: Change to identifier and landmarks class
	int* match_faces(float descriptors[256]); // TODO: change the descriptors type and the return type
	bool check_if_face_exist(string desc, double threshold); // TODO: change the desc type to the correct one
	int check_if_label_exists(string label); // return parsed label and the id of the identity, -1 for no match. TODO: modify the return type
	int add_item(string desc, string label); // add a identity to the database, return parsed label and the id of the identity
	Identity get_item(int idx); // getter of the identity
	int length(); // return the length of the database




private:
	filesystem::path fg_path; // facedb_dir
	//bool no_show;
	vector<string> paths;
	vector<Identity> database;


};





class FaceDetection {
public:
	struct Result {
		int label;
		float confidence;
		cv::Rect location;
	};

	std::string output;
	std::string labels_output;
	double detectionThreshold = 0.3;
	int objectSize = 7;
	float width;
	float height;
	size_t model_input_width;
	size_t model_input_height;
	void fetchResults(std::vector<Result>& results, InferEngine* ai_core);

};


Mat crop(Mat& frame, cv::Rect location);

vector<Mat> cut_rois(Mat& frame, vector<cv::Rect>& locations);