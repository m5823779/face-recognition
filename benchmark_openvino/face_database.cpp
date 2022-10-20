#include "face_database.hpp"

FacesDatabase::FacesDatabase(std::string raw_path, FaceIdentifier& reid) {
	auto path_fs = std::filesystem::path(raw_path);
	
	std::vector<std::filesystem::path> paths;
	if (std::filesystem::is_directory(path_fs)) {
		for (const auto& file : std::filesystem::directory_iterator(path_fs)) {
			// found
			if (std::find(this->IMAGE_EXTENSIONS.begin(), this->IMAGE_EXTENSIONS.end(), file.path().extension().string()) != this->IMAGE_EXTENSIONS.end()) {
				paths.push_back(file);
			}
		}
	}
	else throw "Wrong face images database path. not a directory.";

	if (paths.size() == 0) throw "The images database folder has no images.";

	this->database = std::vector<Identity>();
	for (size_t i = 0; i < paths.size(); i++) {
		std::string label = paths[i].stem().string();

		cv::Mat image = cv::imread(paths[i].string(), cv::IMREAD_COLOR);
		auto orig_img = image.clone();

		size_t w = image.size().width, h = image.size().height;
		vector<float> base_output{ 0.f,0.f,0.f,0.f,0.f,static_cast<float>(w),static_cast<float>(h)};
		auto roi = FaceDetector::Result(base_output);
		vector<FaceDetector::Result> rois;
		rois.push_back(roi);

		vector<vector<float>> landmark;
		vector<vector<float>> results;
		reid.start_sync(image, rois, landmark, results);

		auto descriptor = results[0];
		this->add_item(descriptor, label);
	}
}

vector<pair<int, float>> FacesDatabase::match_faces(std::vector<float>& desc) {
	auto database = this->database;
	vector <pair<int, float>> matches;

	float min_dist = 10;
	int best_match_id;

	for (size_t i = 0; i < database.size(); i++) {
		if (cosine_dist(desc, database[i].descriptor) < min_dist) {
			best_match_id = i;
			min_dist = cosine_dist(desc, database[i].descriptor);
		}
	}
	matches.push_back(pair<int, float>(best_match_id, min_dist));
	return matches;
}


int FacesDatabase::chech_if_face_exist(std::vector<float> desc, float threshold) {
	int match_id = -1;
	float min_dist = 10;
	for (size_t i = 0; i < this->database.size(); i++)
	{
		if (cosine_dist(desc, this->database[i].descriptor) < min(min_dist,threshold)) {
			match_id = i;
			min_dist = cosine_dist(desc, database[i].descriptor);
		}
	}
	return match_id;
}

int FacesDatabase::check_if_label_exists(std::string label) {
	int match_id = -1;
	for (size_t i = 0; i < this->database.size(); i++)
	{
		if (this->database[i].label == label) {
			match_id = i;
		}
	}
	return match_id;
}

int FacesDatabase::add_item(std::vector<float> desc, std::string label) {
	int match = -1;
	match = this->check_if_label_exists(label);

	this->database.push_back(FacesDatabase::Identity(label, desc, "Company: ", "Title: "));
	return match;
}

