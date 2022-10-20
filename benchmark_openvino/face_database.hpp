#pragma once
#include <regex>

#include "pch.h"
#include "face_identifier.hpp"

class FacesDatabase {
public:
	std::vector<std::string> IMAGE_EXTENSIONS{ ".jpg", ".png" };
	class Identity {
	public:
		Identity(
			std::string label,
			std::vector<float>& descriptor,
			std::string company,
			std::string postiion) {

			this->label = label;
			this->descriptor = { descriptor.begin(),descriptor.end() };
			this->company = company;
			this->position = position;
		}
		Identity(){};
		std::string label;
		std::vector<float> descriptor;
		std::string company;
		std::string position;
	};

public:
	FacesDatabase(std::string path, FaceIdentifier& reid );
	std::vector<Identity> database;

private:
	std::vector<std::pair<int, float>> match_faces(std::vector<float>& desc);

	int chech_if_face_exist(std::vector<float> desc, float threshold);
	int check_if_label_exists(std::string label);
	int add_item(std::vector<float> desc, std::string label);
};

inline float cosine_dist(std::vector<float>& a, std::vector<float>& b) {
	float dot = 0., denom_a = 0., denom_b = 0.;
	for (size_t i = 0; i < a.size(); i++) {
		dot += a[i] * b[i];
		denom_a += a[i] * a[i];
		denom_b += b[i] * b[i];
	}
	return  0.5* (1 - (dot / (sqrt(denom_a) * sqrt(denom_b))));
}