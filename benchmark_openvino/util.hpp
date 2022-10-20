#pragma once
#include "pch.h"
#include "face_detector.hpp"

inline Mat crop(Mat& src, cv::Rect roi) {
	int x1 = roi.x, y1 = roi.y;
	// assertion
	roi.x = std::clamp(x1, 0, src.size().width);
	roi.y = std::clamp(y1, 0, src.size().height);
	roi.width = std::clamp(roi.width, 0, src.size().width - roi.x);
	roi.height = std::clamp(roi.height, 0, src.size().height - roi.y);
	auto  ret = src(roi);
	return ret;
}

inline void cut_rois(Mat frame, vector<Mat>& cropped_frames ,vector<FaceDetector::Result>& rois) {
	for (size_t i = 0; i < rois.size(); ++i) {
		auto ret = crop(frame, rois[i].location);
		cropped_frames.push_back(ret);
	}
}

inline void resize_input(std::vector<Mat>& cropped_frames,std::vector<size_t>& input_shape) {
	size_t w = input_shape[3], h = input_shape[2];
	for (size_t i = 0; i < cropped_frames.size(); ++i) {
		auto tmp = cropped_frames[i].clone();
		cv::resize(tmp, cropped_frames[i], cv::Size(w, h));
	}
}

template <typename T>
inline vector<vector<T>> transpose(vector<vector<T>> src) {
	if (src.size() == 0) return src;
	vector<vector<T> > trans_vec(src[0].size(), vector<T>());
	for (int i = 0; i < src.size(); i++) {
		for (int j = 0; j < src[i].size(); j++) {
			trans_vec[j].push_back(src[i][j]);
		}
	}
	return trans_vec;
}

template <typename T>
inline vector<vector<T>> mutil(vector<vector<T>> m1, vector<vector<T>> m2) {
	int m = m1.size();
	int n = m1[0].size();
	int p = m2[0].size();
	vector<vector<T>> array;
	vector<T> temparay;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < p; j++) {
			T sum = 0;
			for (int k = 0; k < n; k++) sum += m1[i][k] * m2[k][j];
			temparay.push_back(sum);
		}
		array.push_back(temparay);
		temparay.erase(temparay.begin(), temparay.end());
	}
	return array;
}

template <typename T>
inline void print_matrix(vector<vector<T>> src) {
	for (int row = 0; row < src.size(); ++row) {
		for (int col = 0; col < src[row].size(); ++col) {
			cout << src[row][col] << "  ";
		}
		cout << endl;
	}
}


