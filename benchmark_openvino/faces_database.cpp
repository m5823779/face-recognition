
#include "faces_database.hpp"


namespace {
	constexpr size_t ndetections = 200;
}  // namespace

/*
Identity::Identity(
	string label,
	string descriptors,
	string company,
	string position) {
	this->label = label;
	this->descriptors = descriptors;
	this->company = company;
	this->position = position;
}


//TODO destructor of identity

//TODO should the distance * 0.5 which is done in python demo code?
float Identity::cosine_dist(const vector<float>& x, const vector<float>& y) {
	if (x.size() != y.size()) {
		throw "Vector sizes don't match!";
	}

	float dot = 0.0, denomX = 0.0, denomY = 0.0;
	for (vector<float>::size_type i = 0; i < x.size(); ++i) {
		dot += x[i] * y[i];
		denomX += x[i] * x[i];
		denomY += y[i] * y[i];
	}
	return static_cast<float>(dot / (sqrt(denomX) * sqrt(denomY) + 1e-6));
}

FacesDatabase::FacesDatabase(string path, string face_identifier, string landmarks_detector) {
	filesystem::path abspath(path);
	abspath = filesystem::absolute(abspath);
	vector<filesystem::path> paths;
	if (filesystem::is_directory(abspath)) {
		for (auto const& dir_entry : filesystem::directory_iterator{abspath}) {
			paths.push_back(dir_entry.path().filename());
		}
	}
	else {
		cerr << "Wrong face images database path. Expected a path to the directory containing" 
			<< IMAGE_EXTENSIONS[0] << "," << IMAGE_EXTENSIONS[1] 
			<< " files, but got " << abspath.extension().string() << "." << endl;
		exit(1);
	}
	if (paths.size() == 0) {
		cerr << "The directory has no images." << endl;
		exit(1);
	}

	for (auto& path : paths) {
		string label = path.stem().string();
		Mat image = cv::imread(path.string(),cv::IMREAD_COLOR);

		Mat orig_image = image.clone();
		// TODO: if face_dectevtor, crop raw images into appropriate face identity images
		const int org_h = orig_image.rows;
		const int org_w = orig_image.cols;
		// TODO!!: facedetector.result([0,0,0,0,0,w,h))
		// rois = 

	}




}


void FaceDetection::fetchResults(std::vector<Result>& results, InferEngine* ai_core) {
	float* detections = ai_core->Get_fd_OutputBlob();

	for (size_t i = 0; i < ndetections; i++)
	{
		float image_id = detections[i * objectSize];
		if (image_id < 0)
			break;
		Result r;
		r.label = static_cast<int>(detections[i * objectSize + 1]);
		r.confidence = detections[i*objectSize + 2];

		if (r.confidence <= detectionThreshold)
			continue;

		r.location.x = static_cast<int>(detections[i * objectSize + 3] * width);
		r.location.y = static_cast<int>(detections[i * objectSize + 4] * height);
		r.location.width = static_cast<int>(detections[i * objectSize + 5] * width - r.location.x);
		r.location.height = static_cast<int>(detections[i * objectSize + 6] * height - r.location.y);


		// TODO: modify the bounding box size  
		// https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/interactive_face_detection_demo/cpp/detectors.cpp#L158
		
		if (r.confidence > detectionThreshold) {
			results.push_back(r);
		}

	}
	cout << "inside: " <<results.size() << endl;

}
*/

Mat crop(Mat& frame, cv::Rect location) {
	auto ret = frame(location);
	return ret.clone();
}

vector<Mat> cut_rois(Mat& frame, vector<cv::Rect>& locations) {
	vector<Mat> cutted_rois;
	for (auto roi : locations) {
		cutted_rois.push_back(frame(roi));
	}
	return cutted_rois;
}


