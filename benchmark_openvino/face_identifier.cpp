#include "face_identifier.hpp"

FaceIdentifier::FaceIdentifier(
    InferenceEngine::Core& core,
    std::string model_path,
    std::string cache_path,
    std::string infer_device,
	float match_threshold) {
	this->Loadmodel(core, model_path, cache_path, infer_device, "face detection");

	// input_shape = (b,c,h,w)
	this->model_input_n = this->input_shape[0];
	this->model_input_c = this->input_shape[1];
	this->model_input_h = this->input_shape[2];
	this->model_input_w = this->input_shape[3];
	this->model_input_size = this->model_input_c * this->model_input_w * this->model_input_h;

	// output =  (1,256,1,1)
	this->model_output_size = this->output_shape[1];

	if (this->output_shape.size() != 4 && this->output_shape.size() != 2)
		throw "The model expects output shape [1, n, 1, 1] or [1, n]";
	this->match_threshold = match_threshold;
}

void FaceIdentifier::preprocess(Mat& frame, std::vector<FaceDetector::Result>& rois, std::vector<std::vector<float>>& landmarks) {
	this->inputs.clear();
	Mat img = frame.clone();
	std::vector<Mat> cropped_frames;
	cut_rois(img, cropped_frames, rois);
	if (!landmarks.empty())
		align_rois(cropped_frames, landmarks);

	// resize (1 x 3 x 128 x 128)
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

void FaceIdentifier::start_sync(Mat& frame, std::vector<FaceDetector::Result>& rois, vector<vector<float>>& landmarks,vector<vector<float>>& results) {
	this->preprocess(frame, rois, landmarks);
	for (size_t i = 0; i < this->inputs.size(); i++) {
		auto input_tensor = this->Get_InputBlob();
		for (size_t j = 0; j < this->model_input_size; j++) {
			input_tensor[j] = this->inputs[i][j];
		}

		// inference
		this->infer();

		// get output
		auto output_tensor = this->Get_OutputBlob();
		vector<float> output_vector(output_tensor, output_tensor + 256);
		results.push_back(output_vector);
	}
}

float FaceIdentifier::get_threshold() { return this->match_threshold; }

void FaceIdentifier::align_rois(vector<Mat>& face_images, vector<vector<float>>& face_landmarks) {
	if (face_images.size() != face_landmarks.size()) throw "Input lengths differ";
	for (size_t i = 0; i < face_images.size(); i++) {
		auto img = face_images[i];
		auto face_landmark = face_landmarks[i];
		auto scale = img.size();  // width, height

		// scale up
		vector<float> desired_landmark = this->REFENCE_LANDMARKS;
		for (size_t j = 0; j < this->REFENCE_LANDMARKS.size(); j+=2) {
			desired_landmark[j] = this->REFENCE_LANDMARKS[j] * (float)scale.width;
			desired_landmark[j + 1] = this->REFENCE_LANDMARKS[j+1] * (float)scale.height;
			face_landmark[j] = face_landmark[j] * (float)scale.width;;
			face_landmark[j + 1] = face_landmark[j + 1] * (float)scale.height;
		}

		cv::Mat dst;
		// get transform matrix
		auto transform = get_transform(desired_landmark, face_landmark);
		Mat tran(2, 3, CV_32FC1);
		for (int i = 0; i < transform.size(); ++i) {
			for (int j = 0; j < transform[i].size(); ++j) {
				tran.at<float>(i, j) = transform[i][j];
			}
		}

		cv::warpAffine(img, img, tran, scale, WARP_INVERSE_MAP);

#ifdef Debug
		for (int l = 0; l < face_landmarks[i].size(); l += 2) {
			cv::circle(img, Point(desired_landmark[l], desired_landmark[l + 1]), 1, cv::Scalar(0, 220, 0), 2);
		}
		cv::imshow("face alignment for debug", img);
		cv::waitKey(1);
#endif
	}
}

std::vector<std::vector<float>> FaceIdentifier::get_transform(vector<float> src, vector<float>& dst) {
	float* tmp_src = &src[0];
	float* tmp_dst = &dst[0];

	vector<vector<float>> src_data(5, vector<float>(2, 0));
	vector<vector<float>> dst_data(5, vector<float>(2, 0));

	for (int row = 0; row < src_data.size(); ++row) {
		for (int col = 0; col < src_data[row].size(); ++col) {
			src_data[row][col] = *(tmp_src++);
			dst_data[row][col] = *(tmp_dst++);
		}
	}

	// calcuate mean
	vector<float> mean_src(src_data[0].size(), 0);
	vector<float> mean_dst(dst_data[0].size(), 0);

	for (int row = 0; row < src_data.size(); ++row) {
		for (int col = 0; col < src_data[row].size(); ++col) {
			mean_src[col] += src_data[row][col];
			mean_dst[col] += dst_data[row][col];
		}
	}

	int num = src_data.size();
	transform(mean_src.begin(), mean_src.end(), mean_src.begin(), [num](float& c) { return c / (float)num; });
	transform(mean_dst.begin(), mean_dst.end(), mean_dst.begin(), [num](float& c) { return c / (float)num; });

	// calcuate std
	float std_src = 0.;
	float std_dst = 0.;

	int total_element = 0;
	for (int row = 0; row < src_data.size(); ++row) {
		for (int col = 0; col < src_data[row].size(); ++col) {
			std_src += pow((src_data[row][col] - mean_src[col]), 2);
			std_dst += pow((dst_data[row][col] - mean_dst[col]), 2);
			total_element++;
		}
	}

	std_src = sqrt(std_src / total_element);
	std_dst = sqrt(std_dst / total_element);

	// normalize
	for (int row = 0; row < src_data.size(); ++row) {
		for (int col = 0; col < src_data[row].size(); ++col) {
			src_data[row][col] = (src_data[row][col] - mean_src[col]) / std_src;
			dst_data[row][col] = (dst_data[row][col] - mean_dst[col]) / std_dst;
		}
	}

	vector<vector<float>> src_tran = transpose(src_data);
	vector<vector<float>> mm = mutil(src_tran, dst_data);

	MatrixXf C;
	C.setRandom(mm.size(), mm[0].size());
	for (int row = 0; row < mm.size(); ++row) {
		for (int col = 0; col < mm[row].size(); ++col) {
			C(row, col) = mm[row][col];
		}
	}
	JacobiSVD<Eigen::MatrixXf> svd(C, ComputeThinU | ComputeThinV);

	vector<vector<float>> u(svd.matrixU().rows(), vector<float>(svd.matrixU().cols(), 0));
	vector<vector<float>> v(svd.matrixV().rows(), vector<float>(svd.matrixV().cols(), 0));

	for (int row = 0; row < u.size(); ++row) {
		for (int col = 0; col < u[row].size(); ++col) {
			u[row][col] = svd.matrixU()(row, col);
			v[row][col] = svd.matrixV()(row, col);
		}
	}

	vector<vector<float>> r = transpose(mutil(u, v));

	vector<vector<float>> transform(2, vector<float>(3, 0));
	for (int row = 0; row < r.size(); ++row) {
		for (int col = 0; col < r[row].size(); ++col) {
			transform[row][col] = r[row][col] * (std_dst / std_src);
		}
	}

	std::vector<std::vector<float>> transform_;
	for (int i = 0; i < transform.size(); ++i) {
		transform_.emplace_back(transform[i].begin(), transform[i].begin() + 2);
	}

	for (int i = 0; i < transform.size(); ++i) {
		transform[i][2] = mean_dst[i] - (transform_[i][0] * mean_src[0] + transform_[i][1] * mean_src[1]);
	}
	return transform;
}
