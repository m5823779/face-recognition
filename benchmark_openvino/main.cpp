#include "pch.h"

#include "face_detector.hpp"
#include "landmarks_detector.hpp"
#include "face_identifier.hpp"
#include "face_database.hpp"

using namespace cv;
using namespace std;

// inference device
 //std::string input_source = "0";
std::string input_source = "../test.mp4";
std::string infer_device = "CPU"; // CPU, GPU, VPUX

// model path
std::string fd_model_path = "../weights/face-detection-adas-0001.xml";
std::string reid_model_path = "../weights/face-reidentification-retail-0095.xml";
std::string landmarks_model_path = "../weights/landmarks-regression-retail-0009.xml";

std::string face_gallery = "../database";
std::string cache_path = "../cache";

float roi_scale_factor = 1.15;
float detect_threshold = 0.6;
float match_threshold = 0.3;

Mat frame, blob_image, show;

// entry points
int main(int argc, char* argv[]) {
    // opencv
    cv::VideoCapture cap;
    if (input_source.empty() || input_source == "0") {
        cap = cv::VideoCapture(0);
    }
    else {
        cap = cv::VideoCapture(input_source);
    }

    // openvino inference engine 
    InferenceEngine::Core ai_core;
    auto face_detector = FaceDetector(ai_core, fd_model_path, cache_path, infer_device, detect_threshold, roi_scale_factor);
    auto landmark_detector = LandmarksDetector(ai_core, landmarks_model_path, cache_path, infer_device);
    auto face_reidentifier = FaceIdentifier(ai_core, reid_model_path, cache_path, infer_device, match_threshold);

    // face_gallery
    auto faces_database = FacesDatabase(face_gallery, face_reidentifier);

    while (true) {
        bool ret = cap.read(frame);
        if (!ret) {
            cout << "Can't receive frame (stream end?). Exiting ...\n";
            break;
        }

        blob_image = frame.clone();

        // region of interest
        std::vector<FaceDetector::Result> rois;
        rois.clear();
        face_detector.start_sync(blob_image);
        face_detector.fetchresults(rois);
        printf("Detect '%d' faces ! ", rois.size());

#ifdef Debug
        std::vector<Mat> cropped_frames;
        cut_rois(frame, cropped_frames, rois);
        if (cropped_frames.size() > 0) {
            for (auto cropped_frame : cropped_frames) {
                cv::imshow("face ROI for debug", cropped_frame);
                cv::waitKey(1);
            }
        }
        else cv::destroyWindow("face ROI for debug");
#endif

        if (rois.size() > 0) {
            // landmarks
            std::vector<std::vector<float>> landmarks_results;
            landmark_detector.start_sync(frame, rois, landmarks_results);

            // reidentification
            std::vector<FaceIdentifier::Result> results;
            std::vector<std::vector<float>> output_results;
            blob_image = frame.clone();
            face_reidentifier.start_sync(blob_image, rois, landmarks_results, output_results);

            for (int i = 0; i < output_results.size(); ++i) {
                float min_dist = 10.;
                int match_id = -1.;

                for (int j = 0; j < faces_database.database.size(); ++j) {
                    auto dist = cosine_dist(output_results[i], faces_database.database[j].descriptor);
                    cout << "consine dist: " << dist << " label: " << faces_database.database[j].label << endl;
                    if (dist < min_dist) {
                        match_id = j;
                        min_dist = dist;
                    }
                }

                if (min_dist < face_reidentifier.get_threshold()) {
                    cout << "match_id: " << match_id << ": " << min_dist << endl;
                    FaceIdentifier::Result r(match_id, min_dist, output_results[i]);
                    results.push_back(r);
                }
                else {
                    results.emplace_back(FaceIdentifier::Result("unknown"));
                }
            }

            for (int i = 0; i < results.size(); ++i) {
                if (results[i].id >= 0) 
                    cout << faces_database.database[results[i].id].label << ": " << (1 - results[i].distance) * 100 << '%' << endl;
                else 
                    cout << "unknown" << endl;
            }

            // drawing
            for (size_t i = 0; i < rois.size(); i++) {
                cv::rectangle(frame, rois[i].location, cv::Scalar(0, 220, 0),2);
                if (results[i].id >=0)
                    cv::putText(frame, faces_database.database[results[i].id].label, cv::Point(rois[i].location.x, rois[i].location.y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
                else
                    cv::putText(frame, "Unknown", cv::Point(rois[i].location.x, rois[i].location.y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

                for (int j = 0; j < 5; ++j) {
                    int x = (int)(rois[i].x + (rois[i].w * landmarks_results[i][2 * j]));
                    int y = (int)(rois[i].y + (rois[i].h * landmarks_results[i][(2 * j) + 1]));
                    cv::circle(frame, Point(x, y), 1, cv::Scalar(0, 220, 0), 2);
                }            
            }
        }
        cv::imshow("Output", frame);
        if (cv::waitKey(1) >= 0) {
            break;
        }
    }
    return EXIT_SUCCESS;
}
