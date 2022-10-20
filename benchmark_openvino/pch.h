#pragma once
#define Debug
#include <vector>
#include <string>
#include <iostream>
#include <Windows.h>
#include <filesystem>

#undef min
#undef max
#include <ie/inference_engine.hpp>
#include <opencv2/opencv.hpp>

#include <Eigen/SVD>  
#include <Eigen//Dense> 

#include "infer_engine.h"

using namespace cv;
using namespace std;
using namespace Eigen;
using namespace InferenceEngine;
