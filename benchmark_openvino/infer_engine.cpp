#define TIMER   
#include "infer_engine.h"

HRESULT InferEngine::Loadmodel(
    InferenceEngine::Core& core,
    std::string model_path,
    std::string cache_path,
    std::string infer_device,
    std::string model_type) {

    std::string bin_path = model_path.substr(0, model_path.find_last_of('.')) + ".bin";
    //std::string fd_blob_path = fd_model_path.substr(0, fd_model_path.find_last_of('.')) + ".blob";

    if (!CheckFileExist(model_path) || !CheckFileExist(bin_path)) {
        wchar_t msg[128];
        swprintf_s(msg, L"%s not exist ...\n", model_path);
        OutputDebugStringW(msg);
        return EXIT_FAILURE;
    }

    this->model_type = model_type;

    // output message
    wchar_t msg[128];

    try {
        // step 1. initialize inference engine core
        this->ie = &core;

        // saving cache file to speed up model loading
        if (!cache_path.empty()) {
            ie->SetConfig({ {CONFIG_KEY(CACHE_DIR), cache_path} });
            std::cout << "Saving cache file to " << cache_path.c_str() << "..." << std::endl;
        }

        // step 2. load IR / ONNX model
        CNNNetwork network = ie->ReadNetwork(model_path, bin_path);
        std::cout << "Loading " << this->model_type << " model from " << model_path << " ..." << std::endl;

        // step 3. get input & output format (allow to set input precision & get input / output layer name)
        InferenceEngine::InputsDataMap inputs_info = network.getInputsInfo();
        InferenceEngine::OutputsDataMap outputs_info = network.getOutputsInfo();

        auto input_shapes = network.getInputShapes();

        std::tie(this->input_name, this->input_shape) = *input_shapes.begin(); // let's consider first input only

        inputs_info.begin()->second->setLayout(Layout::NCHW);
        inputs_info.begin()->second->setPrecision(Precision::FP32);
        inputs_info.begin()->second->getPreProcess().setColorFormat(ColorFormat::RGB);

        this->output_name = outputs_info.begin()->first;
        outputs_info.begin()->second->setPrecision(Precision::FP32);

        // step 4. loading a model to the device
        this->executable_network = ie->LoadNetwork(network, infer_device);
        cout << "Loading " << model_type << " model to device: " << infer_device << " ..." << std::endl;

        // step 5. create an infer request
        this->infer_request = this->executable_network.CreateInferRequest();

        this->input_blob = this->infer_request.GetBlob(this->input_name);
        this->input_shape = this->input_blob->getTensorDesc().getDims();

        this->output_blob = this->infer_request.GetBlob(this->output_name);
        this->output_shape = this->output_blob->getTensorDesc().getDims();
        printf("Input shape: [ %d x %d x %d x %d ]\n", this->input_shape[0], this->input_shape[1], this->input_shape[2], this->input_shape[3]);
    }
    catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
}

InferEngine::~InferEngine() {

    this->infer_request.Cancel();
    this->input_blob.reset();
    this->output_blob.reset();
}

float* InferEngine::Get_InputBlob() {
    float* input_tensor = static_cast<float*>(this->input_blob->buffer());
    return input_tensor;

}

float* InferEngine::Get_OutputBlob() {
    return static_cast<PrecisionTrait<Precision::FP32>::value_type*>(this->output_blob->buffer());
}

void InferEngine::infer() {

#ifdef TIMER
    std::chrono::steady_clock::time_point start_time = std::chrono::high_resolution_clock::now();
#endif

    infer_request.Infer();

#ifdef TIMER
    std::chrono::steady_clock::time_point end_time = std::chrono::high_resolution_clock::now();
    float time_count = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(end_time - start_time).count();
    wchar_t msg[128];
    std::printf("image processing time : %d (ms)\n", (int)(time_count));
#endif
}

bool InferEngine::CheckFileExist(const std::string& file_path) {
    struct stat buffer;
    return (stat(file_path.c_str(), &buffer) == 0);
}




