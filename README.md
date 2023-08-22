# Face recognition

![image](./demo.gif)

### reference 
https://github.com/openvinotoolkit/open_model_zoo/tree/master/demos/face_recognition_demo/python

### depndency
`pip install openvino-dev==2022.1.0`

### model
* Face detection: [face-detection-adas-0001](https://docs.openvino.ai/2019_R1/_face_detection_adas_0001_description_face_detection_adas_0001.html)

    input: 1x3x384x672 BGR image 

    output: 1x1xNx7, where N is the numver of detected face, 7 contain [image_id, label, confidence, x, y, w, h]

* Landmark detection: [landmarks-regression-retail-0009](https://docs.openvino.ai/2019_R1/_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)

    input: 1x3x48x48 BGR image

    output: 1x10, (x0, y0, x1, y1, ..., x5, y5)

* Face reidentification: [face-reidentification-retail-0095](https://docs.openvino.ai/2019_R1/_face_reidentification_retail_0095_description_face_reidentification_retail_0095.html)

    input: 1x3x128x128 BGR image

    output: 1x256x1x1, containing a row-vector of 256 floating point values

### setup
```
git clone git clone https://github.com/openvinotoolkit/open_model_zoo.git
pip install .\demos\common\python\
cd <omz_dir>/demos/face_recognition_demo/python/
omz_downloader --list models.lst
omz_converter --list models.lst
mkdir face_gallery
```

### prepare
place face image into `face_gallery` folder

```
python face_recognition_demo.py -i 0 -m_fd .\intel\face-detection-adas-0001\FP16-INT8\face-detection-adas-0001.xml -m_lm .\intel\landmarks-regression-retail-0009\FP16-INT8\landmarks-regression-retail-0009.xml -m_reid .\intel\face-reidentification-retail-0095\FP16-INT8\face-reidentification-retail-0095.xml --verbose -fg .\face_gallery\ --run_detector
```

enter name. press `enter` to save or `esc` to exit

### Usage (python)
```
python face_recognition_demo.py -i 0 -m_fd .\intel\face-detection-adas-0001\FP16-INT8\face-detection-adas-0001.xml -m_lm .\intel\landmarks-regression-retail-0009\FP16-INT8\landmarks-regression-retail-0009.xml -m_reid .\intel\face-reidentification-retail-0095\FP16-INT8\face-reidentification-retail-0095.xml --verbose -fg .\face_gallery\ -d_fd GPU -d_lm GPU -d_reid GPU
```

### Usage (c++)
1. place cropped face image into `database` folder
2. build solution and run

