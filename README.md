# Convert YOLOv9 Model to TensorFlow Lite

This repository offers scripts and instructions for converting a YOLOv9 model to TensorFlow Lite format. TensorFlow Lite is a lightweight solution for deploying machine learning models on mobile and edge devices, making it ideal for applications that require real-time object detection, such as mobile apps or embedded systems.

## New

We now provide the model weights of TFLite (quantized INT8)<br>
<strong font-size=30>Link: [`YOLOv9-e`](https://drive.google.com/file/d/1fWufebI8zSoOdJHG_QA87yV7tKAVJML8/view?usp=drive_link)</strong>
</br>
<strong font-size=30>Link: [`YOLOv9-e-int8`](https://drive.google.com/file/d/1_yya03ufQFANArKSC1xLvR8GLIW20KcQ/view?usp=sharing)</strong>
## Requirements

- Python 3.8.10
- TensorFlow 2.13.1
- Other dependencies (refer to `requirements.txt`)

## Installation

1. Create a Conda environment:

    ```bash
    conda create --name yolo9-tflite python=3.8.10
    ```

2. Activate the environment:

    ```bash
    conda activate yolo9-tflite
    ```

3. Install required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Convert

1. To convert to TFLite, run the provided script:

    ```bash
    convert_tflite.sh
    ```

## Inference
1. I have provided the config to run yolov9 <b>(config/yolov9.yaml)</b>

2. You run to test the model <br>
```bash
python inference.py
```

## Output

![Alt text](assets/output1.jpg)

## Contact

Email : anh1708001@gmail.com
