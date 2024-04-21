import copy
from dataclasses import dataclass
import cv2
import numpy as np
import tensorflow as tf



@dataclass 
class Yolov9TFlite:
    model_path: str = "yolov9_e_float16_quantize.tflite"
    input_shape = (640, 640)
    num_threads = 10
    interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()


    def inference(self, image):
        copy_image = copy.deepcopy(image)
        image_reshape = self._preprocessing(image)


    def _preprocessing(self, image):
        image_resize = cv2.resize(image, (640, 640))
        print(image_resize.shape)
