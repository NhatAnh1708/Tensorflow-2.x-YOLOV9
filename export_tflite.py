import tensorflow as tf

# converter = tf.lite.TFLiteConverter.from_saved_model('yolov9e_pb')
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# tflite_quantized_model = converter.convert()

# open('yolov9_e_dynamic_range_quantize.tflite', 'wb').write(tflite_quantized_model)

# converter = tf.lite.TFLiteConverter.from_saved_model("yolov9e_pb")
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]
# tflite_quantized_model = converter.convert()

# open("yolox_nano_float16_quantize.tflite", "wb").write(tflite_quantized_model)

import glob
import numpy as np
from PIL import Image

image_pathlist = glob.glob('../*.jpg')
image_pathlist = image_pathlist[:100]

def representative_dataset():
    for test_image_path in image_pathlist:
        image = np.array(Image.open(test_image_path))
        image = image.astype('float32')
        image = tf.image.resize(image, (640, 640))
        image = image - 127.5
        image = image * 0.007843
        image = tf.transpose(image, perm=[2, 0, 1])
        image = tf.reshape(image, [1, 3, 640, 640])
        yield [image]

converter = tf.lite.TFLiteConverter.from_saved_model('yolov9e_pb')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
tflite_quantized_model = converter.convert()

open('yolo9_nano_int8_quantize.tflite', 'wb').write(tflite_quantized_model)
