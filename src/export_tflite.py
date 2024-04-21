import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("yolov9e_pb")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()
with open("weights/yolov9_e_float32.tflite", "wb") as path:
    path.write(tflite_quantized_model)

# converter = tf.lite.TFLiteConverter.from_saved_model("yolov9e_pb")
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]
# tflite_quantized_model = converter.convert()
# open("yolox_nano_float16_quantize.tflite", "wb").write(tflite_quantized_model)
