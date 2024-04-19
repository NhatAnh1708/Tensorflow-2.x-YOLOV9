from config.Yolov9 import Yolov9TFlite
import cv2 


image = cv2.imread("/home/danny/Honda/YOLOX-ONNX-TFLite-Sample/representative_dataset/000000000139.jpg")




yolov9 = Yolov9TFlite(model_path='/home/danny/Honda/Yolov9-tflite/yolov9_e_float16_quantize.tflite')

yolov9.inference(image)
