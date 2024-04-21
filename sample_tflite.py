from config.Yolov9 import Yolov9TFlite
import cv2


image = cv2.imread("/home/../../yolox/representative_dataset/000000000139.jpg")


yolov9 = Yolov9TFlite()

yolov9.inference(image)
