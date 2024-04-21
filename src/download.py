import argparse
import logging
from ultralytics import YOLO

arg = argparse.ArgumentParser()
# model_name = arg.add_argument("--model_name", type=str, default="yolov9e")
# model_name_path = f"{model_name}.pt"
model_name = "YOLOv9-e"
model_name_path = "yolov9e.pt"
model = YOLO(model_name_path)
logging.warning("Model name: {%s} is loaded successfully", model_name)
