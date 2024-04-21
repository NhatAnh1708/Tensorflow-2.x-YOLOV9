import argparse
import time

import cv2
import numpy as np

from config.config import load_config
from src.utils import *

config = load_config("config/model/yolov9.yaml")
## Load configuration from a YAML file.
INPUT_SHAPE = config["input_shape"]
CONF_THRESHOLD = config["conf_threshold"]
MODEL_PATH = config["model_path"]
COCO_CLASSES = config["coco_classes"]

VIDEO_PATH = "data_test/testcase1.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)
labels = load_labels(path=COCO_CLASSES)
while True:
    ret, frame = cap.read()
    if ret:
        start = time.time()
        interpreter, input_details, output_details = load_model_tflite(
            MODEL_PATH, num_threads=10
        )
        frame = cv2.resize(frame, (INPUT_SHAPE, INPUT_SHAPE))
        bgr, ratio, dwdh = letterbox(frame, (INPUT_SHAPE, INPUT_SHAPE))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb)
        tensor = np.ascontiguousarray(tensor)
        interpreter.set_tensor(input_details[0]["index"], tensor)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]["index"])
        predictions = np.array(predictions).reshape((84, 8400))
        predictions = predictions.T

        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > CONF_THRESHOLD, :]
        scores = scores[scores > CONF_THRESHOLD]
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        boxes = predictions[:, :4]
        indices = nms(boxes, scores, CONF_THRESHOLD)
        for bbox, score, label in zip(
            xywh2xyxy(boxes[indices]), scores[indices], class_ids[indices]
        ):
            x, y, x2, y2 = bbox
            x1 = int(x)
            y1 = int(y)
            x2 = int(x2)
            y2 = int(y2)
            cls_id = int(label)
            cls = labels[cls_id]
            if labels[label] == "person" or labels[label] == "car":
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            else:
                continue
            cv2.putText(
                frame,
                labels[label],
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        fps = 1 / (time.time() - start)
        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imwrite("assets/output1.jpg", frame)
        cv2.imshow("video", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
