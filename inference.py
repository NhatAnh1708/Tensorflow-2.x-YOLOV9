import argparse
import logging
import os
import time

import cv2
import numpy as np

from config.config import load_config
from src.utils import *


def get_args():
    """Get command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config/model/Yolov9e/yolov9_416.yaml"
    )
    parser.add_argument("--model", type=str, default="yolov9")
    parser.add_argument("--video", type=str, default="data_test/testcase8.mp4")
    args = parser.parse_args()

    return args


def run_yolov9(
    frame, input_shape, prediction_shape, interpreter, input_details, output_details
):
    """YOLO v9"""
    bgr, ratio, dwdh = letterbox(frame, (input_shape, input_shape))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = blob(rgb)
    tensor = np.ascontiguousarray(tensor)
    interpreter.set_tensor(input_details[0]["index"], tensor)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]["index"])
    predictions = np.array(predictions).reshape((84, prediction_shape))
    predictions = predictions.T
    return predictions


def run_yolov8(
    frame, input_shape, prediction_shape, interpreter, input_details, output_details
):
    """YOLO v8"""
    bgr, _, _ = letterbox(frame, input_shape)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = blob(rgb)
    tensor = np.transpose(tensor, (0, 2, 3, 1))
    tensor = np.ascontiguousarray(tensor)
    interpreter.set_tensor(input_details[0]["index"], tensor)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]["index"])
    predictions = np.array(predictions).reshape((84, prediction_shape))

    predictions = predictions.T
    return predictions


def main():
    """Run"""
    arg = get_args()
    config_path = arg.config
    config = load_config(config_path)
    ## Load configuration from a YAML file.
    input_shape = config["input_shape"]
    conf_threshold = config["conf_threshold"]
    model_path = config["model_path"]
    coco_classes = config["coco_classes"]
    prediction_shape = config["prediction_shape"]
    interpreter, input_details, output_details = load_model_tflite(
        model_path, num_threads=10
    )
    count = 0
    video_path = arg.video
    if video_path == "0":
        video_path = 0
        video_name = "webcam"
    else:
        video_name = video_path.split("/")[-1].split(".")[0]
    model_name = model_path.split("/")[-1].split(".")[0]
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec

    if not os.path.exists(f"output_tflite/{model_name}/{video_name}"):
        os.makedirs(f"output_tflite/{model_name}/{video_name}")
    out = cv2.VideoWriter(f'output_tflite/output_video_{model_name}.avi', fourcc, 30, (input_shape, input_shape))
    labels = load_labels(path=coco_classes)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        start = time.time()
        frame = cv2.resize(frame, (input_shape, input_shape))
        if arg.model == "yolov9":
            logging.info("Running YOLO v9")
            predictions = run_yolov9(
                frame,
                input_shape,
                prediction_shape,
                interpreter,
                input_details,
                output_details,
            )
        elif arg.model == "yolov8":
            logging.info("Running YOLO v8")
            predictions = run_yolov8(
                frame,
                input_shape,
                prediction_shape,
                interpreter,
                input_details,
                output_details,
            )
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > conf_threshold, :]
        scores = scores[scores > conf_threshold]
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        boxes = predictions[:, :4]
        indices = nms(boxes, scores, conf_threshold)
        list_bboxes_person = []
        list_bboxes_vehicle = []
        for bbox, score, label in zip(
            xywh2xyxy(boxes[indices]), scores[indices], class_ids[indices]
        ):
            bbox = scale_bbox(bbox, arg.model, input_shape)
            cls_id = int(label)
            # cls = labels[cls_id]
            if cls_id == 0:
                list_bboxes_person.append(bbox)
                color = (0, 0, 128)
                draw_bbox(image=frame, bbox=bbox, color=color)
            if (
                cls_id == 2
                or cls_id == 5
                or cls_id == 6
                or cls_id == 7
                or cls_id == 8
            ):
                list_bboxes_vehicle.append(bbox)
                color = (0, 128, 0)
                draw_bbox(image=frame, bbox=bbox, color=color)
            else:
                continue
        latency = (time.time() - start) * 1000
        cv2.putText(
            frame,
            f"Person: {len(list_bboxes_person)}",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Vehicle: {len(list_bboxes_vehicle)}",
            (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Latency: {latency:.2f} ms",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
        )
        if count % 20 == 0 and count < 1000:
            logging.warning("Loading frame")
            cv2.imwrite(
                f"output_tflite/{model_name}/{video_name}/frame_{str(count)}.jpg",
                frame,
            )
        if count == 1000:
            logging.info("Finished")
            break
        
        count += 1
        cv2.imshow("video", frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
