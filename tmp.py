import os
import time

import cv2
import numpy as np
import tensorflow as tf

model_path = "/home/danny/Honda/Yolov9-tflite/yolov9_e_float16_quantize.tflite"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
interpreter = tf.lite.Interpreter(
    model_path=model_path,
    num_threads=10,
)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# Get stream from webcam
video_path = "/home/danny/Honda/Yolov9-tflite/data_test/testcase_person.mp4"
video_name = video_path.split("/")[-1].split(".")[0]
model_name = model_path.split("/")[-1].split(".")[0]
if not os.path.exists(f"output_tflite/{model_name}/{video_name}"):
    os.makedirs(f"output_tflite/{model_name}/{video_name}")
batch_size = 1
frames = []
video = cv2.VideoCapture(video_path)
size = 640
count = 0

while True:
    start_time = time.time()
    ret, frame = video.read()
    if not ret:
        break
    frame = cv2.resize(frame, (size, size))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = frame_rgb.astype(np.float32) / 255.0
    # frame_rgb = frame_rgb.astype(np.uint8)  # Normalize
    frame_expanded = np.expand_dims(frame_rgb, axis=0)
    max_score = 0
    max_detection = None
    converted_array = np.transpose(frame_expanded, (0, 3, 1, 2))
    interpreter.set_tensor(input_details[0]["index"], converted_array)
    interpreter.invoke()
    boxes = interpreter.get_tensor(output_details[0]["index"])[0]
    output_boxes = interpreter.get_tensor(output_details[0]["index"])[0]
    for i in range(output_boxes.shape[1]):
        detection = output_boxes[:, i]
        score = np.max(detection[4:])
        if score > max_score:
            max_score = score
            max_detection = detection
        if max_detection is not None and max_score >= 0.1:
            x_center, y_center, width, height = max_detection[:4]
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            cls = np.argmax(max_detection[4:])
            text = f"Class: {cls}, Score: {max_score:.2f}"
            if i % 2 == 0:
                color = (0, 255, 0)
            color = (0, 0, 255)
            if cls == 0:
                text = "Person"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
            elif cls == 2:
                text = "Car"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
            else:
                continue
    fps = (1 / (time.time() - start_time)) / batch_size
    cv2.putText(
        frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )
    if count % 20 == 0 and count < 1000:
        print("done: ", count)
        cv2.imwrite(f"output_tflite/{model_name}/{video_name}/frame_{count}.jpg", frame)
    count += 1
    cv2.imshow("Output", frame)  # Display frame with bounding box
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release everything if job is finished
