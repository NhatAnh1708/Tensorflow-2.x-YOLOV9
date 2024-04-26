import os
import time

import cv2
import numpy as np
import tensorflow as tf

from ssd_utils import *


def run_detection(image, interpreter):
    # Run model: start to detect
    # Sets the value of the input tensor.
    interpreter.set_tensor(input_details[0]["index"], image)
    # Invoke the interpreter.
    interpreter.invoke()
    # get results
    boxes = interpreter.get_tensor(output_details[0]["index"])
    classes = interpreter.get_tensor(output_details[1]["index"])
    scores = interpreter.get_tensor(output_details[2]["index"])
    num = interpreter.get_tensor(output_details[3]["index"])

    boxes, scores, classes = (
        np.squeeze(boxes),
        np.squeeze(scores),
        np.squeeze(classes + 1).astype(np.int32),
    )
    out_scores, out_boxes, out_classes = non_max_suppression(scores, boxes, classes)

    # Print predictions info
    # print('Found {} boxes for {}'.format(len(out_boxes), 'images/dog.jpg'))

    return out_scores, out_boxes, out_classes


def real_time_object_detection(interpreter, colors, model_path):
    video_path = "data_test/testcase11.webm"
    camera = cv2.VideoCapture(video_path)
    video_name = "teet"
    model_name = model_path.split("/")[-1].split(".")[0]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec

    if not os.path.exists(f"output_tflite/{model_name}/{video_name}"):
        os.makedirs(f"output_tflite/{model_name}/{video_name}")
    out = cv2.VideoWriter(f'output_tflite/output_video_{model_name}.avi', fourcc, 30, (320, 320))
    desired_fps = 30
    count = 0
    # camera.set(cv2.CAP_PROP_FPS, desired_fps)
    while True:
        start = time.time()
        ret, frame = camera.read()

        if not ret:
            break

        image_data = preprocess_image_for_tflite(frame, model_image_size=320)
        out_scores, out_boxes, out_classes = run_detection(image_data, interpreter)
        # Draw bounding boxes on the image file
        result = draw_boxes(
            frame, out_scores, out_boxes, out_classes, class_names, colors
        )
        end = time.time()

        # fps
        t = end - start
        fps = "FPS: {:.2f}".format(1 / t)
        cv2.putText(
            result,
            fps,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        elapsed_time = time.time() - start
        text_ms = "Elapsed time:" + "%.0f" % (elapsed_time * 1000)
        text_ms = text_ms + "ms"
        cv2.putText(
            result,
            text_ms,
            (200, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            thickness=2,
        )
        cv2.putText(
            result,
            f"Bboxes :{str(len(out_classes))}",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        out.write(frame)
        cv2.imshow("Object detection - ssdlite_mobilenet_v2", frame)
        # if count % 20 == 0 and count < 1000:
        #     print("Processing : ", count)
        #     cv2.imwrite(
        #         f"output_tflite/{model_name}/{video_name}/frame_{count}.jpg", frame
        #     )
        # if count == 1000:
        #     break
        count += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    out.release()
    camera.release()


if __name__ == "__main__":
    # Load TFLite model and allocate tensors.
    model_path = "weights/ssd_v3_1.tflite"
    interpreter = tf.lite.Interpreter(
        model_path=model_path,
        num_threads=10,
    )
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # label
    class_names = read_classes("config/coco_classes.txt")
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # image_object_detection(interpreter, colors)
    real_time_object_detection(interpreter, colors, model_path)
