import time

import cv2
import numpy as np
import tensorflow as tf


def load_model_tflite(model_path, num_threads=10):
    """Load TFLite model."""
    interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


def load_labels(path):
    """Load labels from file."""
    with open(path, "r", encoding="utf-8") as f:
        labels = f.read().splitlines()
    return labels


def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    """Resize image to a 32-pixel-multiple rectangle, padding with color."""
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, r, (dw, dh)


def blob(im, return_seg=False):
    """Convert image to 4D blob."""
    if return_seg:
        seg = im.astype(np.float32) / 255
    im = im.transpose([2, 0, 1])
    im = im[np.newaxis, ...]
    im = np.ascontiguousarray(im).astype(np.float32) / 255
    if return_seg:
        return im, seg
    else:
        return im

def blob_v1(im, return_seg=False):
    """Convert image to 4D blob."""
    if return_seg:
        seg = im.astype(np.int8) 
    im = im.transpose([2, 0, 1])
    im = im[np.newaxis, ...]
    im = np.ascontiguousarray(im).astype(np.int8)
    if return_seg:
        return im, seg
    else:
        return im

def nms(boxes, scores, iou_threshold):
    """Non-maximum suppression."""
    # Convert to xyxy
    boxes = xywh2xyxy(boxes)
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]
    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)
        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])
        # Remove boxes with IoU over threshold
        keep_indices = np.where(ious < iou_threshold)[0] + 1
        sorted_indices = sorted_indices[keep_indices]

    return keep_boxes


def compute_iou(box, boxes):
    """Compute xmin, ymin, xmax, ymax for both boxes"""
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])
    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area
    # Compute IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    """Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)"""
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def draw_bbox(image, bbox, color=(0, 0, 255)):
    """Draw bounding box on image."""
    thickness = 2
    x1, y1, x2, y2 = bbox
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)
    masking = (51, 51)
    length = (x2 - x1) // 5
    cv2.line(image, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(image, (x1, y1), (x1, y1 + length), color, thickness)
    # Top-right corner
    cv2.line(image, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(image, (x2, y1), (x2, y1 + length), color, thickness)

    # Bottom-left corner
    cv2.line(image, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(image, (x1, y2), (x1, y2 - length), color, thickness)

    # Bottom-right corner
    cv2.line(image, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(image, (x2, y2), (x2, y2 - length), color, thickness)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    blurred_roi = cv2.GaussianBlur(roi, masking, 0)
    image[y1:y2, x1:x2] = blurred_roi

def draw_bbox_v1(image, bbox, color=(0, 0, 255)):
    """Draw bounding box on image."""
    thickness = 2
    x1, y1, x2, y2 = bbox
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)
    print(x1, y1, x2, y2)
    cv2.circle(image, (x1, y1), 5, color, thickness)
    cv2.circle(image, (x2, y2), 5, color, thickness)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

def transform_point(point, surface_size):
    # Tính toán tỷ lệ giữa kích thước thực tế và kích thước ảo của bề mặt
    scale_x = surface_size[0] / (max(point[0], point[1]) - min(point[0], point[1]))
    scale_y = surface_size[1] / (max(point[0], point[1]) - min(point[0], point[1]))
    
    # Áp dụng tỷ lệ cho các tọa độ
    x = (point[0] - min(point[0], point[1])) * scale_x
    y = surface_size[1] - ((point[1] - min(point[0], point[1])) * scale_y)  # Đảo ngược y
    
    return x, y

def add_salt_pepper_noise(image, amount):
    """Add salt and pepper noise to image."""
    noisy_image = np.copy(image)
    row, col, _ = noisy_image.shape
    num_salt = np.ceil(amount * image.size * 0.5)
    num_pepper = np.ceil(amount * image.size * 0.5)
    # Add salt noise
    salt_coords = [
        np.random.randint(0, i - 1, int(num_salt)) for i in noisy_image.shape
    ]
    noisy_image[salt_coords[0], salt_coords[1], :] = 255
    # Add pepper noise
    pepper_coords = [
        np.random.randint(0, i - 1, int(num_pepper)) for i in noisy_image.shape
    ]
    noisy_image[pepper_coords[0], pepper_coords[1], :] = 0
    return noisy_image


def scale_bbox(bbox, model_name, input_shape):
    if model_name == "yolov9":
        return bbox
    elif model_name == "yolov8":

        return bbox * input_shape
