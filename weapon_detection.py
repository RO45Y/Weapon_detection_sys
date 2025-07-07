#weapon_detection.py
import sys
import os

# Add yolov5 root to path
YOLOV5_PATH = os.path.join(os.path.dirname(__file__), 'yolov5')
sys.path.append(YOLOV5_PATH)

import torch
import cv2
import numpy as np
import time

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device


# Load model once
# Load model once
device = select_device('')  # or 'cuda:0' if available
model = DetectMultiBackend('best_compatible.pt', device=device)
model.model.float().eval()

# Detection threshold
conf_threshold = 0.2

def run_detection(frame):
    weapon_found = False

    # Preprocess frame
    resized = cv2.resize(frame, (640, 640))
    img = resized[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img)

    # Convert to tensor
    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        pred = model(img_tensor)
        pred = non_max_suppression(pred, conf_threshold, 0.45)[0]

    detected = []
    if pred is not None and len(pred):
        pred[:, :4] = scale_boxes(img_tensor.shape[2:], pred[:, :4], frame.shape).round()
        for *xyxy, conf, cls in pred:
            label = model.names[int(cls)]
            if label.lower() in ['gun', 'knife']:
                x1, y1, x2, y2 = map(int, xyxy)
                confidence = float(conf)
                detected.append((label, confidence, (x1, y1, x2, y2)))
                weapon_found = True

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Display detection message on frame
    msg = "WEAPON DETECTED" if weapon_found else "NO WEAPON DETECTED"
    color = (0, 0, 255) if weapon_found else (0, 255, 0)
    cv2.putText(frame, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    return detected, frame, weapon_found