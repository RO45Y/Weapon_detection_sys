import cv2
from yolov5.weapon_detection import run_detection

img = cv2.imread("OIP.jpg")
detections = run_detection(img)

print("Detections:", detections)
