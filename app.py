# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time
from yolov5.weapon_detection import run_detection  # Make sure this function is updated as below

st.set_page_config(page_title="Weapon Detection", layout="centered")
st.title("🔫 Weapon Detection System")

# Sidebar for upload and webcam options
mode = st.sidebar.radio("Choose Mode", ["Upload Image/Video", "Webcam"])

# Create a folder to store detected images
if not os.path.exists("detected"):
    os.makedirs("detected")

# Webcam feature status
st.sidebar.markdown("---")
st.sidebar.markdown("### Webcam Detection Status")
webcam_status = st.sidebar.empty()

if mode == "Upload Image/Video":
    uploaded_file = st.file_uploader("Upload image or video", type=["jpg", "jpeg", "png", "mp4"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.close()

        if uploaded_file.type.startswith("image"):
            try:
                img = cv2.imread(tfile.name)
                detections = run_detection(img)

                for label, conf, (x1, y1, x2, y2) in detections:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(img, f"{label} ({conf:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                st.image(img, caption="Detected Image", channels="BGR")
            except Exception as e:
                st.error(f"Failed to process image: {e}")

        elif uploaded_file.type.endswith("mp4"):
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                detections = run_detection(frame)
                for label, conf, (x1, y1, x2, y2) in detections:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                stframe.image(frame, channels="BGR")
            cap.release()

elif mode == "Webcam":
    st.markdown("### Webcam Live Feed")
    start_webcam = st.button("Start Webcam Detection")

    if start_webcam:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        detect_start_time = None
        captured = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = run_detection(frame)

            if detections:
                label, conf, (x1, y1, x2, y2) = detections[0]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                webcam_status.success(f"🚨 {label} detected!")

                if detect_start_time is None:
                    detect_start_time = time.time()
                elif time.time() - detect_start_time > 2 and not captured:
                    filename = f"detected/capture_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    st.success(f"🚨 {label} Detected — Image Saved!")
                    captured = True
            else:
                webcam_status.info("✅ No weapon detected")
                detect_start_time = None
                captured = False

            stframe.image(frame, channels="BGR")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()