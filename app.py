# app.py
import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from yolov5.weapon_detection import run_detection  # Updated function returning frame & flag

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

# Initialize webcam session state
if "webcam_running" not in st.session_state:
    st.session_state.webcam_running = False

if mode == "Upload Image/Video":
    uploaded_file = st.file_uploader("Upload image or video", type=["jpg", "jpeg", "png", "mp4"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.close()

        if uploaded_file.type.startswith("image"):
            try:
                img = cv2.imread(tfile.name)
                detections, processed_img, weapon_found = run_detection(img)

                st.image(processed_img, caption="Detected Image", channels="BGR")

                if weapon_found:
                    st.error("🚨 Weapon Detected!")
                else:
                    st.success("✅ No Weapon Detected")
            except Exception as e:
                st.error(f"Failed to process image: {e}")

        elif uploaded_file.type.endswith("mp4"):
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                detections, processed_frame, weapon_found = run_detection(frame)
                stframe.image(processed_frame, channels="BGR")

            cap.release()

elif mode == "Webcam":
    st.markdown("### Webcam Live Feed")

    if not st.session_state.webcam_running:
        if st.button("▶️ Start Webcam"):
            st.session_state.webcam_running = True
    else:
        if st.button("⏹️ Stop Webcam"):
            st.session_state.webcam_running = False

    if st.session_state.webcam_running:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        detect_start_time = None
        captured = False

        while st.session_state.webcam_running:
            ret, frame = cap.read()
            if not ret:
                break

            detections, processed_frame, weapon_found = run_detection(frame)

            if weapon_found:
                webcam_status.error("🚨 Weapon Detected")

                if detect_start_time is None:
                    detect_start_time = time.time()
                elif time.time() - detect_start_time > 2 and not captured:
                    filename = f"detected/capture_{int(time.time())}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    st.success("🚨 Weapon Detected — Image Saved!")
                    captured = True
            else:
                webcam_status.success("✅ No Weapon Detected")
                detect_start_time = None
                captured = False

            stframe.image(processed_frame, channels="BGR")

        cap.release()
        cv2.destroyAllWindows()
