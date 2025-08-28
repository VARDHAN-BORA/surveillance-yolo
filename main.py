# Install first if you haven't:
# pip install ultralytics opencv-python streamlit opencv-contrib-python

import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
from datetime import datetime

# --- CONFIG ---
MODEL_PATH = "/Users/venkatasaivardhanbora/Desktop/GIT/surveillance-yolo/src/yolov8n.pt"  # YOLOv8 Nano for speed
CONF_THRESHOLD = 0.3
TARGET_CLASSES = ["person", "car", "bicycle", "motorcycle", "bus", "truck"]  # Surveillance-focused

# --- LOAD MODEL ---
model = YOLO(MODEL_PATH)

st.title("Real-Time Object Detection for Surveillance")
st.markdown("Upload an image or video to see YOLOv8 detections (person, vehicles, etc.)")

uploaded_file = st.file_uploader("Upload Image/Video", type=["jpg","jpeg","png","mp4","avi"])
if uploaded_file is not None:
    # Save to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    file_path = tfile.name

    # --- IMAGE CASE ---
    if uploaded_file.type.startswith("image"):
        frame = cv2.imread(file_path)
        results = model(frame)
        
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                if conf < CONF_THRESHOLD or label not in TARGET_CLASSES:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Detections", use_column_width=True)

    # --- VIDEO CASE ---
    elif uploaded_file.type.startswith("video"):
        stframe = st.empty()
        cap = cv2.VideoCapture(file_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            
            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = model.names[cls]
                    if conf < CONF_THRESHOLD or label not in TARGET_CLASSES:
                        continue
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        cap.release()
