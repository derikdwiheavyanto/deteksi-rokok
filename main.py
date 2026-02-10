import streamlit as st
from ultralytics import YOLO
import cv2

st.set_page_config(page_title="Deteksi Rokok", layout="wide")
st.title("🚬 Deteksi Rokok Realtime (YOLOv8)")

# Load model
@st.cache_resource
def load_model():
    return YOLO("weight/best.pt", task="detect")

model = load_model()

# Sidebar
conf_thres = st.sidebar.slider(
    "Confidence Threshold", 0.1, 1.0, 0.5, 0.05
)

start = st.sidebar.button("▶ Start Camera")
stop = st.sidebar.button("⏹ Stop Camera")

frame_window = st.image([])

if start:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Kamera tidak bisa dibuka")
            break

        frame = cv2.flip(frame, 1)

        results = model(
            frame,
            imgsz=640,
            conf=conf_thres,
            stream=True,
            verbose=False,
        )

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls in zip(xyxy, confs, clss):
                x1, y1, x2, y2 = map(int, box)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{model.names[cls]} {conf*100:.2f}%",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        # BGR → RGB (WAJIB di Streamlit)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame, channels="RGB")

        if stop:
            break

    cap.release()
