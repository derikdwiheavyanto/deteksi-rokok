import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import torch
import os
import gdown

WEIGHT_PATH = "weights/best.pt"

# ===============================
# Download model jika belum ada
# ===============================
if not os.path.exists(WEIGHT_PATH):
    os.makedirs("weights", exist_ok=True)

    FILE_ID = "1G5a8tMvZwAiO26cUoXRErGDwhWyAtrgL"
    url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

    gdown.download(url, WEIGHT_PATH, quiet=False)

# ===============================
# Load model (cache)
# ===============================
@st.cache_resource
def load_model():
    return YOLO(WEIGHT_PATH)

model = load_model()


print(f"Cuda is available: {torch.cuda.is_available()}")
device = 0 if torch.cuda.is_available() else "cpu"

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="Deteksi Rokok", layout="wide")
st.title("🚬 Deteksi Rokok Realtime (YOLOv8 + WebRTC)")

conf_thres = st.sidebar.slider(
    "Confidence Threshold", 0.1, 1.0, 0.5, 0.05
)

# ===============================
# Video Processor
# ===============================
class YOLOProcessor(VideoProcessorBase):

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img = cv2.flip(img, 1)
        
        # YOLO inference
        results = model(
            img,
            imgsz=640,
            conf=conf_thres,
            device=device,
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

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    f"{model.names[cls]} {conf*100:.1f}%",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ===============================
# WebRTC Configuration
# ===============================
rtc_config = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="yolo",
    video_processor_factory=YOLOProcessor,
    rtc_configuration=rtc_config,
    media_stream_constraints={"video": True, "audio": False},
)
