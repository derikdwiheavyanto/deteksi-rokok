from ultralytics import YOLO
import cv2


model = YOLO(
    "weight/best.pt",
    task="detect",
)

cap = cv2.VideoCapture("test.mp4")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    filename="output.mp4",
    fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
    fps=fps,
    frameSize=(w, h),
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    result = model(
        frame,
        verbose=True,
        stream=True,
        imgsz=640,
        conf=0.5,
        iou=0.5,
    )

    for r in result:
        anotated = r.plot()
        out.write(anotated)

cap.release()
out.release()
