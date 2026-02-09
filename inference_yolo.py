from ultralytics import YOLO
import torch
import cv2

# print("CUDA available:", torch.cuda.is_available())

# if torch.cuda.is_available():
#     print("GPU:", torch.cuda.get_device_name(0))

model = YOLO('weight/best.pt',task='detect',)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    frame = cv2.flip(frame, 1)
    
    result = model(frame,verbose=False, imgsz=640, stream=True)
    
    for r in result:
        boxes = r.boxes
        if boxes is None:
            continue
        
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        
        for xyxy, conf, clss in zip(xyxy, conf, clss):
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

            #draw bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            #draw label
            cv2.putText(
                frame,
                f'{model.names[clss]} {conf:.2f}',
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.6,
                (0, 255, 0),
                1
            )

    cv2.imshow('Deteksi Rokok', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
        
        

