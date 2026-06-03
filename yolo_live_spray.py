import cv2
from ultralytics import YOLO
import time

MODEL_PATH = "best.pt"  # or your trained model path
TARGET_CLASS = "bell_pepper_bacterial_spot"
CONF_THRESHOLD = 0.7

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)  # 0 is default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLO detection
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)
    annotated = results[0].plot()

    # Debug: print all detected class names and confidences
    print([(model.names[int(box.cls)], float(box.conf)) for box in results[0].boxes])

    # Check if target class is detected - exact match ignoring case
    spray = False
    if results and results[0].boxes:
        for box in results[0].boxes:
            class_name = model.names[int(box.cls)]
            conf = float(box.conf)
            if class_name.lower() == TARGET_CLASS.lower() and conf > CONF_THRESHOLD:
                spray = True
                break

    # Simulate spraying
    if spray:
        print("Target detected! Spraying...")
    else:
        print("No target detected.")

    # Show live video
    cv2.imshow("YOLO Live Spray", annotated)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
