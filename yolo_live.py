
import cv2
from ultralytics import YOLO

MODEL_PATH = "best.pt"  # Your trained YOLO model; replace with yolov8n.pt if testing
TARGET_CLASS = "bell_pepper_bacterial_spot"
CONF_THRESHOLD = 0.5

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    # Annotate frame
    annotated_frame = results[0].plot()

    # Debug: print all detected class names and confidences
    print([(model.names[int(box.cls)], float(box.conf)) for box in results[0].boxes])

    # Optional: check for TARGET_CLASS with exact match and confidence check
    for box in results[0].boxes:
        class_name = model.names[int(box.cls)]
        conf = float(box.conf)
        if class_name.lower() == TARGET_CLASS.lower() and conf > CONF_THRESHOLD:
            print(f"Spray alert! Detected {class_name} with confidence {conf:.2f}")

    # Show annotated frame
    cv2.imshow("YOLO Detection", annotated_frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
