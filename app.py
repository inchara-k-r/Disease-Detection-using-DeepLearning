from flask import Flask, Response, jsonify, render_template
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# YOLO model settings
MODEL_PATH = "best.pt"
CONF_THRESHOLD = 0.5     # LOWERED threshold for more detections

# Initialize model and camera
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)

spray_detected = False

def generate_frames():
    global spray_detected
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Run YOLO inference
        results = model(frame, conf=CONF_THRESHOLD, verbose=False)
        annotated = results[0].plot()

        # Process detection results
        detected_classes = set()
        spray_detected = False
        y_offset = 30  # starting position for overlay text

        for box in results[0].boxes:
            classname = model.names[int(box.cls)]
            conf = float(box.conf)
            detected_classes.add((classname, conf))
            # Overlay label on each detection
            label = f"{classname}: {conf:.2f}"
            cv2.putText(annotated, label, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y_offset += 30

            # If any disease class is detected (not healthy and above threshold), update spray_detected
            if "healthy" not in classname.lower() and conf > CONF_THRESHOLD:
                spray_detected = True

        # Debug print to see what is actually detected by model
        print("Detected classes and confidences:", detected_classes)
        
        # Encode and yield frame
        ret, buffer = cv2.imencode('.jpg', annotated)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/spray_status')
def spray_status():
    return jsonify({'spray': spray_detected})

@app.route('/plant_data')
def plant_data():
    # Sample plant health data; replace with real detection results
    plant_data = [
        {"id": "Plant001", "affected": 35},
        {"id": "Plant002", "affected": 60},
        {"id": "Plant003", "affected": 15},
        {"id": "Plant004", "affected": 80},
        {"id": "Plant005", "affected": 50}
    ]
    return jsonify(plant_data)

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    finally:
        cap.release()
        cv2.destroyAllWindows()
