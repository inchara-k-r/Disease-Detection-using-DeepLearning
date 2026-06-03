from flask import Flask, Response, jsonify, render_template
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# YOLO model settings
MODEL_PATH = "best.pt"
TARGET_CLASS = "bell_pepper_bacterial_spot"
CONF_THRESHOLD = 0.7

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)

spray_detected = False

def generate_frames():
    global spray_detected
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame, conf=CONF_THRESHOLD, verbose=False)
        annotated = results[0].plot()

        spray_detected = False
        if results and results[0].boxes:
            for box in results[0].boxes:
                class_name = model.names[int(box.cls)]
                if TARGET_CLASS.lower() in class_name.lower():
                    spray_detected = True
                    break

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
    app.run(host='0.0.0.0', port=5000, debug=True)
