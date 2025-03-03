from flask import Flask, render_template, Response, request, redirect, url_for, flash
import cv2
from Capture_Image import takeImages  # Import từ file register.py
from Train_Image import FaceTrainer  # Import từ file train.py
from Recognize import FaceRecognizer  # Import từ file recognize.py

app = Flask(__name__)
app.secret_key = "your_secret_key"

camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register', methods=['POST'])
def register():
    student_id = request.form['id']
    name = request.form['name']
    
    takeImages(student_id, name)

    flash(f"✅ Student {name} (ID: {student_id}) registered successfully!")
    return redirect(url_for('index'))

@app.route('/train')
def train():
    FaceTrainer()  
    flash("✅ Training model completed successfully!")
    return redirect(url_for('index'))

@app.route('/recognize')
def recognize():
    FaceRecognizer()  
    flash("✅ Face recognition started!")
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, port=5001)
