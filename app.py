from flask import Flask, render_template, request, redirect, url_for
import face_recognition
import cv2
import numpy as np
from datetime import datetime
import pandas as pd
import os

FACES_DIR = 'static/faces'

if not os.path.exists(FACES_DIR):
    os.makedirs(FACES_DIR)

#app

app = Flask(__name__)

# Directories
UPLOAD_FOLDER = 'static/faces'
ENCODINGS_PATH = 'encodings.npy'
NAMES_PATH = 'names.npy'
CSV_FILE = 'attendance.csv'

# Ensure face upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Main Page
# Home Route 
@app.route('/')
def home():
    return render_template('index.html')

# Register Route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        image_file = request.files['image']
        if image_file:
            path = os.path.join(UPLOAD_FOLDER, f"{name}.jpg")
            image_file.save(path)
            return render_template('success.html', name=name)
    return render_template('register.html')

# Train Route – Generates encodings
@app.route('/train')
def train():
    known_faces = []
    known_names = []

    for filename in os.listdir('static/faces'):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image = face_recognition.load_image_file(f'static/faces/{filename}')
            encodings = face_recognition.face_encodings(image)
            if len(encodings) > 0:
                known_faces.append(encodings[0])
                known_names.append(os.path.splitext(filename)[0])
            else:
                print(f"[WARNING] No face found in {filename}")

    np.save(ENCODINGS_PATH, known_faces)
    np.save(NAMES_PATH, known_names)

    return render_template('train.html')  


# Attendance Route – Marks attendance via webcam
@app.route('/attendance')
def attendance():
    known_faces = []
    known_names = []

    # Load known face encodings
    for filename in os.listdir('static/faces'):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image = face_recognition.load_image_file(f'static/faces/{filename}')
            encodings = face_recognition.face_encodings(image)
            if len(encodings) > 0:
                known_faces.append(encodings[0])
                known_names.append(os.path.splitext(filename)[0])
    
    # ⬇️ Add this line right after loading known faces
    print("[INFO] Starting camera...")
    print(f"[INFO] Loaded {len(known_faces)} known faces: {known_names}")

    video_capture = cv2.VideoCapture(0)
    marked_names = []

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("[ERROR] Failed to read from camera.")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Get face locations and encodings
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            face_distances = face_recognition.face_distance(known_faces, face_encoding)

            # ⬇️ Insert debug lines here
            print(f"[DEBUG] Matches: {matches}")
            print(f"[DEBUG] Distances: {face_distances}")

            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]

                if name not in marked_names:
                    marked_names.append(name)
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open('Attendance.csv', 'a') as f:
                        f.write(f"{name},{now}\n")
                    print(f"[INFO] Marked attendance for: {name}")

        # Display window (optional — skip if running in server)
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    return "Attendance Complete"


@app.route('/report')
def report():
    import os
    import pandas as pd

    if os.path.exists('attendance.csv'):
        df = pd.read_csv('attendance.csv')
        records = df.to_dict(orient='records')  # ✅ Convert to list of dicts (no ambiguity)
        return render_template('report.html', data=records)
    return render_template('report.html', data=[])


if __name__ == '__main__':
    app.run(debug=True)
 
