import os
import cv2
import numpy as np
import dlib
import openpyxl
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications.mobilenet_v2 import preprocess_input

# Load the face shape model
model = load_model('face_shape_model.keras')

# Load frame suggestions
frame_suggestions = pd.read_csv('frame_suggestions.csv')

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize dlib's face detector (HOG-based) and create a facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


# Overlay the frame on the face
def overlay_frame(face_img, frame_img, x, y, w, h):
    frame_h, frame_w = frame_img.shape[:2]

    # Calculate scaling factors to preserve frame aspect ratio
    scale_w = w / frame_w
    scale_h = h / frame_h
    scale = min(scale_w, scale_h)

    # Resize frame while maintaining aspect ratio
    new_w = int(frame_w * scale)
    new_h = int(frame_h * scale)
    frame_img_resized = cv2.resize(frame_img, (new_w, new_h))

    # Calculate starting coordinates for overlay (centering)
    start_x = x + (w - new_w) // 2
    start_y = y + (h - new_h) // 2

    # Extract ROI with potential alpha channel
    roi = face_img[start_y:start_y + new_h, start_x:start_x + new_w]

    # Handle alpha channel if present
    if frame_img_resized.shape[2] == 4:
        alpha_frame = frame_img_resized[:, :, 3] / 255.0
        frame_img_no_alpha = frame_img_resized[:, :, :3]
        for c in range(3):
            roi[:, :, c] = (1 - alpha_frame) * roi[:, :, c] + alpha_frame * frame_img_no_alpha[:, :, c]

    # Directly copy frame image if no alpha channel
    else:
        roi[:] = frame_img_resized
    return face_img


# Initialize the Excel database
excel_file = 'face_shape_database.xlsx'
try:
    wb = openpyxl.load_workbook(excel_file)
    ws = wb.active
except FileNotFoundError:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(['Name', 'Age', 'Gender', 'Face Shape', 'Suggested Frames', 'Feedback'])


# Suggest frames and update database
def suggest_frames_and_update_db(face_shape, age, gender, name):
    suggestions = frame_suggestions[(frame_suggestions['face_shape'] == face_shape) &
                                    (frame_suggestions['min_age'] <= age) &
                                    (frame_suggestions['max_age'] >= age) &
                                    (frame_suggestions['gender'] == gender)]
    suggested_frames = suggestions['frame_image'].tolist()
    ws.append([name, age, gender, face_shape, ', '.join(suggested_frames), ''])  # Save to Excel
    wb.save(excel_file)
    return suggested_frames


# Capture an image from the camera
def capture_image():
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow('Capture Image', img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == 13:  # Enter key
            if len(faces) > 0:
                cap.release()
                cv2.destroyAllWindows()
                return img, faces[0]
            else:
                print("No faces detected. Please try again.")
                continue
    cap.release()
    cv2.destroyAllWindows()


# Predict the face shape using the pre-trained model and dlib landmarks
def predict_face_shape(image, face_coords):
    x, y, w, h = face_coords
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rect = dlib.rectangle(x, y, x + w, y + h)
    shape = predictor(gray, rect)
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])

    # Select a subset of 5 key landmarks (e.g., two eyes, nose tip, and mouth corners)
    key_landmarks = np.concatenate([
        landmarks[36],  # Left eye corner
        landmarks[45],  # Right eye corner
        landmarks[30],  # Nose tip
        landmarks[48],  # Left mouth corner
        landmarks[54]  # Right mouth corner
    ]).flatten().reshape(1, -1)

    face_img = cv2.resize(image[y:y + h, x:x + w], (64, 64))  # Resize the image
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = img_to_array(face_img)
    face_img = preprocess_input(face_img)
    face_img = np.expand_dims(face_img, axis=0)

    print(f"Face image shape: {face_img.shape}")
    print(f"Key landmarks shape: {key_landmarks.shape}")

    predictions = model.predict([face_img, key_landmarks])
    print(f"Model predictions: {predictions}")

    # Get the index of the predicted class
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    # Map index to the corresponding face shape class
    face_shape = ["Diamond", "Heart", "Oval", "Rectangle", "Round"][predicted_class_index]
    return face_shape


# Try on the suggested frames
def tryon_frames(captured_img, face_coords, frame_paths):
    x, y, w, h = face_coords
    num_frames = len(frame_paths)

    # Create two windows if there are at least two suggestions
    if num_frames >= 2:
        cv2.namedWindow('Frame 1', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Frame 2', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Frame 1', 500, 500)  # Set desired window size
        cv2.resizeWindow('Frame 2', 500, 500)
    else:
        cv2.namedWindow('Try-On Frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Try-On Frame', 500, 500)

    for i, frame_path in enumerate(frame_paths):
        frame_img = cv2.imread(frame_path, -1)
        if frame_img is None:
            print(f"Error loading frame image: {frame_path}")
            continue
        img_with_frame = overlay_frame(captured_img.copy(), frame_img, x, y, w, h)

        if num_frames >= 2:
            # Display in Frame 1 or Frame 2 alternately
            window_name = 'Frame 1' if i % 2 == 0 else 'Frame 2'
            cv2.imshow(window_name, img_with_frame)
        else:
            cv2.imshow('Try-On Frame', img_with_frame)

        # Check if all images have been shown and wait for a key press
        if (i + 1) % 2 == 0 or i == num_frames - 1:
            cv2.waitKey(0)

    cv2.destroyAllWindows()


# Main program
def main():
    name = input("Enter your name: ")
    gender = input("Enter your gender (Male/Female): ")
    age = int(input("Enter your age: "))

    captured_img, face_coords = capture_image()
    face_shape = predict_face_shape(captured_img, face_coords)
    print(f"Detected face shape: {face_shape}")

    suggested_frames = suggest_frames_and_update_db(face_shape, age, gender, name)
    if not suggested_frames:
        print("No frame suggestions available in the database.")
        return

    print("Trying suggested frames...")
    frame_paths = [os.path.join('frames', frame) for frame in suggested_frames]
    tryon_frames(captured_img, face_coords, frame_paths)


if __name__ == "__main__":
    main()
