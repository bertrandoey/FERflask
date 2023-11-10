import cv2
import numpy as np
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import joblib
from numpy import expand_dims
import os


# Function to extract face from a given photograph using Haar Cascade and resize it
def extract_face(filename, required_size=(160, 160)):
    image = cv2.imread(filename)
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(160, 160))

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    face = image[y:y+h, x:x+w]
    face = cv2.resize(face, required_size)
    return face

# Load the FaceNet model
MyFaceNet = FaceNet()

# Load the trained SVM model and label encoder
model = joblib.load('svm_model_7expressions.pkl')
label_encoder = joblib.load('label_encoder_7expressions.pkl')

# Load and extract faces from the new data folder
new_data_folder = "real"
new_faces = []
for filename in os.listdir(new_data_folder):
    path = os.path.join(new_data_folder, filename)
    face = extract_face(path)
    if face is not None:
        new_faces.append(face)

# Convert faces to embeddings
new_faces_embeddings = [MyFaceNet.embeddings(expand_dims(face, axis=0))[0] for face in new_faces]

# Predict labels using the trained model
predicted_labels = model.predict(new_faces_embeddings)

# Decode numerical labels to original string labels
predicted_labels_decoded = label_encoder.inverse_transform(predicted_labels)

# Print the predicted labels for each face
for filename, label in zip(os.listdir(new_data_folder), predicted_labels_decoded):
    print(f"{filename}: Predicted Expression - {label}")
