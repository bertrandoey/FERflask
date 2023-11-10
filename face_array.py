import os
from os.path import isdir
from numpy import savez_compressed, asarray
import cv2

class FACELOADING:
    def __init__(self, directory):
        self.directory = directory

    # Function to extract a single face from a given photograph using Haar Cascade and resize it
    def extract_face(self, filename, required_size=(160, 160)):
        # Load image from file
        image = cv2.imread(filename)
        if image is None:
            return None
        # Convert to grayscale for Haar Cascade
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Load Haar Cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(160, 160))

        if len(faces) == 0:
            return None

        # Extract the first detected face
        x, y, w, h = faces[0]
        face = image[y:y+h, x:x+w]
        # Resize pixels to the model size
        face = cv2.resize(face, required_size)
        return face

    # Load images and extract faces for all images in a directory
    def load_faces(self, directory):
        faces = list()
        # Enumerate files
        for filename in os.listdir(directory):
            # Path
            path = os.path.join(directory, filename)
            # Get face
            face = self.extract_face(path)
            if face is not None:
                # Convert face to array
                faces.append(face)
        return faces

    # Load a dataset that contains one subdirectory for each class that contains images
    def load_dataset(self):
        X, y = list(), list()
        # Enumerate folders, one per class
        for subdir in os.listdir(self.directory):
            # Path
            path = os.path.join(self.directory, subdir)
            # Skip any files that might be in the directory
            if not isdir(path):
                continue
            # Load all faces in the subdirectory
            faces = self.load_faces(path)
            # Create labels
            labels = [subdir for _ in range(len(faces))]
            # Summarize progress
            print(f'> Loaded {len(faces)} examples for expression: {subdir}')
            # Store
            X.extend(faces)
            y.extend(labels)
        return asarray(X), asarray(y)

# Instantiate the class
faceloading = FACELOADING("ck low")

# Load the dataset
X, Y = faceloading.load_dataset()

# Save arrays to one file in compressed format
savez_compressed('ck_75fear.npz', X, Y)
