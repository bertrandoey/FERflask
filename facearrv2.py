import os
from os.path import isdir
from numpy import savez_compressed, asarray, expand_dims
import cv2
from keras.preprocessing.image import ImageDataGenerator

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

    # Load images and extract faces for all images in a directory with data augmentation
    def load_faces(self, directory, augment=True):
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

                # Data augmentation
                if augment:
                    # Reshape to (1, height, width, channels) as required by the generator
                    face = expand_dims(face, axis=0)

                    # Create an ImageDataGenerator with augmentation parameters
                    datagen = ImageDataGenerator(
                        rotation_range=20,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.2,
                        zoom_range=0.2,
                        fill_mode='nearest'
                    )

                    # Generate augmented images
                    augmented_faces = datagen.flow(face, batch_size=1)

                    # Take the first augmented face
                    augmented_face = augmented_faces.next()[0]

                    # Append the augmented face to the list
                    faces.append(augmented_face)

        return faces

    # Load a dataset that contains one subdirectory for each class that contains images
    def load_dataset(self, augment=True):
        X, y = list(), list()
        # Enumerate folders, one per class
        for subdir in os.listdir(self.directory):
            # Path
            path = os.path.join(self.directory, subdir)
            # Skip any files that might be in the directory
            if not isdir(path):
                continue
            # Load all faces in the subdirectory
            faces = self.load_faces(path, augment)
            # Create labels
            labels = [subdir for _ in range(len(faces))]
            # Summarize progress
            print(f'> Loaded {len(faces)} examples for expression: {subdir}')
            # Store
            X.extend(faces)
            y.extend(labels)
        return asarray(X), asarray(y)

# Instantiate the class
faceloading = FACELOADING("expression dataset CK+")

# Load the dataset with data augmentation
X, Y = faceloading.load_dataset(augment=True)

# Save arrays to one file in compressed format
savez_compressed('EDCK+.npz', X, Y)
