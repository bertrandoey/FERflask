from numpy import load, savez_compressed, expand_dims, asarray
import matplotlib.pyplot as plt
from keras_facenet import FaceNet
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator

# Load the FaceNet model
MyFaceNet = FaceNet()

# Load the face dataset
data = load('ck_low.npz')
trainX, trainy = data['arr_0'], data['arr_1']

# Convert each face in the train set to an embedding
newTrainX = [MyFaceNet.embeddings(expand_dims(face, axis=0))[0] for face in trainX]

# Convert to NumPy array
newTrainX = asarray(newTrainX)

# Convert labels to numerical representation
label_encoder = LabelEncoder()
trainy = label_encoder.fit_transform(trainy)

# Ensure 'trainy' is numeric
trainy = trainy.astype('float32')

# Reshape the face embeddings to (num_samples, height, width, channels)
num_samples, embedding_dim = newTrainX.shape
height, width, channels = 1, embedding_dim, 1  # Assuming 1D embeddings, adjust if needed
newTrainX = newTrainX.reshape(num_samples, height, width, channels)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Define the model
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=newTrainX.shape[1]))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

# Output layer for multiclass classification
num_categories = len(label_encoder.classes_)
model.add(Dense(num_categories, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with data augmentation
history = model.fit(
    datagen.flow(newTrainX, trainy, batch_size=32),
    epochs=10,
    validation_split=0.2
)

# Save the trained model
model.save('expression_model.h5')

# Plot training accuracy and loss
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
