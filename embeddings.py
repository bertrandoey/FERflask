from numpy import asarray, savez_compressed, load, expand_dims
from keras_facenet import FaceNet

# Load the FaceNet model
MyFaceNet = FaceNet()

# Load the face dataset
data = load('CK+LOWEST.npz')
trainX, trainy = data['arr_0'], data['arr_1']

# Convert each face in the train set to an embedding
newTrainX = [MyFaceNet.embeddings(expand_dims(face, axis=0))[0] for face in trainX]

# Save arrays to one file in compressed format
savez_compressed('CK+LOWESTembeddings.npz', newTrainX, trainy)


