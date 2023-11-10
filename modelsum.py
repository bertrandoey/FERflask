from keras.models import Model
from keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, Lambda, concatenate
from keras_facenet import FaceNet

# Load the FaceNet model
MyFaceNet = FaceNet()
faceNet_input = MyFaceNet.model.input
faceNet_output = MyFaceNet.model.layers[-2].output  # Extracting the output before the bottleneck layer

# Add your custom layers
custom_layers = GlobalAveragePooling2D()(faceNet_output)
custom_layers = Reshape((1, 1, -1))(custom_layers)  # Reshape to (1, 1, 512)
custom_layers = Dropout(0.5)(custom_layers)
custom_layers = Dense(512, activation='relu')(custom_layers)

# Concatenate the outputs of FaceNet and your custom model
combined_output = concatenate([faceNet_output, custom_layers])

# Add additional layers or modify the combined model as needed
# ...

# Create the final combined model
final_model = Model(inputs=faceNet_input, outputs=combined_output)

# Display the summary of the combined model
final_model.summary()
