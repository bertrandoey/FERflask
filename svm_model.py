from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from numpy import load
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the face embeddings and labels for seven expressions
data = load('CK+LOWESTembeddings.npz')
embeddings, labels = data['arr_0'], data['arr_1']

# Label encode targets for multi-class classification
out_encoder = LabelEncoder()
labels_encoded = out_encoder.fit_transform(labels)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(embeddings, labels_encoded, test_size=0.2, shuffle=True)

# Fit the SVM model with class weights to the training data
model = SVC(kernel='poly', C=1 , probability=True, class_weight='balanced')
model.fit(X_train, Y_train)

# Evaluate the model on the testing data (optional)
accuracyTest = model.score(X_test, Y_test)
print(f'Model Accuracy on Test Data: {accuracyTest * 100:.2f}%')
accuracyTrain = model.score(X_train, Y_train)
print(f'Model Accuracy on Train Data: {accuracyTrain * 100:.2f}%')

# Save the trained model and label encoder to files
joblib.dump(model, 'svm_model_7expressions.pkl')
joblib.dump(out_encoder, 'label_encoder_7expressions.pkl')

# Predict the labels for the test data
Y_pred = model.predict(X_test)

# Create the confusion matrix
conf_matrix = confusion_matrix(Y_test, Y_pred)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=out_encoder.classes_, yticklabels=out_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Calculate precision, recall, and F1 score for each label
precision_scores = precision_score(Y_test, Y_pred, average=None)
recall_scores = recall_score(Y_test, Y_pred, average=None)
f1_scores = f1_score(Y_test, Y_pred, average=None)

# Create a DataFrame to store the metrics
metrics_df = pd.DataFrame({
    'Label': out_encoder.classes_,
    'Precision': precision_scores,
    'Recall': recall_scores,
    'F1 Score': f1_scores
})

# Display the DataFrame
print(metrics_df)

