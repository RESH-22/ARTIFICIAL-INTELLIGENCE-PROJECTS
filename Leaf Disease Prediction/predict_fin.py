# -*- coding: utf-8 -*-
# Leaf Disease Prediction

import numpy as np
import cv2
from tensorflow.keras.models import model_from_json

# ---------------------------
# Load Model
# ---------------------------

with open('model1.json', 'r') as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json)
model.load_weights("model1.h5")

print("Model loaded successfully")

# ---------------------------
# Labels
# ---------------------------

labels = [
"Apple___Apple_scab",
"Apple___Black_rot",
"Apple___Cedar_apple_rust",
"Apple___Healthy",
"Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
"Corn_(maize)___Common_rust_",
"Corn_(maize)___Healthy",
"Corn_(maize)___Northern_Leaf_Blight",
"Grape___Black_rot",
"Grape___Esca_(Black_Measles)",
"Grape___Healthy",
"Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
"Potato___Early_blight",
"Potato___Healthy",
"Potato___Late_blight",
"Tomato___Bacterial_spot",
"Tomato___Early_blight",
"Tomato___Healthy",
"Tomato___Late_blight",
"Tomato___Leaf_Mold",
"Tomato___Septoria_leaf_spot",
"Tomato___Spider_mites Two-spotted_spider_mite",
"Tomato___Target_Spot",
"Tomato___Tomato_Yellow_Leaf_Curl_Virus",
"Tomato___Tomato_mosaic_virus"
]

# ---------------------------
# Load Test Image
# ---------------------------

img_path = "im_for_testing_purpose/imagea.png"

# read image
img = cv2.imread(img_path)

if img is None:
    print("Error: Image not found")
    exit()

# convert color (BGR → RGB)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# resize to model size
img = cv2.resize(img, (128,128))

# convert to float
img = img.astype("float32")

# normalize pixel values
img = img / 255.0

# expand dimension for batch
img = np.expand_dims(img, axis=0)

# ---------------------------
# Prediction
# ---------------------------

result = model.predict(img)[0]

# sort probabilities (top predictions)
top_indices = result.argsort()[-3:][::-1]

print("\nTop Predictions:")
print("---------------------")

for i in top_indices:
    print(labels[i].replace("_"," "), ":", round(result[i]*100,2), "%")

# best prediction
predicted_index = np.argmax(result)
predicted_label = labels[predicted_index]
confidence = result[predicted_index] * 100

# ---------------------------
# Output
# ---------------------------

print("\nFinal Prediction:")
print("---------------------")
print("Disease:", predicted_label.replace("_"," "))
print("Confidence:", round(confidence,2), "%")