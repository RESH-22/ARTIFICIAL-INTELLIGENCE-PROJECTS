from tensorflow.keras.models import model_from_json
import numpy as np
from tensorflow.keras.preprocessing import image

# Load model architecture
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)

# Load weights
model.load_weights("model.h5")

print("Loaded model from disk")

# class labels
classes = ['Joker', 'Thanos']

def classify(img_file):

    test_image = image.load_img(img_file, target_size=(64,64))
    test_image = image.img_to_array(test_image)

    test_image = np.expand_dims(test_image, axis=0)
    test_image = test_image / 255.0

    result = model.predict(test_image)

    probability = result[0][0]

    if probability < 0.5:
        prediction = classes[0]
    else:
        prediction = classes[1]

    print("Prediction Value:", probability)
    print("Predicted Character:", prediction)

# test image
classify("test2.jpg")