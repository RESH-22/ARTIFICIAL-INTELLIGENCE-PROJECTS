from numpy import loadtxt
from tensorflow.keras.models import model_from_json

# Load dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')

X = dataset[:,0:8]
Y = dataset[:,8]

# Load model architecture
with open("model.json","r") as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json)

# Load weights
model.load_weights("model.h5")

print("Loaded model from disk")

# Compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Predict
predictions = model.predict(X)

# Convert to 0/1
predictions = (predictions > 0.5).astype(int)

# Show output
for i in range(5,10):
    print(X[i].tolist(), "=> Predicted =", predictions[i][0], "Expected =", int(Y[i]))