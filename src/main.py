import joblib
from PIL import Image
import numpy as np

# Load the trained model
loaded_model = joblib.load('/Users/mehulgoel/Documents/HACKS-CMU_2023/Eco-Bin/src/random_forest_model.pkl')  # Replace with the actual filename

# Load and preprocess the image
image_path = 'Eco-Bin/src/garbage-classification/data/plastic100.jpg'  # Replace with the actual image path
image = Image.open(image_path)  # Load the image using PIL
image = image.convert('L')
# Define the target size for the input image
target_size = (128, 128)

# Resize the image to the target size
image = image.resize(target_size)  # Use Image.ANTIALIAS for high-quality resizing
# Convert the resized PIL image to a NumPy array
image_array = np.array(image)

# Check the shape of the image array
print("Image Shape:", image_array.shape)

# Reshape the image to 1D array if needed
image_data = image_array.reshape(1, -1).astype(np.float32)  # Reshape to a 1D array

print(image_data.shape)
# Make a prediction
predicted_class = loaded_model.predict(image_data)  # Use the reshaped image for prediction

# Use the predicted_class for further processing or display
print("Predicted Class:", predicted_class)