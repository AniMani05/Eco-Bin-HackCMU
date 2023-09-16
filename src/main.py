import joblib
from PIL import Image
import numpy as np
import imageio
from skimage.io import imread, imshow
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage import data, exposure
from skimage.feature import corner_harris, corner_peaks
# Load the trained model
loaded_model = joblib.load('/Users/mehulgoel/Documents/HACKS-CMU_2023/Eco-Bin/src/random_forest_model.pkl')  # Replace with the actual filename

# Load and preprocess the image
image_path = 'Eco-Bin/src/sample_img.webp'  # Replace with the actual image path
# image = Image.open(image_path)  # Load the image using PIL
# image = image.convert('L')
# # Define the target size for the input image
# target_size = (128, 128)

# # Resize the image to the target size
# image = image.resize(target_size, Image.ANTIALIAS)  # Use Image.ANTIALIAS for high-quality resizing
# # Convert the resized PIL image to a NumPy array
# image_array = np.array(image)

# # Check the shape of the image array
# print("Image Shape:", image_array.shape)

# # Reshape the image to 1D array if needed

img = imageio.imread(image_path)
grey_image = rgb2gray(img)
image_resized = resize(grey_image, (128,128), anti_aliasing=True)
corner_image = corner_harris(image_resized)
image_flattened = corner_image.flatten()

image_data = image_flattened.reshape(1, -1).astype(np.float32)  # Reshape to a 1D array

print(image_data.shape)
# Make a prediction
predicted_class = loaded_model.predict(image_data)  # Use the reshaped image for prediction

# Use the predicted_class for further processing or display
print("Predicted Class:", predicted_class)