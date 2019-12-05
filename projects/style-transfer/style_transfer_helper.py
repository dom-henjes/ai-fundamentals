# Imports
import numpy as np
from PIL import Image
from io import BytesIO

from keras import backend

# Converts image to array and removes mean values for each colour channel
def convert_and_normalise_image(image, mean_values = [0,0,0]):

    #Convert the content image to an array using NumPy
    image_array = np.asarray(image, dtype="float32")

    #Expand the shape of the array, Insert a new axis that will appear at the axis position in the expanded array shape, 
    #so that we can later concatenate the representations of these two images into a common data structure
    image_array = np.expand_dims(image_array, axis=0)

    #Now we need to compress the input data by performing two transformations
    #1. Subtracting the RGB mean value from each pixel
    #2. Changing the ordering of array from RGB to BGR 

    image_array[:, :, :, 0] -= mean_values[2]
    image_array[:, :, :, 1] -= mean_values[1]
    image_array[:, :, :, 2] -= mean_values[0]
    image_array = image_array[:, :, :, ::-1]

    return image_array


# Helper function to let us get several images easily
def fetch_image(path, dimensions):
    IMAGE_HEIGHT, IMAGE_WIDTH = dimensions
    # Open the image by filepath/URL
    image = Image.open(BytesIO(requests.get(path).content))
    # Resize the images passing in the width and height into the .resize() function
    image = image.resize((IMAGE_HEIGHT, IMAGE_WIDTH))
    return image

# Calculate gram matrix of x
def gram_matrix(x):
    # Turn a nD tensor into a 2D tensor with same 0th dimension. In other words, it flattens each data samples of a batch.
    features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
    
    # .dot multiplies 2 tensors (and/or variables) and returns a tensor
    gram = backend.dot(features, backend.transpose(features))
    return gram
