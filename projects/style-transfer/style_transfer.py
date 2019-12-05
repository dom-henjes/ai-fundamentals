# Imports
import numpy as np
from PIL import Image
import requests
from io import BytesIO

from keras import backend
from keras.models import Model
from keras.applications.vgg16 import VGG16

from scipy.optimize import fmin_l_bfgs_b

# Own package
from style_transfer_helper import fetch_image, convert_and_normalise_image, gram_matrix

# CONSTANTS
IMAGE_HEIGHT = 500
IMAGE_WIDTH = 500
CHANNELS = 3
IMAGENET_MEAN_RGB_VALUES = [123.68, 116.779, 103.939]


# PARAMETERS
STYLE_WEIGHT = 4.5
CONTENT_WEIGHT = 0.02
TOTAL_VARIATION_WEIGHT = 0.995
TOTAL_VARIATION_LOSS_FACTOR = 1.25

# Input images
content_image_path = "https://ai-camp-content.s3.amazonaws.com/inputImage.jpg"
style_image_path = "https://ai-camp-content.s3.amazonaws.com/styleImage.jpg"


# Use helper function to load & resize images
input_image = fetch_image(content_image_path, (IMAGE_HEIGHT, IMAGE_WIDTH))
style_image = fetch_image(style_image_path, (IMAGE_HEIGHT, IMAGE_WIDTH))

# Save the newly sized images
input_image.save('contentImage.png')
style_image.save('styleImage.png')

# Data normalization and reshaping from RGB to BGR to convert images into a suitable form for processing
input_image_array = convert_and_normalise_image(input_image, IMAGENET_MEAN_RGB_VALUES)
style_image_array = convert_and_normalise_image(style_image, IMAGENET_MEAN_RGB_VALUES)

# Add the content images as backend variables, so that tensorflow can process them
input_image = backend.variable(input_image_array)
style_image = backend.variable(style_image_array)

# Instantiate a placeholder tensor to store the combination image (i.e. the output image)
# The combination image will keep the content of the content image while adding the style of the style image.
combination_image = backend.placeholder(shape=(1, IMAGE_HEIGHT, IMAGE_WIDTH, 3))

# Concatenates the context, style and combination image on the specified axis 0 
input_tensor = backend.concatenate([input_image,style_image,combination_image], axis=0)

# We're using the pre-trained VGG16 model (16 layer model), which Keras offers us to
# work with. It is a convolutional neural network trained on the ImageNet data set.
# `include_top` is a parameter that controls whether to include the 3 fully-connected
# layers at the top of the network, and since we are not interested in image
# classification we set this value to false
model = VGG16(input_tensor=input_tensor, include_top=False)

# Construct a dictionary to easily look up layers by their names
layers = dict([(layer.name, layer.output) for layer in model.layers])

# For the content loss, we draw the content feature from the block2_conv2 layer.
content_layer = "block2_conv2"
layer_features = layers[content_layer]

# Remember that we concatenated along axis 0, in the order:
# content image, style image, combination image
content_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]

# The content loss is the squared Euclidean distance between content and combination images.
CONTENT_LOSS = backend.sum(backend.square(combination_features - content_image_features))
loss = backend.variable(0.)
loss += CONTENT_WEIGHT * CONTENT_LOSS

def compute_style_loss(style, combination):
    style = gram_matrix(style)
    combination = gram_matrix(combination)
    size = IMAGE_HEIGHT * IMAGE_WIDTH
    return backend.sum(backend.square(style - combination)) / (4. * (CHANNELS ** 2) * (size ** 2))
  
# The style layers that we are interested in
style_layers = ["block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3", "block5_conv3"]

for layer_name in style_layers:
    layer_features = layers[layer_name]
    style_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    style_loss = compute_style_loss(style_features, combination_features)
    loss += (STYLE_WEIGHT / len(style_layers)) * style_loss

def total_variation_loss(x):
    a = backend.square(x[:, :IMAGE_HEIGHT-1, :IMAGE_WIDTH-1, :] - x[:, 1:, :IMAGE_WIDTH-1, :])
    b = backend.square(x[:, :IMAGE_HEIGHT-1, :IMAGE_WIDTH-1, :] - x[:, :IMAGE_HEIGHT-1, 1:, :])
    return backend.sum(backend.pow(a + b, TOTAL_VARIATION_LOSS_FACTOR))

loss += TOTAL_VARIATION_WEIGHT * total_variation_loss(combination_image)

outputs = [loss]

# Now we have our total loss , its time to optimize the resultant image
# We start by defining gradients - this is used for back propagation, 
# which is also known as 'gradient descent'
outputs += backend.gradients(loss, combination_image)

def evaluate_loss_and_gradients(x):
    x = x.reshape((1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    outs = backend.function([combination_image], outputs)([x])
    loss = outs[0]
    gradients = outs[1].flatten().astype("float64")
    return loss, gradients

class Evaluator:

    def loss(self, x):
        loss, gradients = evaluate_loss_and_gradients(x)
        self._gradients = gradients
        return loss

    def gradients(self, x):
        return self._gradients

evaluator = Evaluator()

# initialise our output image with a random distribution of pixels, with an average
# colour value of 0
x = np.random.uniform(0, 255, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)) - 128.

iterations = 5

# This resultant image is initially a random collection of pixels, so we use 
# fmin_l_bfgs_b - limited-memory BFGS which is an optimization algorithm
for i in range(iterations):
    x, loss, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.gradients, maxfun=20)
    print("Iteration %d completed with loss %d" % (i, loss))
    
# To get back output image do the following
x = x.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
x = x[:, :, ::-1]
x[:, :, 0] += IMAGENET_MEAN_RGB_VALUES[2]
x[:, :, 1] += IMAGENET_MEAN_RGB_VALUES[1]
x[:, :, 2] += IMAGENET_MEAN_RGB_VALUES[0]
x = np.clip(x, 0, 255).astype("uint8")
output_image = Image.fromarray(x)
output_image.save("output.png")

# Place the content, style and output beside each other in a combined image
image_collage = Image.new("RGB", (IMAGE_WIDTH*3, IMAGE_HEIGHT))
x_offset = 0
for image in map(Image.open, ['contentImage.png', 'styleImage.png', 'output.png']):
    image_collage.paste(image, (x_offset, 0))
    x_offset += IMAGE_WIDTH
image_collage.save('image_collage.png')
