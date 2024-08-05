#keras model system for deep learning, was used by google and upgraded
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Data set for ground truth numbers
# mnist national institute of standard and technology
from tensorflow.keras.datasets import mnist

# the shape of the matrix(3D) called tensor, generally the tensor flow model get 4d shape, the 4th dimension is the
# batch dimension
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# plot the grand truth numbers
#plt.subplots(7, 5)

#fig, axes = plt.subplots(7, 5, figsize=(10, 14))
#fig.subplots_adjust(hspace=0.5)

#for i in range(7):
#    for j in range(5):
#        index = i * 5 + j  # Calculate the index for the images and labels
#        if index < len(train_images):
#            ax = axes[i, j]
#            ax.imshow(train_images[index], cmap='gray')
#            ax.set_title(f'label = {str(train_labels[index])}')
#            ax.axis('off')  # Turn off the axis

#plt.show()

# the simplest way to build a model Keras Sequential
# build all layers as one heap upon other, sequentially


# the idea behind the relu:
# when we have a deep learning network with many layers
# we are trying to teach the network
# we can understand from the background, uses gradient descent
# we can tell about the values of the derivative of sigmoid function
# when z is very larg g(z) ~ 1
# when z is close to zero g(z) ~ 0
# the derivative of the function when z is large or negative but far from 0
# the derivative value of g`(z)  is close to 0
# we can't teach the network when we are getting values that are 0's
# because we are using g`(z) in the back propagation, so the values of the derivative
# are so small
# this is called fading out, the network can't learn when values are so close to 0
# network can't be improved if it's not learning


# the first layer is the Dense layer, all the layers in the input layer
# are connected to the 512 neurons of the first hidden layer
# activation layer used for different purposes
model = keras.Sequential(
    [
        layers.Dense(512, activation='relu'),
        layers.Dense(10, activation='softmax')
    ]
)
# The output layer have 10 neurons, since we have 10 digits (0...9) to learn
# this network have 3 layers, input, hidden, and output

# the number of parameters learned in this network is ( 28*28 + 1(bias) ) * 512 + (512 + 1) * 10 ~ 0.5 million
# parameters


# we will initialize the model with parameters
model.compile(
    optimizer='rmsprop',  # how to learn
    loss='sparse_categorical_crossentropy', #price function
    metrics=['accuracy']  #
)

# pre-processing
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# one epoch is when the network go through all the data
# usually we need 10 epochs for this training set

# size of the batch in feed forward and the back propagation in each iteration ( same as mini batch)
model.fit(train_images, train_labels, epochs=10, batch_size=64)

# test_images = test_images[:10]
predictions = model.predict(test_images)


for i in range(10):
    print(f'Predicted label: {np.argmax(predictions[i])}, Actual label: {test_labels[i]}')

test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print('test acc = ', test_accuracy)
