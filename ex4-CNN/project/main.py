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


# for i in range(35):
#     plt.figure(1, figsize=(9, 3))
#     plt.imshow(train_images[i], cmap='gray')
#     plt.suptitle(f'label ={str(train_labels[i])}')
#     plt.show()
#     print('label =', train_labels[i])
#     plt.pause(0.5)

model = keras.Sequential(
    [
        layers.Dense(512, activation='relu'),
        layers.Dense(10, activation='softmax')
    ]
)

model.compile(
    optimizer='rmsprop',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# pre-processing
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

model.fit(train_images, train_labels, epochs=10, batch_size=64)

# test_images = test_images[:10]
predictions = model.predict(test_images)

for i in range(10):
    print(f'Predicted label: {np.argmax(predictions[i])}, Actual label: {test_labels[i]}')


test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print('test acc = ', test_accuracy)

