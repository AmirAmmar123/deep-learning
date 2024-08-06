# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 14:15:07 2022

@author: 
"""

import tensorflow as tf


fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist
X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]
print("X_train shape =", X_train.shape)
print("type of X_train = ", X_train.dtype)
# scale the pixel intensities down to 0-1 range by dividing them by 255.0
# this also converts them to floats.
X_train, X_valid, X_test = X_train / 255., X_valid / 255., X_test / 255.

# We will use a list of class names for Fashion MNIST to know the classes    
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
# for example the first 5 images represents:
# class_names[y_train[0]]

# presenting some data
# for k in range(10):
#     plt.figure(1, figsize = [5,5])
#     plt.imshow(X_train[k,:,:], cmap = 'gray')
#     plt.suptitle(class_names[y_train[k]])
#     plt.pause(0.5)


if __name__ == '__main__':
    pass

