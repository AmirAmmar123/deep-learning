import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def display_image(vector):
    """
    Reshape a vector into a 28x28 image and display it using matplotlib.

    Parameters:
    vector (numpy.ndarray): A 1D array of 784 elements representing the image.

    """
    # Reshape the vector into a 28x28 image
    image = vector.reshape(28, 28)

    # Display the image using matplotlib
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()



df = pd.read_excel('mat-for-ex4/face_data.xlsx')
M = df.to_numpy()
MT = M.T
X = MT[:-1, 1000:5000]
Y = MT[784, 1000:5000]
Y[Y == -1] = 0
Y = Y.reshape(1, Y.shape[0])


