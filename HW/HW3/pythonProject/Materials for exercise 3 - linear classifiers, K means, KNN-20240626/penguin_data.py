# -*- coding: utf-8 -*-
"""
Created on Sat May 21 23:59:03 2022

@author: User
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})
penguins = pd.read_csv(
    "https://github.com/allisonhorst/palmerpenguins/raw/5b5891f01b52ae26ad8cb9755ec93672f49328a8/data/penguins_size.csv")

penguins = penguins.dropna()
penguins.species_short.value_counts()
penguins.head()
val = penguins['species_short'].value_counts()

sns.pairplot(penguins, hue='species_short')
penguin_data = penguins[
    [
        "culmen_length_mm",
        "culmen_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]
].values
scaled_penguin_data = StandardScaler().fit_transform(penguin_data)
y_labels = penguins.species_short.map({"Adelie": 0, "Chinstrap": 1, "Gentoo": 2})
y = y_labels.to_numpy()
y1 = y.reshape(y.shape[0], 1)