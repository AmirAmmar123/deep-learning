import math
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def load(file_path='./DATA/kleibers_law_data.csv'):
    return pd.read_csv(file_path)

def plot(X, y, x_title, y_title, y_pred=None):
    plt.figure(figsize=(23, 5))
    plt.scatter(X, y, color='black', marker='.')
    if y_pred is not None:
        plt.plot(X, y_pred, color='red', label='Fitted Line')
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title('Linear Regression Fit')
    plt.legend()
    plt.show()

def metabolic_rate_log(mass, theta):
    log_mass = math.log10(mass)
    return theta[0, 0] + theta[1, 0] * log_mass

def LR(X, y):
    model = LinearRegression()
    X_log = np.log(X)
    y_log = np.log(y)

    model.fit(X_log, y_log)

    print(f"Coefficient: {model.coef_[0]}")
    print(f"Intercept: {model.intercept_}")

    y_pred_log = model.predict(X_log)
    plot(X_log, y_log, 'log of mass', 'log of metabolic rate', y_pred_log)

    return np.array([[model.intercept_[0]], [model.coef_[0][0]]])

def joules_to_calories(joules):
    return joules / 4.18

def calories_to_joules(calories):
    return calories * 4.18

if __name__ == "__main__":
    df = load()
    # Prepare the features (X) and the target (y)
    X = df[['mass']].values
    y = df[['metabolic_rate']].values

    plot(np.log(X), np.log(y), 'log of mass', 'log of metabolic rate')
    theta = LR(X, y)

    # Calculate metabolic rate for a 250 kg mammal in Joules
    mass_250_kg = 250
    log_metabolic_rate_250_kg = metabolic_rate_log(mass_250_kg, theta)
    metabolic_rate_250_kg = math.pow(10, log_metabolic_rate_250_kg)  # Joules per day

    # Convert to Calories
    calories_250_kg = joules_to_calories(metabolic_rate_250_kg)
    print(f'Metabolic rate for a 250 kg mammal: {metabolic_rate_250_kg:.2f} Joules/day')
    print(f'Caloric intake for a 250 kg mammal: {calories_250_kg:.2f} Calories/day')

    # Calculate the mass for a metabolic rate of 3.5 kJoules/day
    metabolic_rate_target_joules = 3.5 * 1000  # convert kJoules to Joules
    log_metabolic_rate_target = math.log10(metabolic_rate_target_joules)
    mass_for_target_metabolic_rate = math.pow(10, (log_metabolic_rate_target - theta[0, 0]) / theta[1, 0])
    print(f'Mass for a mammal with a metabolic rate of 3.5 kJoules/day: {mass_for_target_metabolic_rate:.2f} kg')
