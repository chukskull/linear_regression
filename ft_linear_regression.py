import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

class LinearRegression1337:
    def __init__(self):
        self.theta = np.zeros(2)
        self.iterations = 1000
        self.data = pd.read_csv('data.csv')
        self.alpha = 0.01  # learning rate
        self.m = len(self.data)
        self.x1 = np.array(self.data['km'])
        self.y1 = np.array(self.data['price'])

        self.mean_x1 = np.mean(self.x1)
        self.std_x1 = np.std(self.x1)
        self.mean_y1 = np.mean(self.y1)
        self.std_y1 = np.std(self.y1)

        self.x = (self.x1 - self.mean_x1) / self.std_x1
        self.y = (self.y1 - self.mean_y1) / self.std_y1
        self.X = np.vstack((np.ones(self.m), self.x)).T

    def gradient_descent(self):
        for _ in range(self.iterations):
            y_pred = self.X.dot(self.theta)
            grad = 1 / self.m * self.X.T.dot(y_pred - self.y)
            self.theta -= self.alpha * grad
        return self.theta

    def fit(self):
        self.theta = self.gradient_descent()
        return self.theta

    def predict(self, x):
        #Normalize the data
        x_standardized = (x - self.mean_x1) / self.std_x1
        X_pred = np.vstack((np.ones(len(x_standardized)), x_standardized)).T
        y_pred_standardized = X_pred.dot(self.theta)
        #Reverse normalization for prediction
        return y_pred_standardized * self.std_y1 + self.mean_y1

    def plot_fit(self):
        fig, ax = plt.subplots()
        ax.scatter(self.x1, self.y1, cmap='winter', label='Data points')
        ax.plot(self.x1, self.predict(self.x1), color='red', label='Fitted line')
        ax.set(ylabel='Price', xlabel='Kilometers', title="Ft_linear_regression")
        ax.legend()
        plt.show()

    def mse(self):
        y_pred = self.predict(self.x1)
        mse_value = np.mean((self.y1 - y_pred) ** 2)
        return mse_value

    def r_squared(self):
        y_pred = self.predict(self.x1)
        ss_total = np.sum((self.y1 - np.mean(self.y1)) ** 2)
        ss_residual = np.sum((self.y1 - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        return r2

    def model_accuracy(self):
        r2 = self.r_squared()
        return r2 * 100

    def print_metrics(self):
        mse_value = self.mse()
        r2_value = self.r_squared()
        accuracy = self.model_accuracy()
        print(f"Mean Squared Error (custom): {mse_value}")
        print(f"R-squared (custom): {r2_value}")
        print(f"Model Accuracy (custom): {accuracy}%")

    def get_thatas(self):
        # Calculate original scale coefficients from standardized theta
        theta1_original = self.theta[1] * (self.std_y1 / self.std_x1)
        theta0_original = self.mean_y1 + self.std_y1 * self.theta[0] - theta1_original * self.mean_x1
        return theta0_original, theta1_original

def main():
    model = LinearRegression1337()
    theta = model.fit()
    thetas = model.get_thatas()

    thetas_dict = {"theta0": thetas[0], "theta1": thetas[1]}
    with open("thetas.json", "w") as json_file:
        json.dump(thetas_dict, json_file)
    
    print(f"Fitted coefficients (standardized theta): {theta[0]}, {theta[1]}")
    print(f"Coefficients on real data scale: {thetas[0]}, {thetas[1]}")
    
    model.plot_fit()
    
    predictions = model.predict(np.array(model.data['km']))
    pd.DataFrame(predictions, index=model.data['price'], columns=["Predictions"]).to_csv("predicted.csv")
    
    model.print_metrics()
    
    # Example of making a prediction for a particular km value (e.g., 240000 km)
    print(f"Prediction for 240000 km: {240000 * thetas[1] + thetas[0]}")

if __name__ == '__main__':
    main()