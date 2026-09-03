import numpy as np


class linear_reg:
    def __init__(self, learning_rate, n_iters):
        self.learning_rate = learning_rate
        self.n_iters = n_iters

    def fit(self, x, y):
        self.m, self.n = x.shape
        self.weights = np.zeros(self.n)
        self.bias = 0
        self.x = x
        self.y = y

        for _ in range(self.n_iters):
            self.update_weights()

    def update_weights(self):
        y_prediction = self.predict(self.x)
        dw = -(2 * self.x.T.dot(self.y - y_prediction)) / self.m
        db = -2 * np.sum(self.y - y_prediction) / self.m
        self.weights = self.weights - self.learning_rate * dw
        self.bias = self.bias - self.learning_rate * db

    def predict(self, x):
        return x.dot(self.weights) + self.bias


import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


salary_data = pd.read_csv("salary_experience_dataset.csv") #your file name
x = salary_data.iloc[:, :-1].values
y = salary_data.iloc[:, 1].values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=22
)

model = linear_reg(learning_rate=0.01, n_iters=1000)
model.fit(x_train, y_train)

print("weights : ", model.weights[0], "bias : ", model.bias)

test_data_prediction = model.predict(x_test)
print(test_data_prediction)

plt.scatter(x_train, y_train, color="red")
plt.plot(x_test, test_data_prediction, color="blue")
plt.title("Salary vs Experience (Training set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
