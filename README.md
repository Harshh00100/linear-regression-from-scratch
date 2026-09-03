# Linear Regression From Scratch

A simple implementation of **Linear Regression from scratch using Python**, without using machine-learning libraries such as Scikit-learn for model training.

The project predicts **salary based on years of experience** using a simple linear regression model.

## 📌 Project Overview

Linear Regression is a supervised machine learning algorithm used to model the relationship between an independent variable and a dependent variable.

In this project:

* **Input (X):** Years of Experience
* **Output (Y):** Salary
* **Model:** Simple Linear Regression
* **Dataset:** Salary vs. Years of Experience

The model follows the equation:

$$
\hat{y} = b_0 + b_1x
$$

where:

* `b0` = Intercept
* `b1` = Slope
* `x` = Years of Experience
* `ŷ` = Predicted Salary

## 📂 Project Structure

```text
linear-regression-from-scratch/
│
├── linear_regression.py
├── salary_experience_dataset.csv
└── README.md
```

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib

## 📊 Dataset

The dataset contains two columns:

| Column            | Description                        |
| ----------------- | ----------------------------------- |
| `YearsExperience` | Number of years of work experience |
| `Salary`          | Corresponding salary               |

Example:

```text
YearsExperience,Salary
1.1,30500
1.3,31500
1.5,33000
2.0,35000
2.2,36500
```

## ⚙️ How It Works

The model calculates the slope and intercept using the least-squares method.

### Slope

$$
b_1 =
\frac{\sum (x_i-\bar{x})(y_i-\bar{y})}
{\sum (x_i-\bar{x})^2}
$$

### Intercept

$$
b_0 = \bar{y} - b_1\bar{x}
$$

The prediction is then calculated as:

$$
\hat{y} = b_0 + b_1x
$$

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/linear-regression-from-scratch.git
```

### 2. Enter the project directory

```bash
cd linear-regression-from-scratch
```

### 3. Install dependencies

```bash
pip install numpy pandas matplotlib
```

### 4. Run the Python program

```bash
python linear_regression.py
```

## 📈 Expected Output

The program trains a linear regression model using the salary dataset and predicts salary based on years of experience.

It can also be used to visualize the relationship between:

**Years of Experience → Salary**

## 🎯 Learning Objectives

This project helps understand:

* How Linear Regression works internally
* Calculation of slope and intercept
* Least-squares method
* Model prediction
* Training a machine learning model without Scikit-learn
* Working with CSV datasets
* Data visualization using Matplotlib

## 🔮 Future Improvements

* Add Mean Squared Error (MSE)
* Add Root Mean Squared Error (RMSE)
* Add R² score
* Add train/test split
* Implement Gradient Descent from scratch
* Create a prediction interface
* Compare the implementation with Scikit-learn

## 👨‍💻 Author

**Harsh Kumar**
Electronics and Telecommunication Engineering Student

## 📄 License

This project is intended for educational and learning purposes.
