<img width="626" height="907" alt="image" src="https://github.com/user-attachments/assets/40df77d1-be8e-43d5-aefc-a9a3ad7d7392" /># Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
step 1:
Start
step 2:
Input the mathematical problem (equation, expression, or data).
step 3:
Preprocess the input – identify numbers, variables, and operators.
step 4:
Select the suitable method (rule-based method, formula, or AI model).
step 5:
Apply the algorithm/model to compute the result.
step 6:
Generate the solution step-by-step.
step 7:
Verify the result for correctness.
step 8:
Display the final answer to the user.
step 9:
Stop

## Program:
```
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split

# load the boston dataset manually
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

X = data
y = target

# splitting X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1
)

# create linear regression object
reg = linear_model.LinearRegression()

# train the model
reg.fit(X_train, y_train)

# regression coefficients
print('Coefficients: \n', reg.coef_)

# variance score
print('Variance score: {}'.format(reg.score(X_test, y_test)))

# plot for residual error
plt.style.use('fivethirtyeight')

# training data residuals
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train,
            color="green", s=10, label='Train data')

# test data residuals
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test,
            color="blue", s=10, label='Test data')

# zero error line
plt.hlines(y=0, xmin=0, xmax=50, linewidth=2)

plt.legend(loc='upper right')
plt.title("Residual errors")
plt.show()






```
## Output:
<img width="626" height="907" alt="image" src="https://github.com/user-attachments/assets/b1442f28-ca42-4f8d-8292-bee62666bca2" />

### Insert your output

<br>

## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
