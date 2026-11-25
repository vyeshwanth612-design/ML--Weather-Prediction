# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset



## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import NumPy and RandomForestRegressor from sklearn.
2. Create the input data (environmental sensor values) and output data (Temperature, PM2.5 and Energy).
3. Train the Random Forest Regressor using the fit() function.
4. 1. Predict temperature, PM2.5 and energy using the predict() function and display the results.

## Program:
```py

import numpy as np
from sklearn.ensemble import RandomForestRegressor

X = np.array([
    [25, 60, 10, 2],
    [27, 55, 15, 3],
    [30, 50, 20, 4],
    [32, 45, 25, 5],
    [29, 65, 18, 3],
    [26, 70, 12, 2],
    [31, 40, 30, 6],
    [33, 35, 35, 7],
    [28, 60, 16, 3],
    [24, 75, 8, 1]
])

Y = np.array([
    [25, 35, 120],
    [27, 40, 130],
    [30, 55, 150],
    [32, 65, 170],
    [29, 45, 140],
    [26, 30, 110],
    [31, 75, 180],
    [33, 85, 200],
    [28, 42, 135],
    [24, 25, 100]
])

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, Y)

prediction = model.predict([[30, 55, 20, 4]])

print("Predicted Temperature:", prediction[0][0])
print("Predicted PM2.5:", prediction[0][1])
print("Predicted Energy:", prediction[0][2])

```

## Output:
<img width="262" height="82" alt="image" src="https://github.com/user-attachments/assets/79396c9c-0305-49f8-9cd7-b48f12d169ec" />



## Result:
Thus, the Random Forest Regression model was successfully implemented to predict daily temperature, PM2.5 pollution level, and energy consumption using environmental sensor data.
