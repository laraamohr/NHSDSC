import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

dtst_dir = "datasets\diabetes_012_health_indicators_BRFSS2015.csv"

dtst = pandas.read_csv(dtst_dir).dropna()

x = dtst[["GenHlth", "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"]].values
y = dtst["MentHlth"].values

'''
plt.plot(dtst["GenHlth"], dtst["MentHlth"])
plt.show()
'''

print(x[0], y[0])

polyModel = PolynomialFeatures(degree = 3) # Changing degree into 1 results in linear regression model.
polyX = polyModel.fit_transform(x)
polyModel.fit(polyX, y)

linearModel = LinearRegression()
linearModel.fit(polyX, y)

predictions = linearModel.predict(polyX)

print(predictions)
print(mean_squared_error(y, predictions, squared=False))

degrees = [1, 2, 3, 4, 5, 6, 7]
mean_squared_error_list = [6.658825469443437, 6.553170836282355, 6.5089789129822035]

'''
degrees = [1, 2, 3, 4, 5, 6, 7]
averageSquaredError = []

for degree in degrees:

    polyModel = PolynomialFeatures(degree=degree)
    polyX = polyModel.fit_transform(x)
    polyModel.fit(polyX, y)

    linearModel = LinearRegression()
    linearModel.fit(polyX, y)
    predictions = linearModel.predict(polyX)

    averageSquaredError.append(mean_squared_error(y, predictions, squared=False))

plt.scatter(degrees, averageSquaredError, color="green")
plt.plot(degrees, averageSquaredError, color="red")
'''