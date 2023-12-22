import pandas
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

dtst_dir = "datasets\diabetes_012_health_indicators_BRFSS2015.csv"

dtst = pandas.read_csv(dtst_dir)

x = dtst[["GenHlth", "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"]]
y = dtst["MentHlth"]

print(x)

x = PolynomialFeatures(degree = 3).fit_transform(x) # Changing degree into 1 results in linear regression model.
model = linear_model.LinearRegression()
model.fit(x, y)

print(model.predict(x))