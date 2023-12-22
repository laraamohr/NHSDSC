import pandas
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

dtst_dir = "datasets\diabetes_012_health_indicators_BRFSS2015.csv"

dtst = pandas.read_csv(dtst_dir)

x = dtst[["GenHlth"]]
y = dtst["MentHlth"]

print(x)

x = PolynomialFeatures(degree = 5).fit_transform(x)
model = linear_model.LinearRegression()
model.fit(x, y)

print(model.predict(x))