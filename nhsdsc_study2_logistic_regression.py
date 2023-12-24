import pandas
import matplotlib.pyplot as plt

dtst_dir = "datasets\diabetes_012_health_indicators_BRFSS2015.csv"

dtst = pandas.read_csv(dtst_dir)

# Modify dataframe such that the MentHlth column only has values of 0 or 1 (representing person having mental health disorder or not respectively)
# IMPORTANT: Code below for creating new dataset is not perfect, does not create perfect new dataset. First column of new CSV file needs to be deleted for model to work.

'''
for i in range(253679):
    mentalHealthValue = dtst["MentHlth"][i]

    if(mentalHealthValue != 0):
        dtst["MentHlth"][i] = 1

    else:
        dtst["MentHlth"][i] = 0

dtst.to_csv("dataset/newDataset.csv")
'''

newDatasetDir = "datasets/newDataset.csv"
dtst = pandas.read_csv(newDatasetDir)

factors = ["Diabetes_012"]
x = dtst[factors]
y = dtst["MentHlth"]

# Code for splitting dataset into testing and training sections.
'''
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=16)
'''

# Time for the real deal. Initializing model and fitting it.

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=16)
model.fit(x, y)
pred = model.predict([[2]])

print(pred)