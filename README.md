### Project Log

2023/12/22

Wrote a polynomial regression model for the dataset. Results not entirely accurate. Mostly due to model not fitting well because of non-linear/non-polynomial nature of dataset. Moving on to trying logistic regression model.

Just saw mention of a technique termed **multinomial classification**. Worth looking into as it might be relevant for our project. Will try to pursue this model if logistic regression does not work as planned.

**KEY NOTE: As of right now all code is using the "diabetes_012_health_indicators_BRFSS2015.csv" dataset. Other two datasets yet to be investigated and implemented.**

Seems like logistic regression is best used to predict some type of binary outcome. If I modify the dataset such that the MentHlth column only contains values of 0 and 1 (representing the person not having and having mental disorders respectively), perhaps it would work.

Working on logistic regression model rn. Issue is that the x-values apparently does not have "valid feature names", whatever that means. Need to work towards figuring out that error. SO CLOSE!

Keep in mind that the file "newDataset.csv" under the "datasets" directory can as of right not be solely used for the second study ("nhsdsc_study2_logistic_regression.py"). 

2023/12/26

Code now works on a smaller dataset if only 100000 entries. Next steps are trying to write code to create a smaller, randomized dataset from the big, original dataset (diabetes_012_health_indicators_BRFSS2015.csv). 

The original problems seems to be that a dataset with 250000+ entries was being problematic when the program tried to scale the dataframe into arrays of higher dimensions for training. The limited computing power we possessed yielded an "not enough memory" error, which I believe refers to RAM. Hence, by making the dataset smaller it may be easier to process. Splitting the dataset into xtrain, ytrain, xvalidation, and yvalidation sections also seem to be of great help for training a more accurate model.