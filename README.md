# Decodr-Machine-Learning-Project
# Introduction:

I will try to show how different models can improve just by doing simple process on the data .

we are going to work on binary classification problem, where we got some information about sample of peoples , and we need to predict whether we should give some one a loan or not depending on his information . we actually have a few sample size (614 rows), so we will go with machine learning techniques to solve our problem .

# Learning Objective

   basics of visualizing the data .

   how to compare between feature importance (at less in this data) .

   1) feature selection

   2) feature engineer

   some simple techniques to process the data .
   handling missing data .
   how to deal with categorical and numerical data .
   outliers data detection
   but the most important thing that everyone can learn , is how to evaluate your model at every step you take .

# What the project is using

   some important libraries like sklearn, matplotlib, numpy, pandas, seaborn, scipy

   fill the values using backward 'bfill' method for numerical columns , and most frequent value for categorical columns (simple techniques)

   4 different models to train your data, so we can compare between them

   a) logistic regression

   b) KNeighborsClassifier

   C) SVC

   d) DecisionTreeClassifier
    
 # Result Achived:
 
  LogisticRegression:
    
    Precision Score: 0.895
    Recall Score: 0.447
    F1 Score: 0.596
    Log Loss: 6.458
    Accuracy Score: 0.813
----------------------------------------
KNeighborsClassifier:
    
    Precision Score: 0.647
    Recall Score: 0.289
    F1 Score: 0.400
    Log Loss: 9.267
    Accuracy Score: 0.732
----------------------------------------
SVC:
    
    Precision Score: 0.895
    Recall Score: 0.447
    F1 Score: 0.596
    Log Loss: 6.458
    Accuracy Score: 0.813
----------------------------------------
DecisionTreeClassifier:
    
    Precision Score: 0.895
    Recall Score: 0.447
    F1 Score: 0.596
    Log Loss: 6.458
    Accuracy Score: 0.813
