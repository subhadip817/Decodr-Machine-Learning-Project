# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 05:52:10 2020

@author: Subhadip Samanta
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score,log_loss, accuracy_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import norm

import warnings
warnings.filterwarnings('ignore')

                                #SIMPLE LOOK AT THE DATA

df=pd.read_csv('D:/Data Science using Python/Loan Prediction Model/train_u6lujuX_CVtuZ9i.csv')
print(df.shape)
print(df.head())
# If we have missing data , we will handle them as we go
print(df.info())
# Describe the numerical data
print(df.describe())
#we will change the type of Credit_History to object becaues we can see that it is 1 or 0
df['Credit_History'] = df['Credit_History'].astype('O')
# describe categorical data ("object")
print(df.describe(include='O'))
# we will drop ID because it's not important for our model and it will just mislead the model
df.drop('Loan_ID', axis=1, inplace=True)
# we got no duplicated rows
print(df.duplicated().any())
# let's look at the target percentage
plt.figure(figsize=(8,6))
sns.countplot(df['Loan_Status']);
print('The percentage of Y class : %.2f' % (df['Loan_Status'].value_counts()[0] / len(df)))
print('The percentage of N class : %.2f' % (df['Loan_Status'].value_counts()[1] / len(df)))
print(df.head(1))

                            #DEEP DRIVE INTO THE DATA AND INFERENCES

# Credit_History
grid = sns.FacetGrid(df,col='Loan_Status',size=3.2,aspect=1.6)
grid.map(sns.countplot, 'Credit_History');
# we didn't give a loan for most people who got Credit History = 0
# but we did give a loan for most of people who got Credit History = 1
# so we can say if you got Credit History = 1 , you will have better chance to get a loan
# important feature

# Gender
grid = sns.FacetGrid(df,col='Loan_Status',size=3.2,aspect=1.6)
grid.map(sns.countplot, 'Gender');
# most males got loan and most females got one too so (No pattern)
# i think it's not so important feature, we will see later

# Married
plt.figure(figsize=(15,5))
sns.countplot(x='Married', hue='Loan_Status', data=df)
# most people who get married did get a loan
# if you'r married then you have better chance to get a loan :)
# good feature

# Dependents
plt.figure(figsize=(15,5))
sns.countplot(x='Dependents', hue='Loan_Status', data=df)
# first if Dependents = 0 , we got higher chance to get a loan ((very hight chance))
# good feature

#Education
plt.figure(figsize=(15,5))
sns.countplot(x='Education', hue='Loan_Status', data=df)
# If you are graduated or not, you will get almost the same chance to get a loan (No pattern)
# Here you can see that most people did graduated, and most of them got a loan
# on the other hand, most of people who did't graduate also got a loan, but with less percentage from people who graduated
# not important feature

#Self_Employed
plt.figure(figsize=(15,5))
sns.countplot(x='Self_Employed', hue='Loan_Status', data=df)
# No pattern (same as Education)

# Property_Area
plt.figure(figsize=(15,5))
sns.countplot(x='Property_Area', hue='Loan_Status', data=df)
# We can say, Semiurban Property_Area got more than 50% chance to get a loan
# good feature

# ApplicantIncome
plt.scatter(df['ApplicantIncome'], df['Loan_Status']);
# No pattern

# the numerical data
print(df.groupby('Loan_Status').median())
# median because Not affected with outliers
# we can see that when we got low median in CoapplicantInocme we got Loan_Status = N
# CoapplicantInocme is a good feature

                        #SIMPLE PROCESS FOR THE DATA

#Methods for handling missing values
print(df.isnull().sum().sort_values(ascending=False))

# We will separate the numerical columns from the categorical
cat_data = []
num_data = []

for i,c in enumerate(df.dtypes):
    if c == object:
        cat_data.append(df.iloc[:, i])
    else :
        num_data.append(df.iloc[:, i])
cat_data=pd.DataFrame(cat_data).transpose()
num_data=pd.DataFrame(num_data).transpose()
print(cat_data.head())
print(num_data.head())

# cat_data
# If you want to fill every column with its own most frequent value you can use
cat_data = cat_data.apply(lambda x:x.fillna(x.value_counts().index[0]))
cat_data.isnull().sum().any() 
# no more missing data 

# num_data
# fill every missing value with their previous value in the same column
num_data.fillna(method='bfill', inplace=True)
num_data.isnull().sum().any() 
# no more missing data 

#we are going to use LabelEncoder 
#what it is actually do it encode labels with value between 0 and n_classes-1 
le=LabelEncoder()
print(cat_data.head())

# transform the target column
target_values = {'Y': 0 , 'N' : 1}
target = cat_data['Loan_Status']
cat_data.drop('Loan_Status', axis=1, inplace=True)
target = target.map(target_values)

# transform other columns
for i in cat_data:
    cat_data[i] = le.fit_transform(cat_data[i])
    
print(target.head())
print(cat_data.head())
df=pd.concat([cat_data,num_data,target], axis=1)
print(df.head())


                        #TRAIN THE DATA
                        
X=pd.concat([cat_data,num_data], axis=1)
y=target

# we will use StratifiedShuffleSplit to split the data Taking into 
#consideration that we will get the same ratio on the target column
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train, test in sss.split(X, y):
    X_train, X_test = X.iloc[train], X.iloc[test]
    y_train, y_test = y.iloc[train], y.iloc[test]
print('X_train shape', X_train.shape)
print('y_train shape', y_train.shape)
print('X_test shape', X_test.shape)
print('y_test shape', y_test.shape)

# almost same ratio
print('\nratio of target in y_train :',y_train.value_counts().values/ len(y_train))
print('ratio of target in y_test :',y_test.value_counts().values/ len(y_test))
print('ratio of target in original_data :',df['Loan_Status'].value_counts().values/ len(df))

# we will use 4 different models for training
models = {
    'LogisticRegression': LogisticRegression(random_state=42),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'SVC': SVC(random_state=42),
    'DecisionTreeClassifier': DecisionTreeClassifier(max_depth=1, random_state=42)
}
def loss(y_true, y_pred, retu=False):
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    loss = log_loss(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    
    if retu:
        return pre, rec, f1, loss, acc
    else:
        print('  pre: %.3f\n  rec: %.3f\n  f1: %.3f\n  loss: %.3f\n  acc: %.3f' 
              % (pre, rec, f1, loss, acc))

# train_eval_train
# Evaluating the model by using same dataset in which it is trained
def train_eval_train(models, X, y):
    for name, model in models.items():
        print(name,':')
        model.fit(X, y)
        loss(y, model.predict(X))
        print('-'*30)
        
train_eval_train(models, X_train, y_train)

# train_eval_cross
# Evaluating the model by using different dataset
skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
def train_eval_cross(models, X, y, folds):
    # we will change X & y to dataframe because we will use iloc (iloc don't work on numpy array)
    X = pd.DataFrame(X) 
    y = pd.DataFrame(y)
    idx = [' pre', ' rec', ' f1', ' loss', ' acc']
    for name, model in models.items():
        ls = []
        print(name,':')

        for train, test in folds.split(X, y):
            model.fit(X.iloc[train], y.iloc[train]) 
            y_pred = model.predict(X.iloc[test]) 
            ls.append(loss(y.iloc[test], y_pred, retu=True))
        print(pd.DataFrame(np.array(ls).mean(axis=0), index=idx)[0]) 
        #[0] because we don't want to show the name of the column
        print('-'*30)
        
train_eval_cross(models, X_train, y_train, skf)


                        #IMPROVING OUR MODEL
                        
# ooh, we got it right for most of the features, as you can see we've say at the first of the kernel ,
# that Credit_Histroy and Married etc, are good features, actually Credit_Histroy is the best .
data_corr = pd.concat([X_train, y_train], axis=1)
corr = data_corr.corr()
plt.figure(figsize=(10,7))
sns.heatmap(corr, annot=True)
# here we got 58% similarity between LoanAmount & ApplicantIncome 
# and that may be bad for the model

#Now we will introduce to new columns
X_train['new_col'] = X_train['CoapplicantIncome'] / X_train['ApplicantIncome']  
X_train['new_col_2'] = X_train['LoanAmount'] * X_train['Loan_Amount_Term']
data_corr = pd.concat([X_train, y_train], axis=1)
corr = data_corr.corr()
plt.figure(figsize=(10,7))
sns.heatmap(corr, annot=True)
#Correlation of new_col and new_col_2 with Loan_Status os 0.3 and 0.047 respectively
#So we can successfully omit or drop these 4 fields
X_train.drop(['CoapplicantIncome', 'ApplicantIncome', 'Loan_Amount_Term',
              'LoanAmount'], axis=1, inplace=True)
train_eval_cross(models, X_train, y_train, skf)

# first lets take a look at the value counts of every label
for i in range(X_train.shape[1]):
    print(X_train.iloc[:,i].value_counts(),
          end='\n------------------------------------------------\n')
    
            #WE WILL WORK ON THE FEATURES WHICH HAVE VARIED VALUES

# new_col_2
# we can see we got right_skewed
# we can solve this problem with very simple statistical teqniq , by taking the logarithm of all the values
# because when data is normally distributed that will help improving our model
fig, ax = plt.subplots(1,2,figsize=(10,5))
sns.distplot(X_train['new_col_2'], ax=ax[0], fit=norm)
ax[0].set_title('new_col_2 before log')
X_train['new_col_2'] = np.log(X_train['new_col_2'])  # logarithm of all the values
sns.distplot(X_train['new_col_2'], ax=ax[1], fit=norm)
ax[1].set_title('new_col_2 after log');

# new_col
# most of our data is 0 , so we will try to change other values to 1
print('before:')
print(X_train['new_col'].value_counts())
X_train['new_col'] = [x if x==0 else 1 for x in X_train['new_col']]
print('-'*50)
print('\nafter:')
print(X_train['new_col'].value_counts())

train_eval_cross(models, X_train, y_train, skf)
for i in range(X_train.shape[1]):
    print(X_train.iloc[:,i].value_counts(),
          end='\n------------------------------------------------\n')
    
                        #HANDLING OUTLIERS
                        


# we will use boxplot to detect outliers
#sns.boxplot(X_train['new_col_2']);
plt.title('new_col_2 outliers');
plt.xlabel('');

threshold = 0.1 
# this number is hyper parameter , as much as you reduce it, as much as you remove more points
# you can just try different values the deafult value is (1.5) it works good for most cases
# but be careful, you don't want to try a small number because you may loss some important information from the data .
# that's why I was surprised when 0.1 gived me the best result
            
new_col_2_out = X_train['new_col_2']
q25, q75 = np.percentile(new_col_2_out, 25), np.percentile(new_col_2_out, 75) # Q25, Q75
print('Quartile 25: {} , Quartile 75: {}'.format(q25, q75))

iqr = q75 - q25
print('iqr: {}'.format(iqr))

cut = iqr * threshold
lower, upper = q25 - cut, q75 + cut
print('Cut Off: {}'.format(cut))
print('Lower: {}'.format(lower))
print('Upper: {}'.format(upper))

outliers = [x for x in new_col_2_out if x < lower or x > upper]
print('Nubers of Outliers: {}'.format(len(outliers)))
print('outliers:{}'.format(outliers))

data_outliers = pd.concat([X_train, y_train], axis=1)
print('\nlen X_train before dropping the outliers', len(data_outliers))
data_outliers = data_outliers.drop(data_outliers[(data_outliers['new_col_2'] > upper) | (data_outliers['new_col_2'] < lower)].index)

print('len X_train before dropping the outliers', len(data_outliers))
X_train = data_outliers.drop('Loan_Status', axis=1)
y_train = data_outliers['Loan_Status']
sns.boxplot(X_train['new_col_2']);
plt.title('new_col_2 without outliers', fontsize=15);
plt.xlabel('');

train_eval_cross(models, X_train, y_train, skf)


data_corr = pd.concat([X_train, y_train], axis=1)
corr = data_corr.corr()
plt.figure(figsize=(10,7))
sns.heatmap(corr, annot=True);

train_eval_cross(models, X_train, y_train, skf)

                    #EVALUATE THE MODEL ON TEST DATA
print(X_test.head())
X_test_new = X_test.copy()
x = []

X_test_new['new_col'] = X_test_new['CoapplicantIncome'] / X_test_new['ApplicantIncome']  
X_test_new['new_col_2'] = X_test_new['LoanAmount'] * X_test_new['Loan_Amount_Term']
X_test_new.drop(['CoapplicantIncome', 'ApplicantIncome', 'Loan_Amount_Term', 'LoanAmount'], axis=1, inplace=True)

X_test_new['new_col_2'] = np.log(X_test_new['new_col_2'])

X_test_new['new_col'] = [x if x==0 else 1 for x in X_test_new['new_col']]

print(X_test_new.head())
print(X_train.head())

for name,model in models.items():
    print(name, end=':\n')
    loss(y_test, model.predict(X_test_new))
    print('-'*40)





















