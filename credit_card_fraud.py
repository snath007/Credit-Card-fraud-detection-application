# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 17:38:16 2019

@author: ASUS
"""
import os
#os.getcwd()
os.chdir('D:/')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
#from imblearn import under_sampling, over_sampling
import scipy.stats as stats

#importing the dataset

df = pd.read_csv('C:/Users/ASUS/Desktop/creditcard.csv')
df = df.drop('Time', axis = 1)

df.columns
df.shape

#checking for null values

df.isnull().sum()

# checking the correlation between the variables (independent and dependent)

plt.figure(figsize=(14,12))
sns.heatmap(df.corr(),linewidths=.1,cmap="YlGnBu", annot=True)
plt.yticks(rotation=0);

# dropping the 'Amount' field as it is correlated with many variables

df = df.drop('Amount', axis = 1)

# checking the correlation between the variables after removing 'Amount'

plt.figure(figsize=(14,12))
sns.heatmap(df.corr(),linewidths=.1,cmap="YlGnBu", annot=True)
plt.yticks(rotation=0);

#splitting dataset into train and test

X = df.drop('Class', axis = 1)
Y = df['Class']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#importing required function
from sklearn.ensemble import ExtraTreesClassifier

#building model
model = ExtraTreesClassifier()
model.fit(X_train, Y_train)

#printing out the importance scores
print(model.feature_importances_)

#creating series for better visualization
feat_importances = pd.Series(model.feature_importances_, index = X_train.columns)

#plotting graph of feature importances for better visualization
feat_importances.nlargest(10).plot(kind='barh')   #top 10 features
plt.show()

#printing feat_importances to get another idea
print(feat_importances)

#feat_importances[feat_importances < 0.05]

#checking for imbalanced class in the dataset

sns.set()
ax = sns.countplot(x=Y_train[:], y=None, hue=None, data=pd.concat([X_train, Y_train], axis=1, sort=False))
for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points');

df['Class'].value_counts()


#checking the count of the fraud and non fraud cases

count_fraud = 0
count_nonfraud = 0

for i in Y_train[:]:
    if i == 1:
        count_fraud+=1
    else:
        count_nonfraud+=1
        
#as the dataset is highly imbalanced, applying SMOTE to generate synthetic datapoints of the minority class

from imblearn.over_sampling import SMOTE

print("Before OverSampling, counts of label '0':", count_nonfraud)
print("Before OverSampling, counts of label '1':", count_fraud)
print("\n")

sm = SMOTE(random_state = 2)
# print(sm)
train_smoted, traintarget_smoted = sm.fit_sample(X_train, Y_train.ravel())
  
print('After OverSampling, the shape of train: ', train_smoted.shape)
print('After OverSampling, the shape of traintarget: ', traintarget_smoted.shape) 
print("\n")
  
print("After OverSampling, counts of label '0': ", (sum(traintarget_smoted == 0))) 
print("After OverSampling, counts of label '1': ", (sum(traintarget_smoted == 1)))

#naming the columns of the smoted data

train_smoted_df = pd.DataFrame(train_smoted, columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11','V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21','V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28'])
print(train_smoted_df.head())
train_target_cols = []
train_target_cols.append(df.columns[28])
traintarget_smoted_df = pd.DataFrame(traintarget_smoted, columns = train_target_cols)
print(traintarget_smoted_df.head())


#checking outliers

def outliers(y):
#    %matplotlib inline
    plt.boxplot(y)
    plt.show()
    
outliers(train_smoted_df['Amount'])

#investigating the distribution of the independent variables

def histo(x):
    plt.hist(x, normed=True, bins=30)
    plt.show()
    
histo(train_smoted_df['Time'])
histo(train_smoted_df['V5'])

#applying ANOVA to check correlation of the independent variables with the target categorical variable

anova = stats.f_oneway(X_train[traintarget_smoted_df['Class'] == 0], X_train[traintarget_smoted_df['Class'] == 1])
anova

##Feature scaling the smotted train and test datasets
#
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#train_smoted_df = sc_X.fit_transform(train_smoted_df)
#X_test = sc_X.transform(X_test)
#X_train = sc_X.fit_transform(X_train)
#       

#fitting the data into a classification algorithm
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(train_smoted_df, traintarget_smoted_df)

from sklearn.linear_model import LogisticRegression

cl = LogisticRegression(random_state = 0)
cl.fit(train_smoted_df, traintarget_smoted_df)

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()

clf = clf.fit(train_smoted_df, traintarget_smoted_df)

#converting X_test,X_train into dataframe

X_test = pd.DataFrame(X_test, columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11','V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21','V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28'])
X_train = pd.DataFrame(X_train, columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11','V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21','V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28'])


#predicting results

Y_pred = clf.predict(X_test)

#Y_pred = clf.predict(np.array([[0,-1.3598071336738,-0.0727811733098497,2.53634673796914,1.37815522427443,-0.338320769942518,0.462387777762292,0.239598554061257,0.0986979012610507,0.363786969611213,0.0907941719789316,-0.551599533260813,-0.617800855762348,-0.991389847235408,-0.311169353699879,1.46817697209427,-0.470400525259478,0.207971241929242,0.0257905801985591,0.403992960255733,0.251412098239705,-0.018306777944153,0.277837575558899,-0.110473910188767,0.0669280749146731,0.128539358273528,-0.189114843888824,0.133558376740387,-0.0210530534538215,149.62]]).reshape(-1, 1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)


cm


# from the confusion matrix it can be inferred that the decission tree classifier is the best model among the three with accuracy of 99.77%

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = clf, X = X_train, y = Y_train, cv = 10)
accuracies.mean()
accuracies.std()

#the cross validation accuracy is 99.91%, so we choose the decision tree classifier.

# Saving model to disk
pickle.dump(clf, open('credit_card_fraud.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('credit_card_fraud.pkl','rb'))