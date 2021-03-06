# -*- coding: utf-8 -*-
"""AI Project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1POnEV0-GHjsWPRFpgIYWwy5rZKaiDgbA
"""

from google.colab import drive
drive.mount('/content/drive')

# Imports
import pandas as pd
import re
import string
import nltk
import spacy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn import model_selection, naive_bayes
from sklearn import tree

# 1st Model
from sklearn.linear_model import LogisticRegression
# 2nd Model
from sklearn.ensemble import RandomForestClassifier
# 3rd Model
from sklearn.naive_bayes import MultinomialNB
# 4th Model
from sklearn.tree import DecisionTreeClassifier
# 5th Model
from sklearn.neighbors import KNeighborsClassifier
# 6th Model
from sklearn.ensemble import AdaBoostClassifier

##### Model 1 -> Logistic Regression
#AdaBoost also called Adaptive Boosting is a technique in Machine Learning used as an Ensemble Method. The most common algorithm used with AdaBoost is decision trees with one level that means with Decision trees with only 1 split. These trees are also called Decision Stumps.

###### Model 2 -> Random Forest
#A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is controlled with the max_samples parameter if bootstrap=True (default), otherwise the whole dataset is used to build each tree.

###### Model 3 -> Multinomial Naive Bayes
#In probability theory, the multinomial distribution is a generalization of the binomial distribution. For example, it models the probability of counts for each side of a k-sided die rolled n times. For n independent trials each of which leads to a success for exactly one of k categories, with each category having a given fixed success probability, the multinomial distribution gives the probability of any particular combination of numbers of successes for the various categories.

###### Model 4 -> Decision Tree Classifier
#Decision Tree is a Supervised Machine Learning Algorithm that uses a set of rules to make decisions, similarly to how humans make decisions.
#One way to think of a Machine Learning classification algorithm is that it is built to make decisions.
#You usually say the model predicts the class of the new, never-seen-before input but, behind the scenes, the algorithm has to decide which class to assign.

###### Model 5 -> KNN - K Nearest Neighbors
#A supervised machine learning algorithm (as opposed to an unsupervised machine learning algorithm) is one that relies on labeled input data to learn a function that produces an appropriate output when given new unlabeled data.
#Imagine a computer is a child, we are its supervisor (e.g. parent, guardian, or teacher), and we want the child (computer) to learn what a pig looks like. We will show the child several different pictures, some of which are pigs and the rest could be pictures of anything (cats, dogs, etc).

###### Model 6 -> Adaptive Boostings
#AdaBoost also called Adaptive Boosting is a technique in Machine Learning used as an Ensemble Method. The most common algorithm used with AdaBoost is decision trees with one level that means with Decision trees with only 1 split. These trees are also called Decision Stumps.



# Connecting Files from Drive
# As required Train.csv and Test.csv
train_data = pd.read_csv('/content/drive/MyDrive/AI Project Fall 2021/train.csv')
test_data = pd.read_csv('/content/drive/MyDrive/AI Project Fall 2021/test.csv')

test_data.head(2)
train_data.head(2)

# Train
train = pd.DataFrame(train_data)

scaler = MinMaxScaler()
X_Scaled_Train = scaler.fit_transform(train.values)

# Normalize Train
train_norm = pd.DataFrame(X_Scaled_Train)

X = train.Slope
Y = train.Cover_Type
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 109)

# Checking Length
print("X_train : ", len(X_train))
print("Y_train : ", len(Y_train))
print("X_test : ", len(X_test))
print("Y_test : ", len(Y_test))


train_value = X_train.values.reshape(-1, 1)
test_value = X_test.values.reshape(-1, 1)

train_scale_value = scaler.fit_transform(train_value)
test_scale_value = scaler.fit_transform(test_value)

# Model 1 -> Logistic Regression
print("<-- Model 1 Running -->")
Model_1 = LogisticRegression()
Model_1.fit(train_value, Y_train)
M1_prediction = Model_1.predict(test_value)
accuracy_score(M1_prediction,Y_test) * 100

# Model 2 -> Random Forest
print("<-- Model 2 Running -->")
Model_2 = RandomForestClassifier(n_estimators = 40)
Model_2.fit(train_value, Y_train)
M2_prediction =  Model_2.predict(test_value)
confusion_matrix(Y_test, M2_prediction)
Model_2.score(test_value, Y_test) * 100

# Model 3 -> Multinomial Naive Bayes
print("<-- Model 3 Running -->")
Model_3 = MultinomialNB()
Model_3.fit(train_scale_value, Y_train)
M3_prediction = Model_3.predict(test_scale_value)
accuracy_score(M3_prediction, Y_test) * 100

# Model 4 -> Decision Tree Classifier
print("<-- Model 4 Running with Feature Train -->")
Model_4 = DecisionTreeClassifier()
# -> Train DataFrame
Model_4 = Model_4.fit(train_value, Y_train)
M4_prediction = Model_4.predict(test_value)
accuracy_score(M4_prediction, Y_test) * 100

# Model 4 -> Decision Tree Classifier
print("<-- Model 4 Running with Feature Train Norm -->")
Model_4 = DecisionTreeClassifier() 
# -> Train_Norm DataFrame
Model_4 = Model_4.fit(train_scale_value, Y_train)
M4_prediction = Model_4.predict(test_scale_value)
accuracy_score(M4_prediction, Y_test) * 100

# Model 5 -> KNN - K Nearest Neighbors
print("<-- Model 5 Running with Feature Train -->")
Model_5 = KNeighborsClassifier(n_neighbors = 5)
# -> Train DataFrame
Model_5.fit(train_value, Y_train)
M5_prediction =  Model_5.predict(test_value)
confusion_matrix(M5_prediction, Y_test)
accuracy_score(M5_prediction, Y_test) * 100

# Model 5 -> KNN - K Nearest Neighbors
print("<-- Model 5 Running with Feature Train Norm -->")
Model_5 = KNeighborsClassifier(n_neighbors = 5)
# -> Train_Norm DataFrame
Model_5.fit(train_scale_value, Y_train)
M5_prediction =  Model_5.predict(test_scale_value)
confusion_matrix(M5_prediction, Y_test)
accuracy_score(M5_prediction, Y_test) * 100

# Model 6 -> Adaptive Boosting
print("<-- Model 6 Running with Feature Train -->")
Model_6 = AdaBoostClassifier(n_estimators = 100)
# -> Train DataFrame
Model_6.fit(train_value, Y_train)
M6_prediction = Model_6.predict(test_value)
confusion_matrix(M6_prediction, Y_test)
accuracy_score(M6_prediction, Y_test) * 100

# Model 6 -> Adaptive Boosting
print("<-- Model 6 Running with Feature Train Norm -->")
Model_6 = AdaBoostClassifier(n_estimators = 100)
# -> Train DataFrame
Model_6.fit(train_scale_value, Y_train)
M6_prediction = Model_6.predict(test_scale_value)
confusion_matrix(M6_prediction, Y_test)
accuracy_score(M6_prediction, Y_test) * 100

id = test_data["Id"]
Test = pd.DataFrame(id)
arr=[]
for row in id:
  arr.append(labels[row])
Test["Cover type"] = arr
Test.to_csv('response.csv',index = False)