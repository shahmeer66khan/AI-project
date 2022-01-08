
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
train_data = pd.read_csv('/content/drive/MyDrive/AI Project Fall 2021/train.csv')
test_data = pd.read_csv('/content/drive/MyDrive/AI Project Fall 2021/test.csv')

test_data.head(2)

train_data.head(2)

train = pd.DataFrame(train_data)

from sklearn.model_selection import train_test_split
X = train.Slope
Y = train.Cover_Type
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=109)
print("X_train : ",len(X_train))
print("X_test : ",len(X_test))
print("X_train : ",len(y_train))
print("X_test : ",len(y_test))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler

model = MultinomialNB()
scaler = MinMaxScaler()
T= X_train.reshape(-1,1)
X_train = scaler.fit_transform(T)
Test = test_data.values.reshape(-1,1)
X_test = scaler.fit_transform(Test)
model.fit(X_train,y_train)
labels = model.predict(Test)

from sklearn import metrics
# Model Accuracy
print("Accuracy:",metrics.accuracy_score(Test, labels))

id = test_data["Id"]
Test = pd.DataFrame(id)
arr=[]
for row in id:
  arr.append(labels[row])
Test["Cover type"] = arr
Test.to_csv('response.csv',index = False)