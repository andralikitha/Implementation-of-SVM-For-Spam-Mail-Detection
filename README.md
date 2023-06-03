# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary packages.  
2. Read the given csv file and display the few contents of the data.  
3. Assign the features for x and y respectively.  
4. Split the x and y sets into train and test sets.  
5. Convert the Alphabetical data to numeric using CountVectorizer.  
6. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.  
7. Find the accuracy of the model.  

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Andra likitha 
RegisterNumber:  212221220006
*/
```
print("Result Output:")
import chardet 
file='/content/spam.csv'
with open(file, 'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("/content/spam.csv",encoding='Windows-1252')

print("data head:")
data.head()

print("data info:")
data.info()

print("data isnull:")
data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

print("y_prediction  value:")
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
y_pred

print("Accuracy Value:")
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

## Output:

![image](https://github.com/Dhanush12022004/Implementation-of-SVM-For-Spam-Mail-Detection/assets/128135558/98d6402d-65d4-4322-8fa0-9d3dca2dc3e0)
 
![image](https://github.com/Dhanush12022004/Implementation-of-SVM-For-Spam-Mail-Detection/assets/128135558/cd750f0f-c118-448a-a4b7-b9e552cd7f48)

![image](https://github.com/Dhanush12022004/Implementation-of-SVM-For-Spam-Mail-Detection/assets/128135558/669425c1-ad85-4052-a5cf-22fccaf3e40c)

![image](https://github.com/Dhanush12022004/Implementation-of-SVM-For-Spam-Mail-Detection/assets/128135558/01826e8f-0504-4042-8b12-781c43f62bab)

![image](https://github.com/Dhanush12022004/Implementation-of-SVM-For-Spam-Mail-Detection/assets/128135558/af49926b-1c7c-43e2-9ddf-6621f0a2d3f0)

![image](https://github.com/Dhanush12022004/Implementation-of-SVM-For-Spam-Mail-Detection/assets/128135558/c3fa6109-f0c3-401a-a7fb-41f939df9ea4)




## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
