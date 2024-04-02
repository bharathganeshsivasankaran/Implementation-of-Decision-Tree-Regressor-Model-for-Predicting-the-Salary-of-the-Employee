# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

### Algorithm
```
1.Import the libraries and read the data frame using pandas. 
2.Calculate the null values present in the dataset and apply label encoder.
3.Determine test and training data set and apply decison tree regression in dataset. 
4.calculate Mean square error,data prediction and r2.
```
### Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Bharathganesh S
RegisterNumber:  212222230022
*/
```
```
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data[["Salary"]]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])

```

## Output:
### Initial dataset:
![image](https://github.com/bharathganeshsivasankaran/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119478098/9d4ac4fb-91b2-4bc4-90c1-4c6aef350d8b)
### Data Info:
![image](https://github.com/bharathganeshsivasankaran/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119478098/68720f67-f70b-473a-9189-36381b54c6a9)
### Optimization of null values:
![image](https://github.com/bharathganeshsivasankaran/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119478098/cd6387b2-d372-43ca-a16d-1395b497527f)
### Converting string literals to numerical values using label encoder:
![image](https://github.com/bharathganeshsivasankaran/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119478098/b7b2d1e6-c14d-4535-8dd9-10bacdd1068d)
### Assigning x and y values:
![image](https://github.com/bharathganeshsivasankaran/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119478098/cf6b9070-6544-489e-8084-ad655a53ebaa)
### Mean Squared Error:
![image](https://github.com/bharathganeshsivasankaran/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119478098/fd463eb7-42c2-4a88-ada8-12511c0cc553)
### R2 (variance):
![image](https://github.com/bharathganeshsivasankaran/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119478098/b0b8fe9c-73c2-4dea-a4bb-ddfc3be2b404)
### Prediction:
![image](https://github.com/bharathganeshsivasankaran/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119478098/8ec56277-1440-4420-8b0c-003dd05fe6ee)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
