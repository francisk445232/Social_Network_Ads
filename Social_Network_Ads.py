#importing numpy as np
import numpy as np

#importing pandas as pd
import pandas as pd

#importing matplotlib.pyplot as plt
import matplotlib.pyplot as plt

#this code is to create a variable to store dataset
dataset = pd.read_csv("Social_Network_Ads.csv")


#create variable x to store the independent column values
x = dataset.iloc[:,0:4].values
y = dataset.iloc[:,4].values

#Encoding categorical data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
column_trans = make_column_transformer((OneHotEncoder(), [1]), remainder='passthrough')
x = column_trans.fit_transform(x)
 
# we are splitting the Dataset into Train data and Test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# scaling using StandardScaler
from sklearn.preprocessing import StandardScaler

#assigning the StandardScaler to a  variable sc_x
sc_x = StandardScaler()

#fitting and transforming the x train and x test
X_train = sc_x.fit_transform(X_train)


#fittin x_test
X_test = sc_x.transform(X_test)

#training the decision tree module
from sklearn.tree import DecisionTreeRegressor

#we will create a variable and assigning the Decision tree regression algorithm
Social_Network_Ads = DecisionTreeRegressor(random_state=1)

#now am training the module Social_Network_Ads with x_train and y_train
Social_Network_Ads.fit(X_train,y_train)

#making a prediction
prediction_result = Social_Network_Ads.predict(X_test)
prediction_result

#Evaluating the Answers by accurrcy_score
from sklearn.metrics import accuracy_score

#add commit
score= accuracy_score(prediction_result,y_test)
score
print(score*100, '%')
