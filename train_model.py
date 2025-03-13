import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

os.makedirs("model",exist_ok = True)

#step1 : load data

dataset = pd.read_excel('dataset\internet_speed_vs_download_time_1gb.xlsx')

#step2 : xtracts dependent and independent variables

x = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1:2].values

#step3 : handlinhg missing values

imputer = SimpleImputer (missing_values = np.NaN, strategy = 'mean')
x1 = imputer.fit_transform(x)
y1 = imputer.fit_transform(y)

#step4 : spli ttig the dataset into train and test set

x_train,x_test,y_train,y_test = train_test_split(x1,y1,test_size = 0.3,random_state = 0)

#step5 : fit the model to training set

regressor = LinearRegression()
regressor.fit(x_train,y_train)

#print model performance

print("model R^2 Score on entire datset")
print(regressor.score(x1 ,y1))

#create visualisation

plt.scatter(x_test,y_test, color="red")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("internet_speed_vs_download_time_1gb.xlsx")
plt.xlabel("internet_speed")
plt.ylabel("download_time")
plt.show()





