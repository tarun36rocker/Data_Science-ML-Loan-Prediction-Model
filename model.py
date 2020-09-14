import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

data=pd.read_csv('train.csv')
#taking care of null values
nulls = data.isnull().sum()
nulls[nulls > 0] 
data=data.fillna(0) #dont use inplace=true
nulls = data.isnull().sum()
nulls[nulls > 0]
#taking care of categorical columns
data=pd.get_dummies(data, columns=['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status'])
data=data.drop(['Loan_ID','Gender_0','Gender_Female', 'Married_0','Married_No','Education_Not Graduate','Self_Employed_0','Self_Employed_No','Property_Area_Rural','Property_Area_Semiurban','Loan_Status_N'], axis=1)
data=data.replace("3+", 5)
datatest=pd.read_csv('test.csv')
#taking care of categorical columns
datatest=pd.get_dummies(datatest, columns=['Gender','Married','Education','Self_Employed','Property_Area'])
datatest=datatest.drop(['Loan_ID','Gender_Female','Married_No','Education_Not Graduate','Self_Employed_No','Property_Area_Rural','Property_Area_Semiurban'], axis=1)
X_train = data.iloc[:, 0:11].values
y_train = data.iloc[:, 11].values
#from sklearn.model_selection import train_test_split
X_test=datatest.iloc[:,:].values
#standardising only X values


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

# Saving model to disk
pickle.dump(logmodel, open('model4.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model4.pkl','rb'))
print(y_train[7],X_train[7])
print(model.predict([[5, 3036, 2504.0 ,158.0 ,360.0 ,0.0 ,1 ,1 ,1 ,0 ,0]]))