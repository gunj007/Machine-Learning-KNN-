#data preprocessing 
import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
import seaborn as sns

#import dataset
df= pd.read_csv('framingham.csv')
df=df.fillna(0)
#print(d)

#extract dependent and independent
x = df.iloc[:,:-1].values

#print("x:",x)
y = df.iloc[:,15].values
#print("y:",y)

#split into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.20,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
st= StandardScaler()
x_train=st.fit_transform(x_train)
x_test=st.fit_transform(x_test)

#Fitting K-NN classifier to the training set  
from sklearn.neighbors import KNeighborsClassifier  
classifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
classifier.fit(x_train, y_train)

#predicts test result
y_pred = classifier.predict(x_test)

#create confusion matrix to get right and wrong outcome
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

#accuracy score
from sklearn.metrics import  accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)
