import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
data=pd.read_csv('adult.csv',sep=',')
print("data getting read in ")

le=LabelEncoder()
data['occupation']=le.fit_transform(data['occupation'])
data['education']=le.fit_transform(data['education'])
data['native.country']=le.fit_transform(data['native.country'])
print("data  re labeled")

X = data[['age', 'education', 'occupation', 'hours.per.week', 'native.country']]
y = data['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)   
print("test sets ready ")

print("at the gym")
md=RandomForestClassifier(n_estimators=100)
md.fit(X_train,y_train)
print("left the gym")


Pred = md.predict(X_test)
accuracy_score=md.score(X_test, y_test)

classification_report_rf=classification_report(y_test,Pred)
print("Accuracy :",accuracy_score)  
print("Classification Report:")
print(classification_report(y_test, Pred))