import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
df=pd.read_csv("data/aapl.us.txt")
df["Date"]=pd.to_datetime(df["Date"])
df=df.sort_values("Date")
df["Target"]=(df["Close"].shift(-1)> df["Close"]).astype(int)
df=df.dropna()
features=["Open","High","Low","Close","Volume"]
x=df[features]
y=df["Target"]
split=int(len(df)*0.8)
x_train,x_test=x.iloc[:split],x.iloc[split:]
y_train,y_test=y.iloc[:split],y.iloc[split:]
model=RandomForestClassifier(n_estimators=100,
random_state=42)
model.fit(x_train,y_train)
predictions=model.predict(x_test)
accuracy=accuracy_score(y_test,predictions)
print("Model Accuracy:",round(accuracy*100,2),"%")

