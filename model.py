import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

data=pd.read_csv("student_data.csv")
X=data[["hoursStudied","attendance","previousMarks","sleepHours"]]
y=data["finalScore"]

model=LinearRegression()
model.fit(X,y)

with open("model.pkl","wb") as file:
    pickle.dump(model, file)
print("Model Trained and saved as model.pkl")