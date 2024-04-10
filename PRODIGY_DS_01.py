import numpy as np
import pandas as pd

df=pd.read_csv('/content/drive/MyDrive/Document from pratheeksha')
df.tail(5)

df.head()

df.describe()

df.info()

df['species'].value_counts()

df.isnull().sum()

df['sepal_length'].hist()

df['sepal_width'].hist()

import matplotlib.pyplot as plt
colors=['red','orange','blue']
species=['setosa', 'versicolor', 'virginica']
for i in range(3):
  x=df[df['species']==species[i]]
  plt.scatter(x['sepal_length'],x['sepal_width'],c=colors[i],label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal width")
plt.legend()

for i in range(3):
  x=df[df['species']== species[i]]
  plt.scatter(x['petal_length'],x['petal_width'],c=colors[i],label=species[i])
plt.xlabel("Petal Length")
plt.ylabel("Petal width")
plt.legend()

for i in range(3):
    x=df[df['species']== species[i]]
    plt.scatter(x['sepal_width'],x['petal_length'],c=colors[i],label=species[i])
plt.xlabel("Sepal Width")
plt.ylabel("Petal Length")
plt.legend(loc='upper right')

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['species']=le.fit_transform(df['species'])

df.drop(columns=['species'])

df.head()

from sklearn.model_selection import train_test_split
X=df.drop(columns=['species'])
Y=df['species']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30,random_state=42)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()

model.fit(X_train,Y_train)

print("accuracy of the model",model.score(X_test,Y_test )*100)
