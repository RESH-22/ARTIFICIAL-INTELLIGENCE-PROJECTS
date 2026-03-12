
import numpy as np
import pandas as pd
df= pd.read_csv("titanic.csv")


##pd.set_option('display.max_columns', None)
##print(df)

"""# **EDA**"""

print(df.info())

print(df.head(10))

print(df.tail(10))

print(df.columns)

print(df.shape)

print(df.duplicated().sum())

print(df.isnull().sum())

import matplotlib.pyplot as plt
import missingno as ms

ms.bar(df,figsize = (10,5),color="tomato")
plt.title("Bar plot showing missing data values", size = 15,c="r")
plt.show()


df.drop(['Cabin'], axis=1, inplace=True)

print(df.shape)

print(df["Embarked"].unique())

print(df["Embarked"].value_counts())

df["Embarked"]=df["Embarked"].fillna("S")

print(df["Embarked"].value_counts())



print(df["Age"].value_counts())

print(df["Age"].mean())

print(df["Age"].median())

print(df["Age"].mode().value_counts())

print(df.describe().astype(int))


df["Age"] = df["Age"].fillna(df["Age"].mean())

print(df["Age"].isnull().sum())


import matplotlib.pyplot as plt
import missingno as ms

ms.bar(df,figsize = (10,5),color="tomato")
plt.title("Bar plot showing missing data values", size = 15,c="r")
plt.show()


print(df.isnull().sum())


print(df["Survived"].value_counts())

import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x="Survived", data=df)
plt.title("count of passengers who survived")
plt.show()

print(df["Sex"].value_counts())

fig,axes = plt.subplots(1,2,figsize=(5,3))
df["Sex"].value_counts().plot(kind="bar", ax=axes[1], color =['DarkRed','indianred'])
df["Sex"].value_counts().plot(kind="pie",ax=axes[0],autopct='%0.1f' ,colormap="Reds")
plt.show()

sns.catplot(x="Sex",hue="Survived", kind="count",data=df,height=3,)
plt.show()



sns.countplot(x="Pclass", hue="Survived", data=df, palette="Reds",)
plt.show()



df.drop(["Name","Ticket","PassengerId"],axis=1,inplace=True)
df.head()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])



print(df.corr())




df['Age'] = df['Age'].replace(np.nan, 0)
df['Embarked'] = df['Embarked'].replace(np.nan, 0)

print(df)


x=df.drop(["Survived"],axis=1)
y=df["Survived"]

##print("XXXX",x)
print("YYYY",y)




from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=12)   #split the val

print("DF",df.shape)
print("x_train",x_train.shape)
print("x_test",x_test.shape)
print("y_train",y_train.shape)
print("y_test",y_test.shape)




from sklearn.naive_bayes import GaussianNB  
NB = GaussianNB()

NB.fit(x_train, y_train)


###train the data
y_pred=NB.predict(x_test)
print("y_pred",y_pred)
print("y_test",y_test)



from sklearn.metrics import accuracy_score
print('ACCURACY is', accuracy_score(y_test,y_pred))






































##plt.hist(df["Age"][(df["Sex"]=="female")&(df["Survived"] ==1)].dropna(),bins=7,label="female",histtype="stepfilled")
##plt.hist(df["Age"][(df["Sex"]=="male")&(df["Survived"] ==1 )].dropna(),bins=7,label="male",histtype="stepfilled")
##plt.xlabel("Age")
##plt.ylabel("count")
##plt.title("Age wise distribution of male and female survivors")
##plt.legend()
##plt.show()
##
##
##plt.hist(df["Age"][(df["Sex"]=="male")&(df["Survived"] ==0 )].dropna(),bins=7,label="male",alpha=.7,histtype="stepfilled")
##plt.hist(df["Age"][(df["Sex"]=="female")&(df["Survived"] ==0)].dropna(),bins=7,label="female",alpha=.7,histtype="stepfilled")
##plt.xlabel("Age")
##plt.ylabel("count")
##plt.title("Age wise distribution of male and female not survived")
##plt.legend()
##plt.show()





