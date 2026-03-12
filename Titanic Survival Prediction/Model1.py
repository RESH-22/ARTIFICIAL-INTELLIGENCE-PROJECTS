import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


df=pd.read_csv('titanic.csv')    #to read the file
print(df)



from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

##Replacing Null
df['Age'] = df['Age'].replace(np.nan, 0)
df['Embarked'] = df['Embarked'].replace(np.nan, 0)

print(df)


x=df.drop(columns=['Survived','Name','PassengerId','Ticket','Cabin'])
y=df['Survived']      #to create the variable

print("XXXX",x)
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


import pickle
pickle.dump(NB,open('model.pkl','wb'))

##
##testPrediction = NB.predict([[19,1,4,120,166,0,1,138,0,0,2]])
##if testPrediction==1:
##    print("The Patient Have Heart Disease,please consult the Doctor")
##else:
##    print("The Patient Normal")



##
##
##
##from sklearn.metrics import confusion_matrix
##from sklearn.metrics import precision_score, recall_score, f1_score ,confusion_matrix
##cm = confusion_matrix(y_test, y_pred)
##print("Confusion Matrix:")
##print(cm)
##
### Calculate precision
##precision = precision_score(y_test, y_pred)
##print("Precision:", precision)
##
### Calculate recall
##recall = recall_score(y_test, y_pred)
##print("Recall:", recall)
##
### Calculate F1 score
##f1 = f1_score(y_test, y_pred)
##print("F1 Score:", f1)
##





































##





##age=int(input("enter the age"))
##
##
##
##import warnings
##warnings.filterwarnings('ignore')
##
##
##
####
####
####
####
####
####
####
####
##import warnings
##warnings.filterwarnings('ignore')
##import matplotlib.pyplot as plt
##import seaborn as sns
##labels = ['yes', 'No']
##values = df['HeartDisease'].value_counts().values
##plt.pie(values, labels=labels, autopct='%1.0f%%')
##plt.title('HeartDisease')
##plt.show()
####
####
##pd.crosstab(df.ChestPainType,df.HeartDisease).plot(kind = "bar", figsize = (8, 6))
##plt.title('Heart Disease Frequency According to Chest Pain Type')
##plt.xlabel('Chest Pain Type')
##plt.xticks(np.arange(4), ('typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'), rotation = 0)
##plt.ylabel('Frequency')
##plt.show()
######
######
######
######
######
######
######
######
######
######
##










##
##
## 
