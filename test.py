import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
import os

spark = SparkSession.builder.appName("HDFS").getOrCreate()
sparkcont = SparkContext.getOrCreate(SparkConf().setAppName("HDFS")) #creating spark object and initializing it
logs = sparkcont.setLogLevel("ERROR")
filePath = os.path.abspath('Dataset/Mental_illness.csv')
print(filePath)
df = spark.read.option("header","true").csv("file:///"+filePath)

dataset = df.toPandas()#pd.read_csv('Dataset/Mental_illness.csv')
print(type(dataset))
dataset.fillna(0, inplace = True)
dataset.drop(['number'], axis = 1,inplace=True)

cols = ['age','edu']
le = LabelEncoder()
dataset[cols[0]] = pd.Series(le.fit_transform(dataset[cols[0]].astype(str)))
dataset[cols[1]] = pd.Series(le.fit_transform(dataset[cols[1]].astype(str)))

Y = dataset['afftype']
Y = np.asarray(Y)
dataset.drop(['afftype'], axis = 1,inplace=True)
dataset = dataset.values
X = dataset[:,0:dataset.shape[1]]
print(X)
print(Y)

X = normalize(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
cls = RandomForestClassifier()
cls.fit(X_train,y_train)
predict = cls.predict(X_test) 
a = accuracy_score(y_test,predict)*100
print(a)

test = pd.read_csv('Dataset/testData.csv')
test.fillna(0, inplace = True)
test.drop(['number'], axis = 1,inplace=True)
cols = ['age','edu']
le = LabelEncoder()
test[cols[0]] = pd.Series(le.fit_transform(test[cols[0]].astype(str)))
test[cols[1]] = pd.Series(le.fit_transform(test[cols[1]].astype(str)))
test = test.values
test = normalize(test)
predict = cls.predict(test)
print(predict)


