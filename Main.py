from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
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
import matplotlib.pyplot as plt

main = tkinter.Tk()
main.title("Data Analytics in Mental Healthcare")
main.geometry("1300x1200")

global dataset
global accuracy
global filename
global X, Y
global dataset
global le
global X_train, X_test, y_train, y_test
global classifier

def uploadDataset():
    global filename
    filename = filedialog.askopenfilename(initialdir="Dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n");

def sparkDataProcessing():
    global X, Y
    global dataset
    global filename
    text.delete('1.0', END)
    global le
    # spark = SparkSession.builder.appName("HDFS").getOrCreate()
    # sparkcont = SparkContext.getOrCreate(SparkConf().setAppName("HDFS")) #creating spark object and initializing it
    # logs = sparkcont.setLogLevel("ERROR")
    # df = spark.read.option("header","true").csv("file:///"+filename)
    dataset = pd.read_csv('Dataset/Mental_illness.csv') #df.toPandas()
    dataset.fillna(0, inplace = True)
    dataset.drop(['number'], axis = 1,inplace=True)
    cols = ['age','edu']
    le = LabelEncoder()
    dataset[cols[0]] = pd.Series(le.fit_transform(dataset[cols[0]].astype(str)))
    dataset[cols[1]] = pd.Series(le.fit_transform(dataset[cols[1]].astype(str)))
    text.insert(END,str(dataset.head())+"\n\n")
    text.insert(END,"Dataset loading process completed")
    
def trainTestData():
    global X_train, X_test, y_train, y_test
    global X, Y
    global dataset
    text.delete('1.0', END)
    Y = dataset['afftype']
    Y = np.asarray(Y)
    dataset.drop(['afftype'], axis = 1,inplace=True)
    dataset = dataset.values
    X = dataset[:,0:dataset.shape[1]]
    X = normalize(X)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    text.insert(END,"Total Records found is dataset : "+str(len(X))+"\n")
    text.insert(END,"Train & Test Dataset split\n")
    text.insert(END,"Training Dataset size : "+str(len(X_train))+"\n")
    text.insert(END,"Testing Dataset size : "+str(len(X_test))+"\n")

def trainML():
    global X_train, X_test, y_train, y_test
    global classifier
    global accuracy
    text.delete('1.0', END)
    cls = RandomForestClassifier()
    cls.fit(X_train,y_train)
    predict = cls.predict(X_test) 
    accuracy = accuracy_score(y_test,predict)*100
    classifier = cls
    text.insert(END,"Random Forest Accuracy On Mental illness Test Data : "+str(accuracy)+"\n")
    

def predictMentalIllness():
    global le
    global classifier
    filename = filedialog.askopenfilename(initialdir="Dataset")
    test = pd.read_csv(filename)
    test.fillna(0, inplace = True)
    test.drop(['number'], axis = 1,inplace=True)
    cols = ['age','edu']
    le = LabelEncoder()
    test[cols[0]] = pd.Series(le.fit_transform(test[cols[0]].astype(str)))
    test[cols[1]] = pd.Series(le.fit_transform(test[cols[1]].astype(str)))
    test = test.values
    test1 = test
    test = normalize(test)
    predict = classifier.predict(test)
    print(predict)

        
    for i in range(len(predict)):
        if predict[i] == 1:
            text.insert(END,str(test1[i])+" Mental Illness Detected as : bipolar II Disorder\n\n")
        if predict[i] == 2:
            text.insert(END,str(test1[i])+" Mental Illness Detected as : Unipolar Depressive Disorder\n\n")
        if predict[i] == 3:
            text.insert(END,str(test1[i])+" Mental Illness Detected as : Personality Disorder\n\n")    
            
def graph():
    global accuracy
    height = [accuracy,(100 - accuracy)]
    bars = ('Random Forest ML Prediction Accuracy','Random Forest ML Prediction Error Rate')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

def close():
    main.destroy()

font = ('times', 16, 'bold')
title = Label(main, text='Data Analytics in Mental Healthcare',anchor=W, justify=CENTER)
title.config(bg='yellow4', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Dataset", command=uploadDataset)
upload.place(x=50,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='yellow4', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=50,y=150)

dataButton = Button(main, text="Read & Process Dataset", command=sparkDataProcessing)
dataButton.place(x=50,y=200)
dataButton.config(font=font1)

svrButton = Button(main, text="Split Dataset into Train & Test", command=trainTestData)
svrButton.place(x=50,y=250)
svrButton.config(font=font1)

mlrButton = Button(main, text="Train Mental Illness Data with Random Forest ML", command=trainML)
mlrButton.place(x=50,y=300)
mlrButton.config(font=font1)

lstmButton = Button(main, text="Upload Test Data & Predict Mental Illness", command=predictMentalIllness)
lstmButton.place(x=50,y=350)
lstmButton.config(font=font1)

graphButton = Button(main, text="Accuracy Comparison Graph", command=graph)
graphButton.place(x=50,y=400)
graphButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=50,y=450)
exitButton.config(font=font1)



font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=490,y=200)
text.config(font=font1)


main.config(bg='magenta3')
main.mainloop()
