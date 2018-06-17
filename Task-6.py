# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 21:49:48 2018

@author: Amartya
"""
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
didata = pd.read_csv('diabetes.csv')
train, test = train_test_split(didata, test_size = 0.3, random_state=30)
trainLabel = np.asarray(train['Outcome'])
trainData = np.asarray(train.drop('Outcome',1))
testLabel = np.asarray(test['Outcome'])
testData = np.asarray(test.drop('Outcome',1))
diabetesCheck = svm.SVC(kernel='poly', C=1.0 , gamma=1)
diabetesCheck.fit(trainData, trainLabel)
accuracy = diabetesCheck.score(testData, testLabel)
print(accuracy*100)