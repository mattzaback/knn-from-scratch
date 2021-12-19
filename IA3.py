# -*- coding: utf-8 -*-
#*********************************************************************
#File name: IA3.py
#Author: Matthew Zaback
#Date: 10/12/2021
#Class: DSCI 440 ML
#Assignment: IA 3
#Purpose: To create and show the KNN algorithm from scratch
#**********************************************************************
import numpy as np
import pylab as pp
from numpy import genfromtxt
from sklearn import preprocessing

#Load train data
file = open("Downloads\knn_train.csv", "r")
my_data = genfromtxt('Downloads\knn_train.csv', delimiter=',')

rows = len(my_data)
columns = 31

trainFeat = np.empty ((int(rows), columns - 1))
trainLabel = np.empty ((int(rows), 1))

trainFeat = np.delete(my_data, 0, axis = 1)

r=0
while r < len(my_data):
    trainLabel[r][0] = my_data[r][0]
    r += 1
        
        
file.close()

#Load test data
file = open("Downloads\knn_test.csv", "r")
test_data = genfromtxt('Downloads\knn_test.csv', delimiter=',')

rows = len(test_data)

testFeat = np.empty((int(rows), columns - 1))
testLabel = np.empty((int(rows), 1))

testFeat = np.delete(test_data, 0, axis = 1)

r = 0
while r < len(test_data):
    testLabel[r][0] = test_data[r][0]
    r += 1
    
        
file.close()


#Normalize features and delete noisy features
trainFeat = np.delete(trainFeat, 3, axis = 1)
trainFeat = np.delete(trainFeat, 22, axis = 1)

testFeat = np.delete(testFeat, 3, axis = 1)
testFeat = np.delete(testFeat, 22, axis = 1)

nTrainFeat = preprocessing.normalize (trainFeat, axis = 0)
nTestFeat = preprocessing.normalize (testFeat, axis = 0)

#KNN
def euclidean_distance(v1, v2):    
    return np.sqrt(np.sum(v1 - v2)**2)
    

def k_nearest_neighbors(k, vToClassify, x, y):
    distances = list ()
    labels = list ()
    j = 0
    rows = len(x)
    while j < rows:
        dist = euclidean_distance(vToClassify, x[j])
        distances.append((j, dist))
        j += 1
        
    distances.sort(key = lambda tup: tup[1])
    neighbors = list ()
    i = 0
    while i < k:
        neighbors.append(distances[i][0])
        i += 1
        
    v=0
    while v < k:
        labels.append(int(y[neighbors[v]]))
        v+=1
    
    summation = 0
    m=0
    while m < len(labels):
        summation += labels[m]
        m+=1
    
    if summation > 0:
        m = 1
    else:
        m = -1
    
    return m
    #goal is to return label for x, vote of labels from k neighbors

#Implement knn 
trainError = np.zeros(25)
testError = np.zeros(25)
cvError = np.zeros(25)
folds = list ()
i = 0
k = 1
errorIndex = 0
while k < 50:
    i = 0
    while i < len(trainFeat):
        prediction = nTrainFeat[i]
        lblPrediction = k_nearest_neighbors(k, prediction, nTrainFeat, trainLabel)
        if lblPrediction != int(trainLabel[i]):
            trainError[errorIndex] += 1
        i+=1

   
    i = 0
    while i < len(testFeat):
        prediction = nTestFeat[i]
        lblPrediction = k_nearest_neighbors (k, prediction, nTrainFeat, trainLabel)
        if lblPrediction != int(testLabel[i]):
            testError[errorIndex] += 1
        i += 1
    
    folds = np.array_split(range(len(nTrainFeat)), 5)
    
    count = 0
    while count < 5:
        cvVal = nTrainFeat[folds[count]]
        cvTest = nTrainFeat
        cvTest = np.delete(cvTest, folds[count], axis = 0)
        i = 0
        while i < len(folds[count]):
            lblPrediction = k_nearest_neighbors(k, cvVal[i], cvTest, trainLabel)
            if lblPrediction != int(trainLabel[i]):
                cvError[errorIndex] += 1
            i += 1
            
        count += 1
    errorIndex += 1
    k+=2


#Put error numbers into percentages
i = 0
pTrainError = np.zeros(25)
pTestError = np.zeros(25)
pCVError = np.zeros(25)
while i < len(trainError):
    pTrainError[i] = trainError[i] / len(trainFeat)
    pTestError[i] = testError[i] / len(testFeat)
    pCVError[i] = cvError[i] / len(trainFeat)
    i+= 1




#Graph
kValues = np.arange(1, 50, 2)
pp.plot(kValues, pCVError, 'b', label = '5-fold')
pp.plot(kValues, pTrainError, 'r', label = 'Training')
pp.plot(kValues, pTestError, 'g', label = 'Test')
pp.title('K Values vs. Error')
pp.xlabel('K Values')
pp.ylabel('Error(%)')
pp.gca().invert_xaxis()
pp.legend()
pp.show ()



