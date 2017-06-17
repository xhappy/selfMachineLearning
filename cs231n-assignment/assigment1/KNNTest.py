from CifarDataUtil import CifarDataUtil
from KNearestNeighbor import KNearestNeighbor
import numpy as np

import time


if __name__=='__main__':
    filePath = 'C:/Code/PythonCode/cases/source/cs231n/assignment/assignment1_2017/cs231n/datasets/cifar-10-batches-py'
    cifarDataObj = CifarDataUtil(filePath)
    xTrain, yTrain, xTest, yTest = cifarDataObj.loadData()
    #print(xTrain.size, yTrain.size, xTest.size, yTest.size)
    
    xTrain, xTest = cifarDataObj.reShapeDataAsImage(xTrain, xTest)
    #print(xTest)
    
    ################################## start: select the size of training data and test data#############################
    numTraining = 5000   #training size 
    mask = list(range(numTraining))
    xTrainTemp = xTrain[mask]
    yTrainTemp = yTrain[mask]
    
    numTest = 500
    mask = list(range(numTest))
    xTestTemp = xTest[mask]
    yTestTemp = yTest[mask]
    ################################## end: select the size of training data and test data#############################
    
    np.set_printoptions(precision=2, threshold=20000000000)
    
    knnClassifier = KNearestNeighbor(xTrainTemp, yTrainTemp, xTestTemp)
    #knnClassifier = KNearestNeighbor(xTrain, yTrain, xTest)
    
    ################################## start: compute the distance using 3 methods #############################
    startTime = time.clock()
    distanceNoLoop = knnClassifier.computerDistanceNoLoop()
    #knnClassifier.saveDistanceToFile(distanceNoLoop, filePath+'/' + 'distanceNoLoop')
    print(distanceNoLoop.shape)
    print("NO Loop completed.")
    endTime = time.clock()
    print("cost time: %d"%(endTime-startTime))    
    
    #startTime = time.clock()
    #distance1Loop = knnClassifier.computerDistanceWith1Loop()
    #knnClassifier.saveDistanceToFile(distance1Loop, filePath+'/' + 'distance1Loop')
    #print(distance1Loop.shape)
    #print("1 Loop completed.")
    #endTime = time.clock()
    #print("cost time: %d"%(endTime-startTime))
    
    #startTime = time.clock()
    #distance2Loop = knnClassifier.computerDistanceWith2Loop()
    #knnClassifier.saveDistanceToFile(distance2Loop, filePath+'/' + 'distance2Loop')
    #print(distance2Loop.shape)
    #print("2 Loop completed.")
    #endTime = time.clock()
    #print("cost time: %d"%(endTime-startTime))
    ################################## end: compute the distance using 3 methods #############################
    
    # predict test set.
    distance = distanceNoLoop
    yTest = knnClassifier.predict_lables(distance, 100)
    
    # cross validation to check the effect of KNN algorithm
    knnClassifier.crossValidation()