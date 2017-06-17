import pickle
import numpy as np
import matplotlib.pyplot as plt

class CifarDataUtil(object):
    
    def __init__(self, path):
        self.path = path
        
    def loadBatchData(self, fileName):
        with open(self.path+'/'+fileName, 'rb') as batchFile:
            pickleObj = pickle.load(batchFile, encoding='latin1')
            xTrainingData = pickleObj['data']
            yTrainingLabel = pickleObj['labels']
            yTrainingLabel = np.array(yTrainingLabel)
        return xTrainingData, yTrainingLabel
    
    def loadData(self):
        xTrainingData = []
        yTrainingData = []
        
        for i in range(1, 6):
            fileName = 'data_batch_%d'%(i)
            xData, yData = self.loadBatchData(fileName)
            xTrainingData.append(xData)
            yTrainingData.append(yData)
        
        xTrainData = np.concatenate(xTrainingData)
        yTrainData = np.concatenate(yTrainingData)
        del xData
        del yData
        fileName = 'test_batch'
        xTestingData, yTestingData = self.loadBatchData(fileName)
        
        return xTrainData, yTrainData, xTestingData, yTestingData
    
    def reShapeDataAsImage(self, xTrain, xTest):
        
        return xTrain.reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float"), xTest.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    
    
    def showSingleTypeImage(self, xTrain, yTrain, option, number):
        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        print(classes[option])
        optionData = np.flatnonzero(yTrain == option)
        rowNum = number/10 if number%10 == 0 else number/10+1
        for row in range(0, int(rowNum)):
            for column in range(0, 10):
                imageIndex = row * 10 + column + 1
                plt.subplot(rowNum, 10,  imageIndex)
                plt.axis('off')
                plt.imshow(xTrain[optionData[imageIndex]])
        plt.show()
        
        

if __name__=='__main__':
    filePath = 'C:/Code/PythonCode/cases/source/cs231n/assignment/assignment1_2017/cs231n/datasets/cifar-10-batches-py'
    cifarDataObj = CifarDataUtil(filePath)
    xTrain, yTrain, xTest, yTest = cifarDataObj.loadData()
    print(xTrain.size, yTrain.size, xTest.size, yTest.size)
    
    xTrain, xTest = cifarDataObj.reShapeDataAsImage(xTrain, xTest)
    print(xTest)
    
    #cifarDataObj.showSingleTypeImage(xTrain, yTrain, 0, 200)
    
    
    