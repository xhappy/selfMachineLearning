

import numpy as np
import matplotlib.pyplot as plt

class KNearestNeighbor(object):
    
    def __init__(self, xtrainData, ytrainData, testData):
        self.xTrain = np.reshape(xtrainData, (xtrainData.shape[0], -1))
        self.yTrain = ytrainData
        self.xTest  = np.reshape(testData, (testData.shape[0], -1))
        print(self.xTrain.shape, self.xTest.shape)

        
    def computerDistanceWith2Loop(self):
        trainImageNum = self.xTrain.shape[0]
        testImageNum = self.xTest.shape[0]
        distance = np.zeros((testImageNum, trainImageNum))
        
        for i in range(0, testImageNum):
            for j in range (0, trainImageNum):
                distance[i, j] = np.sqrt(np.dot(self.xTest[i] - self.xTrain[j], self.xTest[i] - self.xTrain[j]))
        
        return distance

    def computerDistanceWith1Loop(self):
        trainImageNum = self.xTrain.shape[0]
        testImageNum  = self.xTest.shape[0]
        
        distance = np.zeros((testImageNum, trainImageNum))
        
        for i in range(testImageNum):
            distance[i,:] = np.sqrt(np.sum(np.square(self.xTest[i] - self.xTrain), axis=1))
            
        return distance
    
    def computerDistanceNoLoop(self):
        trainImageNum = self.xTrain.shape[0]
        testImageNum  = self.xTest.shape[0]
    
        distance = np.zeros((testImageNum, trainImageNum))
        
        distance = np.sqrt(self.getNormMatrix(self.xTest, trainImageNum).T 
                       +  self.getNormMatrix(self.xTrain, testImageNum)
                       -  2 * np.dot(self.xTest, self.xTrain.T))
        
        #M = np.dot(self.xTest, self.xTrain.T)
        #print(M.shape)
        #te = np.square(self.xTest).sum(axis = 1)
        #print(te.shape)
        #tr = np.square(self.xTrain).sum(axis = 1)
        #print(tr.shape)
        #distance = np.sqrt(-2*M+tr+np.matrix(te).T) #tr add to line, te add to row
        
        #distance += np.sum(self.xTrain ** 2, axis=1).reshape(1, trainImageNum)
        #distance += np.sum(self.xTest ** 2, axis=1).reshape(testImageNum, 1) # reshape for broadcasting
        #distance -= 2 * np.dot(self.xTest, self.xTrain.T)
        
        return distance
    
    def getNormMatrix(self, data, lines_num):
        return np.ones((lines_num, 1)) * np.sum(np.square(data), axis = 1)
    
    def saveDistanceToFile(self, distance, fileName):
        np.savetxt(fileName, distance, fmt='%1.1f')
    
    def predict_lables(self, distance, k=1):
        '''
        Given a matrix of distance between test points and training points.
        pridect a label for each test point.
        
        Inputs:
        - distance: A numpy array of shape (num_test, num_train) where distance[i,j]
        gives the distance between the ith test point and the jth training point.
        
        Returns:
        - y: A numpy array of shape (num_test, ) containing predicted lables for the
        test data, where y[i] is the predicted lable for the test point X[i].
        '''
        
        numTest = distance.shape[0]
        y_pred=np.zeros(numTest)
        for i in range(numTest):
            # A list of length k storing the lables of the k nearest neighbors to 
            # the ith test point
            
            closest_y=[]
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            kids = np.argsort(distance[i])
            closest_y = self.yTrain[kids[:k]]
            
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            
            count = 0
            label = 0
            for j in closest_y:
                tmp = 0
                for kk in closest_y:
                    tmp += (kk==j)
                if tmp > count:
                    count = tmp
                    label = j
            y_pred[i] = label
        return y_pred
    
    def crossValidation(self):
        num_folds = 5
        k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
        X_train_folder = []
        y_train_folder = []
        
        ################################################################################
        # TODO:                                                                        #
        # Split up the training data into folds. After splitting, X_train_folds and    #
        # y_train_folds should each be lists of length num_folds, where                #
        # y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
        # Hint: Look up the numpy array_split function.                                #
        ################################################################################
        X_train_folds = np.array_split(self.xTrain, num_folds)
        y_train_folds = np.array_split(self.yTrain, num_folds)
        
        ################################################################################
        #                                 END OF YOUR CODE                             #
        ################################################################################
        
        # A dictionary holding the accuracies for different values of k that we find
        # when running cross-validation. After running cross-validation,
        # k_to_accuracies[k] should be a list of length num_folds giving the different
        # accuracy values that we found when using that value of k.
        k_to_accuracies = {}
        
        ################################################################################
        # TODO:                                                                        #
        # Perform k-fold cross validation to find the best value of k. For each        #
        # possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
        # where in each case you use all but one of the folds as training data and the #
        # last fold as a validation set. Store the accuracies for all fold and all     #
        # values of k in the k_to_accuracies dictionary.                               #
        ################################################################################
        for k in k_choices:
            k_to_accuracies[k] = np.zeros(num_folds)
            for i in range(num_folds):
                xTr = np.array(X_train_folds[:i] + X_train_folds[i+1:] )
                yTr = np.array(y_train_folds[:i] + y_train_folds[i+1:] )
                xTe = np.array(X_train_folds[i])
                yTe = np.array(y_train_folds[i])
                
                xTr = np.reshape(xTr, (xTr.shape[0]*xTr.shape[1], -1))
                yTr = np.reshape(yTr, yTr.shape[0]*yTr.shape[1])
            
                knnClassifier = KNearestNeighbor(xTr, yTr, xTe)
                distanceNoLoop = knnClassifier.computerDistanceNoLoop()

                yte_pred = knnClassifier.predict_lables(distanceNoLoop, k)
                
                num_correct = np.sum(yte_pred == yTe)
                accuracy = float(num_correct) / len(yTe)
                k_to_accuracies[k][i] = accuracy
        ################################################################################
        #                                 END OF YOUR CODE                             #
        ################################################################################
        
        # Print out the computed accuracies
        for k in sorted(k_to_accuracies):
            for accuracy in k_to_accuracies[k]:
                print('k = %d, accuracy = %f' % (k, accuracy))
        
        return k_to_accuracies