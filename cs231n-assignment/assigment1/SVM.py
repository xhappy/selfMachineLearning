import numpy as np

class SVM(object):
    def __init__(self, W, X_train, y_train, regular):
        '''
        Inputs have dimension D, there are C classes, and we operate on minibatches of N examples.
        
        Inputs:
        - W: A numpy array of shape (D, C) containing weights.
        - X_train: A numpy array of shape (N, D) containing a minibatch of data.
        - y_train: A numpy array of shape (N,) containing training labels; y[i] = c means
          that X[i] has label c, where 0 <= c < C.
        - regular: (float) regularization strength
        '''
        
        self.W = W
        self.X_train = X_train
        self.y_train = y_train
        self.regular = regular
    
    def svm_loss_naive(self):
        '''
         Returns a tuple of:
         - loss as single float
         - gradient with respect to weights W; an array of same shape as W
        '''
        deltW = np.zeros(self.W.shape) # initialize the gradient as zero, shape: (3073, 10)
        
        # compute the loss and gradient
        num_classes = self.W.shape[1]    # size: 10
        num_train = self.X_train.shape[0]   #size: 490000
        
        loss = 0
        for i in range(num_train):
            scores = self.X_train[i].dot(self.W)
            correct_class_scores = scores[self.y_train[i]]
            for j in range(num_classes):
                if (j == self.y_train[i]):
                    continue
                margin = scores[j] - correct_class_scores + 1  # node delta = 1
                if margin > 0:
                    loss += margin
                    
                    #############################################################
                    # compute the gradient deltW, Sj - Si + 1  # note delta = 1    #
                    # http://blog.csdn.net/yc461515457/article/details/51921607 #
                    #############################################################
                    deltW[:, j] += self.X_train[i, :].T  # j != self.y_train[i]
                    deltW[:, self.y_train[i]] -= self.X_train[i, :].T
                    
        # Right now the loass is a sum over all training examples, but we want it
        # to be an average instead, so we divide by num_train
        loss /= num_train
        deltW /= num_train
        
        #############################################################
        # Add regularization to the loss                            #
        #############################################################
        loss += 0.5 * self.regular * np.sum(self.W * self.W)
        deltW += self.regular * self.W
        
        #############################################################################
        # TODO:                                                                     #
        # Compute the gradient of the loss function and store it deltW.             #
        # Rather that first computing the loss and then computing the derivative,   #
        # it may be simpler to compute the derivative at the same time that the     #
        # loss is being computed. As a result you may need to modify some of the    #
        # code above to compute the gradient.                                       #
        #############################################################################
        
        return loss, deltW
    
    def svm_loss_vectorized(self):
        """
        Structured SVM loss function, vectorized implementation.
      
        Inputs and outputs are the same as svm_loss_naive.
        """
        loss = 0.0
        deltW = np.zeros(self.W.shape) # initialize the gradient as zero
        
        #############################################################################
        # TODO:                                                                     #
        # Implement a vectorized version of the structured SVM loss, storing the    #
        # result in loss.                                                           #
        #############################################################################
        XW = self.X_train.dot(self.W)
        num_train = self.X_train.shape[0]
        Sy = np.zeros(num_train)
        
        for i in range(num_train):
            Sy[i] = XW[i, self.y_train[i]]
        
        #print('Sy.shape: ', Sy.shape)  # Sy.shape: num_train
        WX = XW.T -Sy + 1
        
        for i in range(num_train):
            WX[self.y_train[i], i] -= 1
        
        loss = np.sum(WX[WX>0])
        loss /= num_train
        
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        
        #############################################################################
        # TODO:                                                                     #
        # Implement a vectorized version of the gradient for the structured SVM     #
        # loss, storing the result in dW.                                           #
        #                                                                           #
        # Hint: Instead of computing the gradient from scratch, it may be easier    #
        # to reuse some of the intermediate values that you used to compute the     #
        # loss.                                                                     #
        #############################################################################
        # keep only positive elements
        XW = WX.T
        num_classes = self.W.shape[1]
        for i in range(num_train):
            for j in range(num_classes):
                if (XW[i, j] > 0):
                    deltW[:, j] += self.X_train[i, :].T
                    deltW[:, self.y_train[i]] -= self.X_train[i, :].T
        deltW /= num_train
        deltW += self.regular * self.W
        
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        
        return loss, deltW  