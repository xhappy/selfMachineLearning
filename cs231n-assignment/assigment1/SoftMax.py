
import numpy as np


def softmax_loss_naive(W, X, y, reg):
    '''
    Softmax loss function, naive implementation with loops.
    
    Input have dimension D, there are C classes, and we operate on minibatches of N examples.
    
    Inputs:
    - W: A numpy array of shape (D, C) containing weights
    - X: A numpy array of shape (N, D) containing a minibatch of data
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means that X[i] has lable C. where 0 <=c < C
    - reg: (float) regularization strength
    
    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of some shape as W
    '''
    
    # Initilization the loss and gradient to zero
    loss = 0.0
    deltW = np.zeros_like(W)
    
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_classes = W.shape[1]
    num_train = X.shape[0]
    
    for i in range (num_train):
        scores = X[i].dot(W)  # 1 * 10   XW
        correct_class_scores = y[i]  # correct label
        exp_scores = np.zeros_like(scores)  
        row_sum = 0
        for j in range(num_classes):
            exp_scores[j] = np.exp(scores[j])  # exp(XW) 1*10 
            row_sum += exp_scores[j] # sum of exp(XW) for 10 classes
        loss += -np.log(exp_scores[correct_class_scores]/row_sum)
        
        # compute deltW loops:
        for k in range(num_classes):
            if (k != correct_class_scores):
                deltW[:, k] += exp_scores[k]/row_sum*X[i]
            else:
                deltW[:, correct_class_scores] += (exp_scores[k]/row_sum - 1)*X[i]
    
    
    loss /= num_train
    loss += 0.5 * reg * np.sum(W*W) # regularization item
    
    deltW /= num_train
    deltW += reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    
    return loss, deltW
    
def softmax_loss_vectorized(W, X, y, reg):
    """
    SoftMax loss function, vectorized version.
    Inputs and outputs are the same as softmax_loss_naive.
    """
    
    # Initialize the loss and gradient to zero
    loss = 0.0
    deltW = np.zeros_like(W)
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    scores = X.dot(W)
    exp_scores=np.exp(scores)
    row_sum = exp_scores.sum(axis=1)
    row_sum = row_sum.reshape((num_train, 1))
    
    # compute loss
    norm_exp_scores = exp_scores / row_sum
    row_index = np.arange(num_train)
    data_loss = norm_exp_scores[row_index, y].sum()
    loss = data_loss / num_train + 0.5 * reg * np.sum(W*W)
    norm_exp_scores[row_index, y] = -1
    
    deltW = X.T.dot(norm_exp_scores)
    deltW = deltW/num_train + reg * W
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    
    return loss, deltW