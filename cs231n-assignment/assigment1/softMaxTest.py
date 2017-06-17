
import numpy as np
import matplotlib.pyplot as plt
import time

from CifarDataUtil import CifarDataUtil
from linear_classifier import Softmax
from SoftMax import *
from gradient_check import *

def getCifarDataForSoftmax(num_training = 49000, num_validation = 1000, num_test = 1000, num_dev = 500):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the linear classifier. These are the same steps as we used for the
    SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data    
    
    filePath = 'C:/Code/PythonCode/cases/source/cs231n/assignment/assignment1_2017/cs231n/datasets/cifar-10-batches-py'
    cifarDataUtilObj = CifarDataUtil(filePath)
    X_train, y_train, X_test, y_test = cifarDataUtilObj.loadData()
    X_train, X_test = cifarDataUtilObj.reShapeDataAsImage(X_train, X_test)
    
    # subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]
    
    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    y_dev = y_train[mask]
    
    ##########################################################################
    # pre-processing data: reshape the image data into rows
    ##########################################################################
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))
    
    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis = 0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image
    
    # add bias dimension and transform into columns
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])    
    
    return X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev

def cross_validataion(X_train, y_train, X_val, y_val):
    """
    Use the validation set to tune hyper-parameter (regularization strength and learning rate)
    You should experiment with different ranges for the learning rates and regularization strengths
    If you are careful, you should be able to get a classification accuracy of over 0.35 on the validation set
    """
    results = {}
    best_val = -1
    best_softmax = None
    learning_rate = [1e-7, 5e-7]
    regularization_strengths=[2.5e4, 5e4]
    iters = 1500
    
    for lr in learning_rate:
        for rs in regularization_strengths:
            softmax = Softmax()
            softmax.train(X_train, y_train, learning_rate = lr, reg = rs, num_iters= iters)
            Tr_pred = softmax.predict(X_train)
            acc_train = np.mean(y_train==Tr_pred)
            Val_pred = softmax.predict(X_val)
            acc_val = np.mean(y_val == Val_pred)
            results[(lr, rs)] = (acc_train, acc_val)
            if best_val < acc_val:
                best_val = acc_val
                best_softmax = softmax
    # Print out result:
    for lr, reg in sorted(results):
        train_accuracy, val_accuracy = results[(lr, rs)]
        print('lr %e reg %e train accuracy: %f val_accuracy: %f' %(lr, rs, train_accuracy, val_accuracy))
    print('best validation accuracy achieved during cross-validation: %f' %(best_val))
    
    return best_softmax
    
def evaluate_best_softmax_on_test(best_softmax):
    Ts_pred = best_softmax.predict(X_test)
    test_accuracy = np.mean(y_test == Ts_pred)   #around 37.4%
    print('SoftMax on raw pixel of CIFAR-10 final test set accuracy: %f' % test_accuracy)

def visualizeWeightForEachClass(best_softmax):
    w = best_softmax.W[:-1, :]  #strip out the bias
    w = w.reshape(32, 32, 3, 10)
    
    w_min, w_max = np.min(w), np.max(w)
    
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(10):
        plt.subplot(2, 5, i+1)
        
        # Rescale the weights to be between 0 and 255
        wimg = 255.0 * (w[:, :, :, i].squeeze() -w_min )/(w_max-w_min)
        plt.imshow(wimg.astype('uint8'))
        plt.axis('off')
        plt.title(classes[i])
    plt.show()
    

if __name__ == "__main__":
    
    X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = getCifarDataForSoftmax(
        num_training=49000, 
        num_validation=1000, 
        num_test=10000, 
        num_dev=500)
    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Validation data shape: ', X_val.shape)
    print('Validation labels shape: ', y_val.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)
    print('dev data shape: ', X_dev.shape)
    print('dev labels shape: ', y_dev.shape)    
    
    # First implement the naive softmax loss function with nested loops in softmax_loss_naive function on file SoftMax.py.
    # Generate a random softmax weight martix and use it to compute the loss.
    W = np.random.randn(3073, 10) * 0.001
    loss, grad = softmax_loss_naive(W, X_train, y_train, 0.0)
    
    # As a rough sanity check, our loss should be something close to -log(0,1)
    print('loss: %f' % loss)
    print('sanity check: %f' % (-np.log(0.1)))
    
    # complete the implementation of the softmax_loss_naive and implement a (naive) version of the gradient that uses nested loops
    loss, grad = softmax_loss_naive(W, X_dev, y_dev, 0.0)
    print('loss: ', loss)
    print('grad: ', grad)
    
    # As we did for the SVM, use numeric gradient checking as a debugging tool
    # The numeric gradient should be close to the analytic gradient
    f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 0.0)[0]
    grad_numerical = grad_check_sparse(f, W, grad, 10)
    
    # similar to SVM case, do another gradient check with regularization:
    loss, grad = softmax_loss_naive(W, X_dev, y_dev, 5e1)
    f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 5e1)[0]
    grad_numerical = grad_check_sparse(f, W, grad, 10)
    
    # Now that we have a naive implementation of the softmax loss function and its gradient,
    # implement a vectorized version in softmax_loss_vectorized
    # The two versions should compute the same results, but the vectorized version should be much faster
    tic = time.time()
    loss_naive, grad_naive = softmax_loss_naive(W, X_train, y_train, 0.000005)
    toc = time.time()
    print('naive loss: %e computed in %fs' %(loss_naive, toc-tic))
    
    tic = time.time()
    loss_vectorized, grad_vectorized = softmax_loss_vectorized(W, X_train, y_train, 0.000005)
    toc = time.time()
    print('vectorized loss: %e computed in %fs' %(loss_naive, toc-tic))
    
    # As we did for the SVM, we use the Frobenius norm to compare the two versions of the gradient
    grad_difference = np.linalg.norm(grad_naive - grad_vectorized, ord= 'fro')
    print('Loss difference: %f' % np.abs(loss_naive-loss_vectorized))
    print('Gradient difference: %f' % grad_difference)
    
    best_softmax = cross_validataion(X_train, y_train, X_val, y_val)
    evaluate_best_softmax_on_test(best_softmax)
    
    visualizeWeightForEachClass(best_softmax)