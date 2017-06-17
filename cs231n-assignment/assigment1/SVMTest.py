
from CifarDataUtil import CifarDataUtil
from SVM import SVM
from linear_classifier import LinearSVM

from gradient_check import *
import numpy as np
import matplotlib.pyplot as plt
import time
import math


def cross_validation (X_train, y_train, X_val, y_val):
    #############################################################################################
    # Use the validation set to tune hyperparameters (regularization strength and
    # learning rate). You should experiment with different ranges for the learning
    # rates and regularization strengths; if you are careful you should be able to
    # get a classification accuracy of about 0.4 on the validation set.    
    #############################################################################################
    learning_rates = [1e-7, 5e-5]
    regularization_strengths = [5e4, 1e5]
    
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the fraction
    # of data points that are correctly classified.
    results = {}
    best_val = -1   # The highest validation accuracy that we have seen so far.
    best_svm = None # The LinearSVM object that achieved the highest validation rate.
    
    ################################################################################
    # TODO:                                                                        #
    # Write code that chooses the best hyperparameters by tuning on the validation #
    # set. For each combination of hyperparameters, train a linear SVM on the      #
    # training set, compute its accuracy on the training and validation sets, and  #
    # store these numbers in the results dictionary. In addition, store the best   #
    # validation accuracy in best_val and the LinearSVM object that achieves this  #
    # accuracy in best_svm.                                                        #
    #                                                                              #
    # Hint: You should use a small value for num_iters as you develop your         #
    # validation code so that the SVMs don't take much time to train; once you are #
    # confident that your validation code works, you should rerun the validation   #
    # code with a larger value for num_iters.                                      #
    ################################################################################
    iters = 2000 #100
    for lr in learning_rates:
        for rs in regularization_strengths:
            svm = LinearSVM()
            svm.train(X_train, y_train, learning_rate=lr, reg=rs, num_iters=iters)
    
            y_train_pred = svm.predict(X_train)
            acc_train = np.mean(y_train == y_train_pred)
            y_val_pred = svm.predict(X_val)
            acc_val = np.mean(y_val == y_val_pred)
    
            results[(lr, rs)] = (acc_train, acc_val)
    
            if best_val < acc_val:
                best_val = acc_val
                best_svm = svm
    ################################################################################
    #                              END OF YOUR CODE                                #
    ################################################################################
    
    # Print out results.
    for lr, reg in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
                    lr, reg, train_accuracy, val_accuracy))
    
    print('best validation accuracy achieved during cross-validation: %f' % best_val)
    
    return results, best_svm

def visualizeCrossValidation(results):
    # Visualize the cross-validation results
    x_scatter = [math.log10(x[0]) for x in results]
    y_scatter = [math.log10(x[1]) for x in results]
    
    # plot training accuracy
    marker_size = 100
    colors = [results[x][0] for x in results]
    plt.subplot(2, 1, 1)
    plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
    plt.colorbar()
    plt.xlabel('log learning rate')
    plt.ylabel('log regularization strength')
    plt.title('CIFAR-10 training accuracy')
    
    # plot validation accuracy
    colors = [results[x][1] for x in results] # default size of markers is 20
    plt.subplot(2, 1, 2)
    plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
    plt.colorbar()
    plt.xlabel('log learning rate')
    plt.ylabel('log regularization strength')
    plt.title('CIFAR-10 validation accuracy')
    plt.show()

def testSetOnBestSVM(X_test, y_test, best_svm):
    # Evaluate the best svm on test set
    y_test_pred = best_svm.predict(X_test)
    test_accuracy = np.mean(y_test == y_test_pred)
    print('linear SVM on raw pixels final test set accuracy: %f' % test_accuracy)    

def visualizeWeightForAllClasses(best_svm):
    # Visualize the learned weights for each class.
    # Depending on your choice of learning rate and regularization strength, these may
    # or may not be nice to look at.
    w = best_svm.W[:-1,:] # strip out the bias
    w = w.reshape(32, 32, 3, 10)
    w_min, w_max = np.min(w), np.max(w)
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(10):
        plt.subplot(2, 5, i + 1)
          
        # Rescale the weights to be between 0 and 255
        wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
        plt.imshow(wimg.astype('uint8'))
        plt.axis('off')
        plt.title(classes[i])    
        
    plt.show()

if __name__ == '__main__':
    
    #####################################################################
    # Data Loading and Preprocessing:
    #####################################################################
    filePath = 'C:/Code/PythonCode/cases/source/cs231n/assignment/assignment1_2017/cs231n/datasets/cifar-10-batches-py'
    cifarDataObj = CifarDataUtil(filePath)
    X_train, y_train, X_test, y_test = cifarDataObj.loadData()
    
    # reshape the data to be: 1000*32*32*3
    X_train, X_test = cifarDataObj.reShapeDataAsImage(X_train, X_test)
    
    print('X_train.shape: ', X_train.shape)
    print('X_test.shape: ', X_test.shape)
    print('----------')
    
    #####################################################################
    # Split the data into train, val, and test sets:
    #####################################################################    
    # Split the data into train, val, and test sets. In addition we will
    # create a small development set as a subset of the training data;
    # we can use this for development so our code runs faster.
    num_training = 49000
    num_validation = 1000
    num_test = 1000
    num_dev = 500
    
    # Our validation set will be num_validation points from the original
    # training set.
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    
    # Our training set will be the first num_train points from the original
    # training set.
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    
    # We will also make a development set, which is a small subset of
    # the training set.
    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    y_dev = y_train[mask]
    
    # We use the first num_test points of the original test set as our
    # test set.
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]
    
    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Validation data shape: ', X_val.shape)
    print('Validation labels shape: ', y_val.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)
    print('---------')
    
    #####################################################################
    # Preprocessing: reshape the image data into rows:
    #####################################################################
    # Preprocessing: reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))
    
    # As a sanity check, print out the shapes of the data
    print('Training data shape: ', X_train.shape)
    print('Validation data shape: ', X_val.shape)
    print('Test data shape: ', X_test.shape)
    print('dev data shape: ', X_dev.shape)
    print('---------')
    
    #####################################################################
    # Preprocessing: subtract the mean image:
    #####################################################################    
    # Preprocessing: subtract the mean image
    # first: compute the image mean based on the training data
    mean_image = np.mean(X_train, axis=0)
    
    # second: subtract the mean image from train and test data
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image
    
    # third: append the bias dimension of ones (i.e. bias trick) so that our SVM
    # only has to worry about optimizing a single weight matrix W.
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])
    
    print(X_train.shape, X_val.shape, X_test.shape, X_dev.shape)
    print('------------')
    
    # generate a random SVM weight matrix of small numbers
    W = np.random.randn(3073, 10) * 0.0001
    ticNaive = time.time()
    svmTrainObj = SVM(W, X_train, y_train, 0.000005)
    loss, grad = svmTrainObj.svm_loss_naive()
    tocNaive = time.time()
    print('X_train.shape: ', X_train.shape, ' loss: ', loss, ' grad: ', grad, ' grad.shape: ', grad.shape)
    
    ticVector = time.time()
    loss2, grad2 = svmTrainObj.svm_loss_vectorized()
    tocVector = time.time()
    print('X_train.shape: ', X_train.shape, ' loss2: ', loss2, ' grad2: ', grad2, ' grad2.shape: ', grad2.shape)
    print('Naive cost time: ', tocNaive-ticNaive)
    print('Vector cost time: ', tocVector - ticVector)
    
    # Numerically compute the gradient along several randomly chosen dimensions, and
    # compare them with your analytically computed gradient. The numbers should match
    # almost exactly along all dimensions.f
    svmDevObj1 = SVM(W, X_dev, y_dev, 0.0)
    # Compute the loss and its gradient at W.
    loss, grad = svmDevObj1.svm_loss_naive()
    
    f = lambda w: svmDevObj1.svm_loss_naive()[0]
    grad_numerical = grad_check_sparse(f, W, grad)
    print('grad_numerical: ', grad_numerical)
    
    # do the gradient check once again with regularization turned on
    # you didn't forget the regularization gradient did you?
    svmDevObj2 = SVM(W, X_dev, y_dev, 5e1)
    loss, grad = svmDevObj2.svm_loss_naive()
    f = lambda w: svmDevObj2.svm_loss_naive()[0]
    grad_numerical = grad_check_sparse(f, W, grad)
    print('grad_numerical: ', grad_numerical)
    
    
    
    #####################
    linearSVM = LinearSVM()
    tic = time.time()
    loss_hist = linearSVM.train(X_train, y_train, learning_rate = 1e-7, reg = 2.5e4, num_iters = 1500, verbose = True)
    toc = time.time()
    print('That took %fs' % (toc - tic))
    
    # A useful debugging strategy is to plot the loss as a function of
    # iteration number:
    plt.plot(loss_hist)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.show()
    
    # Write the LinearSVM.predict function and evaluate the performance on both the
    # training and validation set
    y_train_pred = linearSVM.predict(X_train)
    print('training accuracy: %f' % (np.mean(y_train == y_train_pred), ))
    y_val_pred = linearSVM.predict(X_val)
    print('validation accuracy: %f' % (np.mean(y_val == y_val_pred), ))    
    
    results, best_svm = cross_validation (X_train, y_train, X_val, y_val)
    visualizeCrossValidation(results)
    
    testSetOnBestSVM(X_test, y_test, best_svm)
    
    visualizeWeightForAllClasses(best_svm)