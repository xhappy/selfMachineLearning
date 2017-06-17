

import numpy as np
import matplotlib.pyplot as plt
import time

from CifarDataUtil import CifarDataUtil
from gradient_check import *
from neural_net import *
from gradient_check import eval_numerical_gradient
from vis_utils import *

def rel_error(x,y):
    """ returns relative error """
    return np.max(np.abs(x-y)/np.maximum(1e-8, np.abs(x) + np.abs(y)))

def getCifarDataForTwoLayerNet(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    filePath = 'C:/Code/PythonCode/cases/source/cs231n/assignment/assignment1_2017/cs231n/datasets/cifar-10-batches-py'
    cifarDataUtilObj = CifarDataUtil(filePath)
    X_train, y_train, X_test, y_test = cifarDataUtilObj.loadData()
    X_train, X_test = cifarDataUtilObj.reShapeDataAsImage(X_train, X_test)
        
    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":

    input_size = 4
    hidden_size = 10
    num_classes = 3
    num_inputs = 5
    
    def init_toy_model():
        np.random.seed(0)
        return TwoLayerNet(input_size, hidden_size, num_classes, std= 1e-1)
    
    def init_toy_data():
        np.random.seed(1)
        X = 10 * np.random.randn(num_inputs, input_size)
        y = np.array([0, 1, 2, 2, 1])
        
        return X, y
    
    net = init_toy_model()
    X, y = init_toy_data()
    
    # Implement the first part of the forward pass which uses the weights and biases to compute the scores for all inputs.
    scores = net.loss(X)
    print("Your scores: ")
    print(scores)
    print()
    print('correct scores:')
    correct_scores = np.asarray([
        [-0.81233741, -1.27654624, -0.70335995],
        [-0.17129677, -1.18803311, -0.47310444],
        [-0.51590475, -1.01354314, -0.8504215 ],
        [-0.15419291, -0.48629638, -0.52901952],
        [-0.00618733, -0.12435261, -0.15226949]])
    print(correct_scores)
    print()
    # The difference should be very small. We get < 1e-7
    print('Difference between your scores and correct scores:')
    print(np.abs(scores - correct_scores))
    print(np.sum(np.abs(scores - correct_scores)))   
    
    #####################################################################################################
    #In the same function, implement the second part that computes the data and regularizaion los
    #####################################################################################################    
    loss, grads = net.loss(X, y=y, reg=0.1)
    correct_loss = 1.30378789133
    print('compute loss: ', loss)
    print('correct loss: ', correct_loss)
    print('grads: ', grads)
    
    # should be very small, we get < 1e-12
    print('Difference between your loss and correct loss:')
    print(np.sum(np.abs(loss - correct_loss)))
    
    loss, grads = net.loss(X, y=y, reg=0.1)
    
    # Use numeric gradient checking to check your implementation of the backward pass. 
    # If your implementation is correct, the difference between the numeric and analytic
    # gradients should be less than 1e-8 for each of W1, W2, b1, and b2.
    # these should be less than 1e-8 or so
    for param_name in grads:
        f = lambda W: net.loss(X, y, reg = 0.1)[0]
        param_grad_num = eval_numerical_gradient(f, net.parameter[param_name], verbose=False)
        print('%s max relative error: %e'%(param_name, rel_error(param_grad_num, grads[param_name])))
    
    net = init_toy_model()
    stats = net.train(X, y, X, y,
                      learning_rate=1e-1, reg=1e-5,
                num_iters=100, verbose=False)
    
    print('Final training loss: ', stats['loss_history'][-1])
    
    # plot the loss history
    plt.plot(stats['loss_history'])
    plt.xlabel('iteration')
    plt.ylabel('training loss')
    plt.title('Training Loss history')
    plt.show()
    
    # Invoke the above function to get our data.
    X_train, y_train, X_val, y_val, X_test, y_test = getCifarDataForTwoLayerNet()
    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Validation data shape: ', X_val.shape)
    print('Validation labels shape: ', y_val.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)
    
    
    # To train our network we will use SGD with momentum. In addition, we will adjust the learning rate
    # with an exponential learning rate schedule as optimization proceeds; after each epoch, we will 
    # reduce the learning rate by multiplying it by a decay rate.    
    input_size = 32 * 32 * 3
    hidden_size = 50
    num_classes = 10
    net = TwoLayerNet(input_size, hidden_size, num_classes)
    
    # Train the network
    stats = net.train(X_train, y_train, X_val, y_val,
                      num_iters=1000, batch_size=200,
                learning_rate=1e-4, learning_rate_decay=0.95,
                reg=0.5, verbose=True)
    
    # Predict on the validation set
    val_acc = (net.predict(X_val) == y_val).mean()
    print('Validation accuracy: ', val_acc)
    
    '''
    With the default parameters we provided above, you should get a validation
    accuracy of about 0.29 on the validation set. This is not very good.
    
    One strategy for getting insight into what is wrong is to plot the loss 
    function and the accuracies on the training and validation sets during optimization.

    Another strategy is to visualize the weights that were learned in the first layer 
    of the network. In most neural networks trained on visual data, the first layer 
    weights typically show some visible structure when visualized.
    '''
    
    # Plot the loss function and train / validation accuracies
    plt.subplot(2, 1, 1)
    plt.plot(stats['loss_history'])
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    
    plt.subplot(2, 1, 2)
    plt.plot(stats['train_acc_history'], label='train')
    plt.plot(stats['val_acc_history'], label='val')
    plt.title('Classification accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Clasification accuracy')
    plt.show()
    
    #######################################################################
    # Visualize the weights of the network
    #######################################################################
    def show_net_weights(net):
        W1 = net.parameter['W1']
        W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
        plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
        plt.gca().axis('off')
        plt.show()
    
    show_net_weights(net)
    
    '''
    Tune your hyperparameters

    What is wrong?. Looking at the visualizations above, we see that the loss is decreasing more or less linearly, 
    which seems to suggest that the learning rate may be too low. Moreover, there is no gap between the training 
    and validation accuracy, suggesting that the model we used has low capacity, and that we should increase its 
    size. On the other hand, with a very large model we would expect to see more overfitting, which would manifest itself 
    as a very large gap between the training and validation accuracy.

    Tuning. Tuning the hyperparameters and developing intuition for how they affect the final performance is a large part of 
    using Neural Networks, so we want you to get a lot of practice. Below, you should experiment with different values of the 
    various hyperparameters, including hidden layer size, learning rate, numer of training epochs, and regularization strength. 
    You might also consider tuning the learning rate decay, but you should be able to get good performance using the default value.

    Approximate results. You should be aim to achieve a classification accuracy of greater than 48% on the validation set. 
    Our best network gets over 52% on the validation set.

    Experiment: You goal in this exercise is to get as good of a result on CIFAR-10 as you can, with a fully-connected Neural Network. 
    For every 1% above 52% on the Test set we will award you with one extra bonus point. Feel free implement your own techniques 
    (e.g. PCA to reduce dimensionality, or adding dropout, or adding features to the solver, etc.).
    '''
    best_net = None # store the best model into this 
    
    #################################################################################
    # TODO: Tune hyperparameters using the validation set. Store your best trained  #
    # model in best_net.                                                            #
    #                                                                               #
    # To help debug your network, it may help to use visualizations similar to the  #
    # ones we used above; these visualizations will have significant qualitative    #
    # differences from the ones we saw above for the poorly tuned network.          #
    #                                                                               #
    # Tweaking hyperparameters by hand can be fun, but you might find it useful to  #
    # write code to sweep through possible combinations of hyperparameters          #
    # automatically like we did on the previous exercises.                          #
    #################################################################################
    best_acc = -1
    input_size = 32 * 32 * 3
    
    best_stats = None
    
    #hidden_size_choice = [x*100+50 for x in xrange(11)]
    #reg_choice = [0.1, 0.5, 5, 15, 50, 100, 1000]
    #learning_rate_choice = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 1e-1, 1]
    #batch_size_choice = [8, 40, 80, 160, 500, 1000]
    
    
    hidden_size_choice = [400]
    learning_rate_choice = [3e-3]
    reg_choice = [0.02, 0.05, 0.1]
    batch_size_choice =[500]
    num_iters_choice = [5000]
    
    for batch_size_curr in batch_size_choice:
        for reg_cur in reg_choice:
            for learning_rate_curr in learning_rate_choice:
                for hidden_size_curr in hidden_size_choice:
                    for num_iters_curr in num_iters_choice:
                        print()
                        print("current training hidden_size:",hidden_size_curr)
                        print("current training learning_rate:",learning_rate_curr)
                        print("current training reg:",reg_cur)
                        print("current training batch_size:",batch_size_curr)
                        net = TwoLayerNet(input_size, hidden_size_curr, num_classes)
                        best_stats = net.train(X_train, y_train, X_val, y_val,
                                num_iters=num_iters_curr, batch_size=batch_size_curr,
                                learning_rate=learning_rate_curr, learning_rate_decay=0.95,
                                reg=reg_cur, verbose=True)
                        val_acc = (net.predict(X_val) == y_val).mean()
                        print("current val_acc:",val_acc)
                        if val_acc>best_acc:
                            best_acc = val_acc
                            best_net = net
                            best_stats = stats
                            print()
                            print("best_acc:",best_acc)
                            print("best hidden_size:",best_net.parameter['W1'].shape[1])
                            print("best learning_rate:",best_net.hyper_params['learning_rate'])
                            print("best reg:",best_net.hyper_params['reg'])
                            print("best batch_size:",best_net.hyper_params['batch_size'])
                            print()
    #################################################################################
    #                               END OF YOUR CODE                                #
    #################################################################################    
    
    #################################################################################
    # continue to debug the hyper-parameter, insert it by meself                    #
    #################################################################################
    test_net = TwoLayerNet(input_size, 450, num_classes)
    test_stats = test_net.train(X_train, y_train, X_val, y_val,
                                num_iters=5000, batch_size=500,
                           learning_rate=2e-3, learning_rate_decay=0.95,
                           reg=0.02, verbose=True)
    test_val_acc = (test_net.predict(X_val) == y_val).mean()
    print()
    print("test_acc:",test_val_acc)
    print("test hidden_size:",test_net.hyper_params['hidden_size'])
    print("test learning_rate:",test_net.hyper_params['learning_rate'])
    print("test reg:",test_net.hyper_params['reg'])
    print("test batch_size:",test_net.hyper_params['batch_size'])
    print("test num_iter:",test_net.hyper_params['num_iter'])
    
    
    # Plot the loss function and train / validation accuracies
    plt.subplot(2, 1, 1)
    plt.plot(test_stats['loss_history'])
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    
    plt.subplot(2, 1, 2)
    plt.plot(test_stats['train_acc_history'], label='train')
    plt.plot(test_stats['val_acc_history'], label='val')
    plt.title('Classification accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Clasification accuracy')
    plt.show()
    
    # visualize the weights of the best network
    show_net_weights(best_net)
    print("best hidden_size:",best_net.hyper_params['hidden_size'])
    print("learning_rate",best_net.hyper_params['learning_rate'])
    print("reg",best_net.hyper_params['reg'])
    print("batch_size",best_net.hyper_params['batch_size'])
    
    
    '''
    Run on the test set

    When you are done experimenting, you should evaluate your final trained network on the test set; you should get above 48%.

    We will give you extra bonus point for every 1% of accuracy above 52%.
    '''
    final_test_acc = (best_net.predict(X_test) == y_test).mean()
    print('Test accuracy: ', final_test_acc)
    test_acc = (test_net.predict(X_test) == y_test).mean()
    print('Test accuracy: ', test_acc)