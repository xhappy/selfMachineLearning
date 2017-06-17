
import numpy as np

def ReLU(x):
    ''''ReLU non-linearity function'''
    return np.maximum(0, x)

class TwoLayerNet(object):
    '''
    A two-layer fully connected neural network. 
    The net has an input dimension of D, a hidden layer dimension of H, and perfomrs classification over C classes
    The network has the architecutre:
    input - fully connected layer - ReLU - fully connected layer - softmax
    The output of the second fully-connected layer are the scores for each class.
    '''
    
    def __init__(self, input_size, hidden_size, output_size, std = 1e-4):
        self.parameter = {}
        self.parameter['W1'] = std * np.random.randn(input_size, hidden_size)
        self.parameter['b1'] = np.zeros(hidden_size)
        self.parameter['W2'] = std * np.random.randn(hidden_size, output_size)
        self.parameter['b2'] = np.zeros(output_size)
    
    def loss(self, X, y=None, reg = 0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.
        
        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.
        
        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].
        
        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.parameter['W1'], self.parameter['b1']
        W2, b2 = self.parameter['W2'], self.parameter['b2']
        N, D = X.shape
        
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################        
        #####################################################################################
        # Compute the forward pass:
        #####################################################################################
        scores = None
        h1 = ReLU(np.dot(X, W1) + b1) # hidden layer 1 (N, H)
        out = np.dot(h1, W2) + b2
        scores = out
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        
        if y is None:
            return scores
        
        # Compute the loss
        loss = None        
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################
        # Considering the Numeric Stability:
        scores_max = np.max(scores, axis = 1, keepdims = True)  # (N, 1)
        
        # Computer the class probabilities:
        exp_scores = np.exp(scores - scores_max)                # (N , C)
        probs = exp_scores / np.sum(exp_scores, axis = 1, keepdims= True)   # (N , C)
        
        # cross-entropy loss and L2-regularization:
        correct_logprobs = -np.log(probs[range(N), y])          # (N, 1)
        data_loss = np.sum(correct_logprobs)/N
        reg_loss = 0.5 * reg * np.sum(W1*W1) + 0.5 * reg * np.sum(W2 * W2)
        loss = data_loss + reg_loss         
        
        ###########num_train = X.shape[0]
        ###########exp_scores=np.exp(scores)
        ###########row_sum = exp_scores.sum(axis=1).reshape((num_train, 1))
        
        ############ compute loss
        ###########norm_exp_scores = exp_scores / row_sum
        ###########row_index = np.arange(num_train)
        ###########data_loss = -np.log(norm_exp_scores[row_index, y]).sum()
        ###########loss = data_loss / num_train + 0.5 * reg * np.sum(W1*W1) + 0.5 * reg * np.sum(W2*W2)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        
        #####################################################################################
        # Compute the backward pass: compute the gradients
        #####################################################################################        
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        # refer link: http://blog.csdn.net/rtygbwwwerr/article/details/42147783
        probs[np.arange(N), y] += -1  
        grads['W2'] =  1.0/N * h1.T.dot(probs) + reg* W2  
        grads['b2'] =  1.0/N * np.sum(probs, axis = 0)   
        
        dh1 = probs.dot(W2.T)
        dh1_ReLU = (X.dot(W1)+b1 >0)*dh1
        
        grads['W1'] = 1.0/N * X.T.dot(dh1_ReLU) + reg* W1  
        grads['b1'] = 1.0/N * np.sum(dh1_ReLU, axis = 0)        
        
        
        # Compute the gradient of scores, please refer: http://ufldl.stanford.edu/wiki/index.php/Softmax%E5%9B%9E%E5%BD%92
        ######dscores = probs             # (N, C)
        ######dscores[range(N), y] -= 1;
        ######dscores /= N
        
        ####### Backprop into W2 and b2:
        ######deltW2 = np.dot(h1.T, dscores)  # (N, C)
        ######deltB2 = np.sum(dscores, axis = 0, keepdims=True) # (1, C)
        
        ####### Backprop into hidden layer
        ######deltH1 = np.dot(dscores, W2.T)  # (N, H)
        
        ####### Backprop into ReLU non-linearity 
        ######deltH1[h1 < 0] = 0
        
        ####### Backprop into W1 and b1
        ######deltW1 = np.dot(X.T, deltH1)
        ######deltB1 = np.sum(deltH1, axis = 0, keepdims=True)
        
        ####### Add the regularization gradient contribution
        ######deltW2 += reg * W2
        ######deltW1 += reg * W1
        
        ######grads['W1'] = deltW1
        ######grads['b1'] = deltB1
        ######grads['W2'] = deltW2
        ######grads['b2'] = deltB2
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################        
        
        return loss, grads
        
        
    def train(self, X, y, X_val, y_val, learning_rate=1e-3, learning_rate_decay=0.95, reg=1e-5, num_iters=100, batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.
    
        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        self.hyper_params = {}
        self.hyper_params['learning_rate'] = learning_rate
        self.hyper_params['reg'] = reg
        self.hyper_params['batch_size'] = batch_size
        self.hyper_params['hidden_size'] = self.parameter['W1'].shape[1]
        self.hyper_params['num_iter'] = num_iters
    
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)
    
        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []
    
        for it in range(num_iters):
            X_batch = None
            y_batch = None
    
            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            batch_inx = np.random.choice(num_train, batch_size)
            X_batch = X[batch_inx,:]
            y_batch = y[batch_inx]
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################
    
            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)
    
            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.parameter)   #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            self.parameter['W1'] -= learning_rate * grads['W1']
            self.parameter['b1'] -= learning_rate * grads['b1']
            self.parameter['W2'] -= learning_rate * grads['W2']
            self.parameter['b2'] -= learning_rate * grads['b2']
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################
    
            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))
    
            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                # Decay learning rate
                learning_rate *= learning_rate_decay
    
        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }
        
    def predict(self, X):    
        """
        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to        
             classify.    
        Returns:    
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of           
                  the elements of X. For all i, y_pred[i] = c means that X[i] is 
                  predicted to have class c, where 0 <= c < C.   
        """    
        y_pred = None    
        h1 = ReLU(np.dot(X, self.parameter['W1']) + self.parameter['b1'])    
        scores = np.dot(h1, self.parameter['W2']) + self.parameter['b2']    
        y_pred = np.argmax(scores, axis=1)    
    
        return y_pred