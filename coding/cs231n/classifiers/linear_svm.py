from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
   # Initialize loss and the gradient of W to zero.
    dW = np.zeros(W.shape)
    loss = 0.0
    num_classes = W.shape[1]
    num_train = X.shape[0]

    # Compute the data loss and the gradient.
    for i in range(num_train):  # For each image in training.
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        num_classes_greater_margin = 0

        for j in range(num_classes):  # For each calculated class score for this image.

        # Skip if images target class, no loss computed for that case.
            if j == y[i]:
                continue

            # Calculate our margin, delta = 1
            margin = scores[j] - correct_class_score + 1

            # Only calculate loss and gradient if margin condition is violated.
            if margin > 0:
                num_classes_greater_margin += 1
                # Gradient for non correct class weight.
                dW[:, j] = dW[:, j] + X[i, :]
                loss += margin

        # Gradient for correct class weight.
        dW[:, y[i]] = dW[:, y[i]] - X[i, :]*num_classes_greater_margin

    # Average our data loss across the batch.
    loss /= num_train

    # Add regularization loss to the data loss.
    loss += reg * np.sum(W * W)

    # Average our gradient across the batch and add gradient of regularization term.
    dW = dW /num_train + 2*reg *W
    return loss, dW


    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = X.dot(W)

    correct_class_scores = scores[range(num_train), list(y)].reshape(-1,1) #(N, 1)
    margins = np.maximum(0, scores - correct_class_scores +1)

    margins[range(num_train), list(y)] = 0

    loss = np.sum(margins) / num_train + 0.5 * reg * np.sum(W * W)
    # pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    coeff_mat = np.zeros((num_train, num_classes))
    coeff_mat[margins > 0] = 1
    coeff_mat[range(num_train), list(y)] = 0
    coeff_mat[range(num_train), list(y)] = -np.sum(coeff_mat, axis=1)

    dW = (X.T).dot(coeff_mat)
    dW = dW/num_train + reg*W
    # pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
