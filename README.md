# Linear-Regression-implementation-using-only-Numpy-gradient-descent-approach


This is part of Machine Learning course work the script cannot be posted here due to academic integrity. The Assignment Report is
made available for verification.


The implementation is explained here steps by step.

1) Taking the input samples and features for creating the weight and bias matrix
    n_s = X.shape[0]
    d_f = X.shape[1]
    m_o = y.shape[1]

2) Initializing the weight and bias matrix to set the starting point of the optimization algorithm
    self.weights = np.random.randn(d_f, m_o)*.05
    self.bias = np.random.randn(1, m_o)
    
3) Spliting the training data into train and validation set with 90 percent for training and 10% for validation

4) Assigning the best parameters into variable for comapring it with the updated values while optimizing

5) Gradient Descent

  1) Initializing the loss to be zero so that after each iteration the values will be zero.

  2) Selecting a mini-batch of data from the training set for use in the optimization process.

  3) Pre+diction and finding the loss

  4) Calculating the gradients of the cost function with respect the weights and the bias.
      dw = (2 / batch_size) * (X_batch.T).dot(y_pred - y_batch) 
      db = (2 / batch_size) * np.sum(y_pred - y_batch)
      
  5) Updating weights and bias
      self.weights -= learning_rate * dw
      self.bias -= learning_rate * db
      
  6) Appending the averaged loss of each batch to a list for visualization purpose

  7) Predicting values for the target values with the best values of the weights

  8) Calculates the MSE of on the validation set values

  9) Comparing validation loss with the best loss so far

## Note
**This repository serves as a demonstration of my work for the Machine learning coursework. Kindly refrain from using the exact code provided here in any academic assignments. Its purpose is to provide a reference and facilitate learning for others.**
          
  11) Checking if the patience count becomes equal to or more than the desired patience count training gets stopped.
  
  12) Saving best weights and bias
  
6) Predicting with the final weights and bias on the testing data.

7) Importing Iris dataset and trying out different sets of models on by taking different combinations of data.
