
#Linear Regression can be solved by two methods, Normal Equation which is not used
#pertaining to the time complexity of calculation inverses as we have already seen.
#Other method is Gradient Descent which we are not covering here but is the most 
#commonly used method. 
#Normal Equation is defined as:
#   beta = (X^T.X)^-1.X^T.y     ### I hope I can use latex when I'm making the 
#                                    course to write the equation well

#X and y are pre-defined. Using the given equation, find the betas for X

import numpy as np

X = np.random.rand(100,1)
y = 3 * X+np.random.randn(100,1)


inv_XTX = np.linalg.inv(X.T.dot(X))

beta = inv_XTX.dot(X.T).dot(y)