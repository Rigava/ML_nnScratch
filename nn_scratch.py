import numpy as np
import copy
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

data = load_breast_cancer()
print(data.keys())
print(data['target_names'])
X= data['data']
y= data['target']
print(X.shape,y.shape)
#Plot the datasets
# plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
# plt.show()
# Train Test split
X_train_org, X_test_org, y_train_org, y_test_org = train_test_split (X,y,test_size = 0.2)
m_train = X_train_org.shape[0]
m_test = X_test_org.shape[0]
num_px = X_train_org.shape[1]
print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Feature of each example: num_px = " + str(num_px))
#Standardisation of dataset
min_max_scaler = MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train_org)
X_test = min_max_scaler.transform(X_test_org)
# Reshape the training and test examples
train_set_x = X_train.reshape(X_train.shape[0],-1).T
test_set_x = X_test.reshape(X_test.shape[0],-1).T
train_set_y = y_train_org.reshape(y_train_org.shape[0],-1).T
test_set_y = y_test_org.reshape(y_test_org.shape[0],-1).T
print ("train_set_x_flatten shape: " + str(train_set_x.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

def initialize_with_zeros(dim):
    """
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    """
    w = np.zeros((dim,1))
    b = 0.0
    return w, b
def sigmoid(z):
    """    Arguments:
    z -- A scalar or numpy array of any size.
    """
    s = 1/(1+np.exp(-z))
    return s
def propagate(w, b, X, Y):
    """
    Arguments:
    w -- weights, a numpy array of size (num_px, 1)
    b -- bias, a scalar
    X -- data of size (num_px, number of examples)
    Y -- true "label" vector (containing 0 if benign, 1 if malignant) of size (1, number of examples)
    """
    m = X.shape[1]
    # FORWARD PROPAGATION (FROM X TO COST)
    z = np.dot(w.T,X)+b
    A = sigmoid(z)
    cost = 1/m*np.sum((-Y*np.log(A)-(1-Y)*np.log(1-A)))
    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = 1/m*(np.dot(X,(A-Y).T))
    db = 1/m*(np.sum(A-Y))
    cost = np.squeeze(np.array(cost))
    
    grads = {"dw": dw,
             "db": db}
    return grads, cost
def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    """
    
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    
    costs = []
    
    for i in range(num_iterations):
        grads,cost = propagate(w,b,X,Y)
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        w = w - learning_rate*dw
        b = b - learning_rate*db
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
            # Print the cost every 100 training iterations
            if print_cost:
                print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs
def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
   
    '''
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T,X)+b)
        
    for i in range(m):
        
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if A[0, i] > 0.5:
            Y_prediction[0,i] = 1 
        else:
            Y_prediction[0,i] = 0 
        
    return Y_prediction
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to True to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """   
 
    # initialize parameters with zeros
    dim = X_train.shape[0]
    w, b = initialize_with_zeros(dim)
    # Gradient descent 
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations=2000, learning_rate=0.5, print_cost=False)
    # Retrieve parameters w and b from dictionary "params"
    w = params["w"]
    b = params["b"]
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)

    # Print train/test Errors
    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d
logistic_regression_model = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)

# Plot learning curve (with costs)
costs = np.squeeze(logistic_regression_model['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(logistic_regression_model["learning_rate"]))
plt.show()