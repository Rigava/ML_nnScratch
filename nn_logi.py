import numpy as np
import copy
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from functions import predict,optimize,propagate,initialize_with_zeros,sigmoid,model

# #Breast cancer Dataset
data = load_breast_cancer()
print(data.keys())
X= data['data']
y= data['target']
#Plot the datasets
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
plt.show()
#Train Test split
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


logistic_regression_model = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)

# Plot learning curve (with costs)
costs = np.squeeze(logistic_regression_model['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(logistic_regression_model["learning_rate"]))
plt.show()

learning_rates = [0.0009, 0.0005, 0.0001]
models = {}

for lr in learning_rates:
    print ("Training a model with learning rate: " + str(lr))
    models[str(lr)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1500, learning_rate=lr, print_cost=False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for lr in learning_rates:
    plt.plot(np.squeeze(models[str(lr)]["costs"]), label=str(models[str(lr)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations (hundreds)')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()