from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
import sklearn
import pandas as pd
import numpy as np

# # Synthetic dataset
# np.random.seed(0)
# X, y = sklearn.datasets.make_moons(200, noise=0.20)

data = load_breast_cancer()
print(data.keys())
# print(data['feature_names'])
# print(data['target_names'])
X= data['data']
y= data['target']
print(y.shape, X.shape)

#Plot the data with first two columns as features and y as target label
plt.scatter(X[:,0],X[:,1],s=40,c=y)
plt.show()
# ML Split
X_train, X_test, y_train, y_test = train_test_split (X,y,test_size = 0.2)
# Train the  classifier
clf = KNeighborsClassifier()
# clf =LogisticRegressionCV()
clf.fit(X_train, y_train)
print(clf.score(X_test,y_test))

# ## Setting up the data into a pandas dataframe to look at correlation among features
# columns_data = np.concatenate([X,y[:,None]],axis=1)
# columns_names = np.concatenate([data['feature_names'],["class_labels"]])
# df =pd.DataFrame(columns_data,columns=columns_names)
# print(df.head(5))

# import seaborn as sns
# corr = df.corr()
# sns.heatmap(corr,cmap ="coolwarm",annot=True,annot_kws={"fontsize": 8})
# plt.tight_layout()
# plt.show()

# plt.scatter(df['mean radius'],df['mean concave points'],s=40,c=df['class_labels'],cmap=plt.cm.Spectral)
# plt.show()

