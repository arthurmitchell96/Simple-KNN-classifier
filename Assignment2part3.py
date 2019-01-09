import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import sklearn
#import data
Filename = 'Assignment item 2 Dataset.csv'
Assignment_data = pd.read_csv(Filename)
Assignment_data.head()
data = np.array(Assignment_data)
#shuffle rows to randomise data
np.random.shuffle(data)
#create y values
y = data[:,0]
#create x values
x = data[:,[1,2,3,4,5,6,7,8,9,10]]
sklearn.preprocessing.normalize(x)
#split data randomly
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.1)
#classify data
clf10 = MLPClassifier(hidden_layer_sizes = (10), activation = 'logistic', max_iter = 1000)
#fit data
Z10 = clf10.fit(xTrain,yTrain)
#predict values for x test
Y = clf10.predict_proba(xTest)
#score predictions
scoreANN10 = clf10.score(xTest,yTest)
#create nearest neighbors model
nb5 = KNeighborsClassifier(n_neighbors = 5)
#fit model to data
fitNN5 = nb5.fit(xTrain,yTrain)
#score data
scoreNN5 = nb5.score(xTest, yTest)
#plt.scatter(xTrain[:, 0], xTrain[:, 1], c=yTrain)
#plt.plot()
compare = svm.SVC(kernel = 'linear', C=1).fit(xTrain,yTrain)
cscore = compare.score(xTest,yTest)