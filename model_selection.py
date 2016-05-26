import csv
import numpy as np
import pandas as pd
from sklearn import linear_model

def read_data():
    train = pd.read_csv('train.csv')
    YX = train.values
    return YX

def split_dataset(YX):
    np.random.shuffle(YX)
    N = len(YX)
    YX_train = YX[:N*8/10, :]
    YX_test  = YX[N*8/10:, :]
    X = YX_train[:, 1:]
    Y = np.ravel(YX_train[:, :1])
    X_test = YX_test[:, 1:]
    Y_test = np.ravel(YX_test[:, :1])
    return X, Y, X_test, Y_test
    
def classify(X, Y, X_test, Y_test):
    logreg = linear_model.LogisticRegression()
    print "Training..."
    logreg.fit(X, Y)
    print "Predicting..."
    Y_predicted = logreg.predict(X_test)
    matches = (Y_predicted == Y_test)
    print "Accuracy = {accuracy:.2f}%".format(accuracy=100*sum(matches)/len(X_test))

YX = read_data()
# YX = YX[:1000]
X, Y, X_test, Y_test = split_dataset(YX)
classify(X, Y, X_test, Y_test)
