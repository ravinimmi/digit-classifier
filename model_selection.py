import csv
import numpy as np
from sklearn import linear_model

def read_data():
    csvfile = open('train.csv')
    csvreader = csv.reader(csvfile)
    rows = []
    for row in csvreader:
        rows.append(row)
    csvfile.close()
    rows = np.array(rows)
    XY = rows[1:,:]
    XY = XY.astype(int)
    X = XY[:,1:]
    Y = XY[:,:1]
    return X, Y

def split_dataset(X, Y):
    XY = np.hstack((X, Y))
    np.random.shuffle(XY)
    N = len(XY)
    XY_train = XY[:N*8/10, :]
    XY_test  = XY[N*8/10:, :]
    X = XY_train[:, :-1]
    Y = np.ravel(XY_train[:, -1:])
    X_test = XY_test[:, :-1]
    Y_test = np.ravel(XY_test[:, -1:])
    return X, Y, X_test, Y_test
    
def classify(X, Y, X_test, Y_test):
    logreg = linear_model.LogisticRegression()
    print "Training..."
    logreg.fit(X, Y)
    print "Predicting..."
    Y_predicted = logreg.predict(X_test)
    matches = (Y_predicted == Y_test)
    print "Accuracy = {accuracy:.2f}%".format(accuracy=100*sum(matches)/len(X_test))

X, Y = read_data()
X = X[:1000]
Y = Y[:1000]
X, Y, X_test, Y_test = split_dataset(X, Y)
classify(X, Y, X_test, Y_test)
