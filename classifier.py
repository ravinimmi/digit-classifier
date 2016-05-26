import csv
import numpy as np
import pandas as pd
from sklearn import linear_model

def read_training_data():
    train = pd.read_csv('train.csv')
    YX = train.values
    X = YX[:,1:]
    Y = np.ravel(YX[:,:1])
    return X, Y

def read_test_data():
    test = pd.read_csv('test.csv')
    X_test = test.values
    return X_test

def generate_submission(X, Y, X_test):
    logreg = linear_model.LogisticRegression()
    print "Training..."
    logreg.fit(X, Y)
    print "Predicting..."
    Y_predicted = logreg.predict(X_test)
    df = pd.DataFrame({
                    'ImageId': range(1, len(X_test)+1), 
                    'Label'  : Y_predicted
                })
    df.to_csv('submission.csv', index=False)

X, Y = read_training_data()
X_test = read_test_data()
generate_submission(X, Y, X_test)
