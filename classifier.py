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

def read_test_data():
    csvfile = open('test.csv')
    csvreader = csv.reader(csvfile)
    rows = []
    for row in csvreader:
        rows.append(row)
    csvfile.close()
    rows = np.array(rows)
    X_test = rows[1:,:]
    X_test = X_test.astype(int)
    return X_test

def generate_submission(X, Y, X_test):
    Y = np.ravel(Y)
    logreg = linear_model.LogisticRegression()
    print "Training..."
    logreg.fit(X, Y)
    print "Predicting..."
    Y_predicted = logreg.predict(X_test)
    submission_file = open('submission.csv', 'w')
    csvwriter = csv.writer(submission_file)
    csvwriter.writerow(['ImageId', 'Label'])
    for i in range(len(Y_predicted)):
        csvwriter.writerow([i+1, Y_predicted[i]])
    submission_file.close()

X, Y = read_data()
X_test = read_test_data()
generate_submission(X, Y, X_test)
