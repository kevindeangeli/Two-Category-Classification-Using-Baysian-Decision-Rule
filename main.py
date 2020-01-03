'''
Created by: Kevin De Angeli
Email: kevindeangeli@utk.edu
Date: 2020-01-03

'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sym

#Data came in two files:
training_data = "synth.tr.txt"
test_data = "synth.te.txt"


def mahalanobis(x, y, Sigma):
    # calculate squared Mahalanobis distance
    # check dimension
    assert x.shape == y.shape and max(x.shape) == max(Sigma.shape)
    diff = x - y
    return np.dot(np.dot(diff, np.linalg.inv(Sigma)), diff)

def accuracy_score(y, y_model):
    """ return accuracy score """
    assert len(y) == len(y_model)
    return np.count_nonzero(y==y_model)/len(y)

def loadData():
    """
    x,y  == Trainning
    X,Y  == Testing
    """
    dataTrain = pd.read_csv(training_data, delim_whitespace=1, header=None).to_numpy()
    dataTest = pd.read_csv(test_data, delim_whitespace=1, header=None).to_numpy()
    columnsNum = dataTrain.shape[1]

    y = np.array(dataTrain)[:,-1]
    x = np.array(dataTrain)[:,0:columnsNum-1]
    Y = np.array(dataTest)[:,-1]
    X = np.array(dataTest)[:,0:columnsNum-1]

    return x,y,X,Y


class Bayes():
    #The prior should be passed as an array of size numOfClasses. Ex: [.5,.5] for 2 classes.
    def __init__(self,case=1, prior = None):
        self.case = case
        self.prior = prior



    def fit(self, x, y):
        self.covs_, self.means_ = {}, {}
        self.covsum_ = None
        self.classes_ = np.unique(y)     # get unique labels as dictionary items
        self.classn_ = len(self.classes_)

        for c in self.classes_:
            arr = x[y == c]
            self.covs_[c] = np.cov(np.transpose(arr))
            self.means_[c] = np.mean(arr, axis=0)  # mean along rows
            if self.covsum_ is None: #Assign the first covariance
                self.covsum_ = self.covs_[c]
            else: #add the other covariances so you can later take the average or the mean of the diagonals. See right below
                self.covsum_ += self.covs_[c]


        # used by case II
        self.covavg_ = self.covsum_ / self.classn_

        # used by case I
        self.varavg_ = np.sum(np.diagonal(self.covavg_)) / len(self.classes_) #Average of diagonals = average of variances




    def predict(self, x):
        # eval all data
        y = []
        disc = np.zeros(self.classn_)
        nr, _ = x.shape

        if self.prior is None:
            self.prior = np.full(self.classn_, 1 / self.classn_)



        for i in range(nr):
            for c in self.classes_:
                c = int(c)
                if self.case == 1:
                    edist2 = np.linalg.norm(self.means_[c] - x[i])
                    disc[c] = -edist2 / (2 * self.varavg_) + np.log(self.prior[c])
                elif self.case == 2:
                    mdist2 = mahalanobis(self.means_[c], x[i], self.covavg_)
                    disc[c] = -mdist2 / 2 + np.log(self.prior[c])
                elif self.case == 3:
                    mdist2 = mahalanobis(self.means_[c], x[i], self.covs_[c])
                    disc[c] = -mdist2 / 2 - np.log(np.linalg.det(self.covs_[c])) / 2 \
                              + np.log(self.prior[c])
                else:
                    print("Can only handle case numbers 1, 2, 3.")
                    sys.exit(1)
            y.append(disc.argmax())

        return y



def main():
    x,y,X,Y = loadData()
    model = Bayes(case=3, prior=[.5,.5])
    model.fit(x,y)
    predict = model.predict(X)
    accuracy = accuracy_score(Y, predict)
    print("Accuracy ", accuracy)



if __name__ == "__main__":
    main()
