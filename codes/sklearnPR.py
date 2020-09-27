# Author: Tim Head <betatim@gmail.com>
#
# License: BSD 3 clause
import pandas as pd
import numpy as np
np.random.seed(10)

import matplotlib.pyplot as plt
from pylab import *
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import make_pipeline


def loadData(file):
    
    dataSet = pd.read_table(file)
    labelColor = dataSet['DEF_PRTY']
    label = [0]*len(labelColor)
    for row in range(len(labelColor)):
        if labelColor[row] == 'RED':
            label[row] = 1

    Features = dataSet[['DEF_AMPLTD','DEF_LGTH']]
    Features = np.array(Features)
    Features = Features.tolist()
    
    num = len(Features)

    perce = int(num*0.8)

    trainX = Features[0:perce]
    trainY = label[0:perce]
    testX  = Features[perce::]
    testY  = label[perce::]

    return trainX, testX, trainY, testY


if __name__ == '__main__':
    files = ['lineThreeAll.txt', 'lineThreeXlevel.txt', 'lineThreeSurface.txt', 'lineThreeDip.txt']
    lables = ['Line 3', 'Line 3 - Xlevel', 'Line 3 - Surface', 'Line 3 - Dip']
    markers = ['o','>','+','*']
    colors = ['b',  'r', 'lime', 'k']
    sizes = []
    n_estimator = 10

    for ii in range(4):
        X_train, X_test, y_train, y_test = loadData(files[ii])
        X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train, y_train, test_size=0.2)
        rf = RandomForestClassifier(n_estimators=n_estimator)
        rf.fit(X_train, y_train)
        y_score = rf.predict_proba(X_test)[:,1]
        p, r, _ = precision_recall_curve(y_test, y_score)
        plt.step(r, p, color=colors[ii], marker=markers[ii], where='post',label=lables[ii])
        #plt.fill_between(r, p, step='post', alpha=0.2, color='b')

    #------------------------------------------------------------------   
    font = {'family': 'serif', 'weight':'normal', 'size':20}
    plt.gca().set_xlabel('Recall', font, labelpad=15)
    plt.gca().set_ylabel('Precision', font, labelpad=15)
    plt.tick_params(labelsize=20)
    #plt.xticks(fontsize=20,fontproperties='serif')
    #plt.yticks(fontsize=20,fontproperties='serif')
    plt.legend(loc='best', prop=font)
    #------------------------------------------------------------------
    plt.show()
    # It is important to train the ensemble of trees on a different subset
    # of the training data than the linear regression model to avoid
    # overfitting, in particular if the total number of leaves is
    # similar to the number of training samples
    


    # Supervised transformation based on random forests


    # The random forest model by itself

