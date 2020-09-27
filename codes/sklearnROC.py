# Author: Tim Head <betatim@gmail.com>
#
# License: BSD 3 clause
import pandas as pd
import numpy as np
np.random.seed(10)

import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
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
    files = ['lineFourAll.txt', 'lineFourXlevel.txt', 'lineFourSurface.txt', 'lineFourDip.txt']
    lables = ['Line 4', 'Line 4 - Xlevel', 'Line 4 - Surface', 'Line 4 - Dip']
    markers = ['o','>','+','*']
    colors = ['b',  'r', 'lime', 'k']
    n_estimator = 10

    for ii in range(4):
        X_train, X_test, y_train, y_test = loadData(files[ii])
        X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train, y_train, test_size=0.2)
        rf = RandomForestClassifier(n_estimators=n_estimator)
        rf_enc = OneHotEncoder()
        rf.fit(X_train, y_train)
        rf_enc.fit(rf.apply(X_train))
        y_pred_rf = rf.predict_proba(X_test)[:, 1]
        fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
        plt.plot(fpr_rf, tpr_rf, marker=markers[ii], color=colors[ii], label=lables[ii])
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.tick_params(labelsize=20)
    #------------------------------------------------------------------   
    font = {'family': 'serif', 'weight':'normal', 'size':20}
    plt.gca().set_xlabel('False Positive Rate', font, labelpad=15)
    plt.gca().set_ylabel('True Positive Rate', font, labelpad=15)
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

