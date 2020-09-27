'''
Experiments on data of line 1
lineOneXlevel

'''
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier 
import matplotlib.pyplot as plt   
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
from sklearn import metrics
import graphviz
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


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

    return trainX, trainY, testX, testY

def trainingModel(trainX, trainY, testX, testY):
    print('begin training!')

    #SVM = svm.SVC()
    #GNB = GaussianNB()
    #SGD = SGDClassifier(loss='hinge', penalty="l2")
    DeT = tree.DecisionTreeClassifier()
    #RF  = RandomForestClassifier(n_estimators=10) 
    #NN  = MLPClassifier(solver='lbfgs', alpha=1e-05,hidden_layer_sizes=(5,2),random_state=1)    

    #trainedSVM = SVM.fit(trainX, trainY)
    #trainedGNB = GNB.fit(trainX, trainY)
    #trainedSGD = SGD.fit(trainX, trainY)
    DeT.fit(trainX, trainY)

    y_pred_DeT = DeT.predict_proba(testX)[:, 1]
    fpr_DeT, tpr_DeT, _ = metrics.roc_curve(testY, y_pred_DeT)
    #trainedRF  = RF.fit(trainX, trainY)
    #trainedNN  = NN.fit(trainX, trainY)

    print('finish training!')

    return fpr_DeT, tpr_DeT

def testingModel(testX,testY, model):
    preY = model.predict(testX)
    count = 0
    for i in range(len(preY)):
        if testY[i] == preY[i]:
            count += 1
    acc = count/float(len(preY))
    
    return preY, acc

def Plot_decision_surface(trainX, trainY):
    clf = tree.DecisionTreeClassifier().fit(trainX,trainY)
    x_min, x_max = trainX[:,0].min() - 1, trainX[:,0].max() + 1
    y_min, y_max = trainX[:,1].min() - 1, trainX[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    #------------------------------------------------------------------
    font = {'family': 'serif', 'weight':'normal', 'size':20}
    plt.gca().set_xlabel('Defect Amplitude', font, labelpad=15)
    plt.gca().set_ylabel('Defect Length', font, labelpad=15)
    plt.xticks(fontsize=20,fontproperties='serif')
    plt.yticks(fontsize=20,fontproperties='serif')
    #------------------------------------------------------------------

    for i, color in zip(range(2), 'rb'):
        idx = np.where(trainY == i)
        plt.scatter(trainX[idx, 0], trainX[idx, 1], c=color, cmap=plt.cm.RdYlBu, edgecolor='black',s=15)
    plt.axis('tight')
    plt.show()

def plot_tree(clf):
    dot_data = tree.export_graphviz(clf,out_file='tree.dot',
                                  feature_names=['DEF_AMPLTD','DEF_LGTH'],
                                  class_names=['red','yellow'])
    graph = graphviz.Source(dot_data)

    return graph

def ROC_plot(fpr_DeT, tpr_DeT, LABLE):    
    plt.plot(fpr_DeT, tpr_DeT, label=LABLE)
    #------------------------------------------------------------------   
    font = {'family': 'serif', 'weight':'normal', 'size':20}
    plt.gca().set_xlabel('False Positive Rate', font, labelpad=15)
    plt.gca().set_ylabel('True Positive Rate', font, labelpad=15)
    plt.xticks(fontsize=20,fontproperties='serif')
    plt.yticks(fontsize=20,fontproperties='serif')
    plt.legend(loc='best', prop=font)
    #------------------------------------------------------------------
    

if __name__ == '__main__':


    #Plot_decision_surface(np.array(trainX), np.array(trainY))
    files = ['lineFourAll.txt', 'lineFourXlevel.txt', 'lineFourSurface.txt', 'lineFourDip.txt']
    lables = ['Line 4', 'Line 4 - Xlevel', 'Line 4 - Surface', 'Line 4 - Dip']

    for ii in range(4):
        trainX, trainY, testX, testY = loadData(files[ii])
        fpr_DeT, tpr_DeT = trainingModel(trainX, trainY, testX, testY)
        ROC_plot(fpr_DeT, tpr_DeT, lables[ii])
    
    plt.show()

    #p = metrics.precision_score(testY, preY_DeT, average='macro')
    #r = metrics.recall_score(testY, preY_DeT, average='micro')

    #print(p)
    #print(r)
    
    

    #out = pd.DataFrame([list(testY), list(preY_DeT)]).T




'''

    #preY_SVM, acc_SVM = testingModel(testX, testY, SVM)
    #preY_GNB, acc_GNB = testingModel(testX, testY, GNB)
    #preY_SGD, acc_SGD = testingModel(testX, testY, SGD)
    #preY_SVM, acc_NN  = testingModel(testX, testY, NN)
    
    #print('acc of svm is %f' %(acc_SVM))
    #print('acc of GNB is %f' %(acc_GNB))
    #print('acc of SGD is %f' %(acc_SGD))
    #print('acc of DeT is %f' %(acc_DeT))
    #print('acc of RF  is %f' %(acc_RF))
    #print('acc of NN  is %f' %(acc_NN))

    #totalOne = sum(testY)
    #ratioOne = totalOne/float(len(testY))
    #print('ratioOne is %f' %(ratioOne))
    
'''
