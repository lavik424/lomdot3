import numpy as np
import pandas as pd

# imports for classifier estimation
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score



from operator import itemgetter





def clfMetricCalculator(clf,X:pd.DataFrame,Y:pd.DataFrame,numOfSplits=4):
    """
    Function to estimate classifier's performance
    :param clf: classifier to train and test on. Should have fit and predict methods
    :param X: samples data set
    :param Y: labels correspond to X
    :param numOfSplits: k for Kfold
    :return: map of accuracy,f1,precision,recall and confusion matrix in a tuple
    """
    totalAccuracy = 0
    totalRecall = 0
    totalF1 = 0
    totalPrecision = 0
    numOflabels = Y.iloc[:,0].nunique() # Y should contain only one column - label column
    totalConfusion = np.zeros((numOflabels, numOflabels))
    X = X.select_dtypes(include=[np.number])
    partiesLabels = Y.iloc[:,0].unique()
    # run kfold on clf
    kf = KFold(n_splits=numOfSplits, shuffle=True)
    for train_index, test_index in kf.split(X):
        # split the data to train set and validation set:
        features_train, features_test = X.iloc[train_index], X.iloc[test_index]
        classification_train, classification_test = Y.iloc[train_index], Y.iloc[test_index]

        # train the clf on X set
        clf.fit(features_train, classification_train)
        # test the tree on validation set
        pred = clf.predict(features_test)

        totalAccuracy += accuracy_score(classification_test, pred)
        totalF1 += f1_score(classification_test, pred, average='micro')
        totalPrecision += precision_score(classification_test, pred, average='micro')
        totalRecall += recall_score(classification_test, pred, average='micro')
        totalConfusion += confusion_matrix(classification_test.values, pred, labels=partiesLabels)


    # calculate metric and confusion matrix
    totalAccuracy = totalAccuracy / numOfSplits
    totalF1 = totalF1 / numOfSplits
    totalPrecision = totalPrecision / numOfSplits
    totalRecall = totalRecall / numOfSplits

    metricMap = {'Accuracy':totalAccuracy, 'F1':totalF1, 'Precision':totalPrecision, 'Recall':totalRecall}
    totalConfusion = np.rint(totalConfusion).astype(int)

    print('Total Accuracy of tree is:',totalAccuracy)

    return metricMap,totalConfusion