import numpy as np
import pandas as pd

# imports for classifier estimation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import tree


def clfMetricCalculator(clf,X:pd.DataFrame,Y:pd.DataFrame,avgMethod='weighted',numOfSplits=4):
    """
    Function to estimate classifier's performance
    :param clf: classifier to train and test on. Should have fit and predict methods
    :param X: samples data set
    :param Y: labels correspond to X
    :param avgMethod: use 'weighted' for winning party 'macro' for ..
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
        pred = pd.DataFrame(pred, index=classification_test.index) # convert into col vector and match index

        totalAccuracy += accuracy_score(classification_test, pred)
        totalF1 += f1_score(classification_test, pred, average=avgMethod)
        totalPrecision += precision_score(classification_test, pred, average=avgMethod)
        totalRecall += recall_score(classification_test, pred, average=avgMethod)
        totalConfusion += confusion_matrix(classification_test.values, pred, labels=partiesLabels)


    # calculate metric and confusion matrix
    totalAccuracy = totalAccuracy / numOfSplits
    totalF1 = totalF1 / numOfSplits
    totalPrecision = totalPrecision / numOfSplits
    totalRecall = totalRecall / numOfSplits

    metricMap = {'Accuracy':totalAccuracy, 'F1':totalF1, 'Precision':totalPrecision, 'Recall':totalRecall}
    totalConfusion = np.rint(totalConfusion).astype(int)

    # print('Total Accuracy of tree is:',totalAccuracy)

    return metricMap,totalConfusion





def randomHyperParamsForClassifier(params):
    """

    :param params: shold be a dict with name of hyper param and list of 2 index0 - lower bound, index1 upper bound
    :return: dict with the name and the randomize value
    """
    return {p:(np.random.randint(value[0],value[1]) if isinstance(value[0],int)
                         else np.random.uniform(value[0],value[1])) for p,value in params.items()}


def hyperParamsForTree(x_train,y_train,avgMethod='weighted'):

    hyperParams = {'min_impurity_split':[0.0,1.0], 'min_samples_split':[0.0001,0.05]}
    bestSetOfParams = {'min_impurity_split':0, 'min_samples_split':0}
    # hyperParams = {'min_impurity_decrease':[0.0,1.0], 'min_samples_split':[0.0001,0.05]}
    # bestSetOfParams = {'min_impurity_decrease':0, 'min_samples_split':0}
    bestAccuracy = 0
    for i in range(100):
        currSetOfParams = randomHyperParamsForClassifier(hyperParams)
        estimator = tree.DecisionTreeClassifier(criterion="entropy",**currSetOfParams)
        metric, _ = clfMetricCalculator(estimator, x_train, y_train,avgMethod)
        # print('with this set:\n',currSetOfParams)
        # print('we achived those scores\n',metric)
        if bestAccuracy < metric['Accuracy']:
            bestAccuracy = metric['Accuracy']
            bestSetOfParams = currSetOfParams

    return bestSetOfParams,bestAccuracy


def hyperParamsForKNN(x_train,y_train,avgMethod='weighted'):

    hyperParams = {'n_neighbors':[3,30]}
    bestSetOfParams = {'n_neighbors': 0}
    bestAccuracy = 0
    for i in range(3,30):
        currSetOfParams = {'n_neighbors': i}
        estimator = KNeighborsClassifier(**currSetOfParams)
        metric, _ = clfMetricCalculator(estimator, x_train, y_train,avgMethod)
        # print('with this set:\n',currSetOfParams)
        # print('we achived those scores\n',metric)
        if bestAccuracy < metric['Accuracy']:
            bestAccuracy = metric['Accuracy']
            bestSetOfParams = currSetOfParams

    return bestSetOfParams,bestAccuracy



def hyperParamsForRF(x_train,y_train,avgMethod='weighted'):

    hyperParams = {'min_impurity_split':[0.0,1.0], 'min_samples_split':[0.0001,0.05],'n_estimators':[10,150]}
    bestSetOfParams = {'min_impurity_split':0, 'min_samples_split':0,'n_estimators':0}
    # hyperParams = {'min_impurity_decrease':[0.0,1.0], 'min_samples_split':[0.0001,0.05]}
    # bestSetOfParams = {'min_impurity_decrease':0, 'min_samples_split':0}
    bestAccuracy = 0
    for i in range(100):
        print('Starting iteration number:',i)
        currSetOfParams = randomHyperParamsForClassifier(hyperParams)
        estimator = RandomForestClassifier(criterion="entropy",**currSetOfParams)
        metric, _ = clfMetricCalculator(estimator, x_train, y_train,avgMethod)
        # print('with this set:\n',currSetOfParams)
        # print('we achived those scores\n',metric)
        if bestAccuracy < metric['Accuracy']:
            bestAccuracy = metric['Accuracy']
            bestSetOfParams = currSetOfParams

    return bestSetOfParams,bestAccuracy



def trainWithBestHyperparamsKNN(methodDict,x_train,y_train):
    """

    :param methodDict:
    :param x_train:
    :param y_train:
    :return:
    """
    print('\tKNN')
    for method in methodDict:
        estimator = KNeighborsClassifier(**methodDict[method])
        metric,confusionMatrix = clfMetricCalculator(estimator,x_train,y_train,avgMethod=method)
        print('\tIn',method,'method')
        print(metric)
        # make nice confusion matrix with labels
        partiesLabels = y_train.iloc[:, 0].unique()
        confusionMatrix = pd.DataFrame(confusionMatrix,columns=partiesLabels,index=partiesLabels)
        print(confusionMatrix)


def trainWithBestHyperparamsTree(methodDict, x_train, y_train):
    """

    :param methodDict:
    :param x_train:
    :param y_train:
    :return:
    """
    print('\tTREE')
    for method in methodDict:
        estimator = tree.DecisionTreeClassifier(criterion="entropy", **methodDict[method])
        metric, confusionMatrix = clfMetricCalculator(estimator, x_train, y_train, avgMethod=method)
        print('\tIn', method, 'method')
        print(metric)
        # make nice confusion matrix with labels
        partiesLabels = y_train.iloc[:, 0].unique()
        confusionMatrix = pd.DataFrame(confusionMatrix, columns=partiesLabels, index=partiesLabels)
        print(confusionMatrix)


