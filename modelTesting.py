import random

import numpy as np
import pandas as pd

# imports for classifier estimation

from sklearn.model_selection import KFold

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


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

        totalConfusion += confusion_matrix(classification_test.values, pred, labels=partiesLabels)

        totalAccuracy += accuracy_score(classification_test, pred)
        totalF1 += f1_score(classification_test, pred, average=avgMethod)
        totalPrecision += precision_score(classification_test, pred, average=avgMethod)
        totalRecall += recall_score(classification_test, pred, average=avgMethod)



    # calculate metric and confusion matrix
    totalAccuracy = totalAccuracy / numOfSplits
    totalF1 = totalF1 / numOfSplits
    totalPrecision = totalPrecision / numOfSplits
    totalRecall = totalRecall / numOfSplits

    metricMap = {'Accuracy':totalAccuracy, 'F1':totalF1, 'Precision':totalPrecision, 'Recall':totalRecall}
    totalConfusion = np.rint(totalConfusion).astype(int)

    # print('Total Accuracy of tree is:',totalAccuracy)

    clf.fit(X, Y)
    return metricMap, totalConfusion, clf





def randomHyperParamsForClassifier(params):
    """
    function to generate random number (int\float) for every hyperparams
    :param params: should be a dict with name of hyper param and list of 2 index0 - lower bound, index1 upper bound
    :return: dict with the name and the randomize value
    """
    return {p:(np.random.randint(value[0],value[1]) if isinstance(value[0],int)
                         else np.random.uniform(value[0],value[1])) for p,value in params.items()}




def hyperParamsForTreeOrRF(clfType,x_train,y_train,avgMethod='weighted'):
    """
    function that train and find best hyperparams for RF/Tree
    :param clfType: 'Tree' for tree or 'RF' for random forest
    :param x_train:
    :param y_train:
    :param avgMethod: from ['weighted','macro'] , differs from one classification to another
    :return:
    """
    if clfType == 'Tree':
        hyperParams = {'min_impurity_split':[0.0,1.0], 'min_samples_split':[0.0001,0.05]}
        bestSetOfParams = {'min_impurity_split':0, 'min_samples_split':0}
    if clfType == 'RF':
        hyperParams = {'min_impurity_split': [0.0, 1.0], 'min_samples_split': [0.0001, 0.05], 'n_estimators': [10, 150]}
        bestSetOfParams = {'min_impurity_split': 0, 'min_samples_split': 0, 'n_estimators': 0}
    measureToCompare = 'F1' if avgMethod == 'macro' or avgMethod == 'weighted' else 'Accuracy'
    bestMeasure = 0
    for i in range(100):
        currSetOfParams = randomHyperParamsForClassifier(hyperParams)
        if clfType == 'RF':
            print('Starting iteration number:', i)
            estimator = RandomForestClassifier(criterion="entropy", **currSetOfParams)
        if clfType == 'Tree':
            estimator = tree.DecisionTreeClassifier(criterion="entropy",**currSetOfParams)
        metric, _,_ = clfMetricCalculator(estimator, x_train, y_train, avgMethod)
        # print('with this set:\n',currSetOfParams)
        # print('we achived those scores\n',metric)
        if bestMeasure < metric[measureToCompare]:
            bestMeasure = metric[measureToCompare]
            bestSetOfParams = currSetOfParams

    return bestSetOfParams,bestMeasure


def hyperParamsForKNN(x_train,y_train,avgMethod='weighted'):
    """
    Function that find best hyperparams for KNN
    :param x_train:
    :param y_train:
    :param avgMethod: from ['weighted','macro'] , differs from one classification to another
    :return:
    """
    hyperParams = {'n_neighbors':[3,30]}
    bestSetOfParams = {'n_neighbors': 0}
    measureToCompare = 'F1' if avgMethod == 'macro' or avgMethod == 'weighted' else 'Accuracy'
    bestMeasure = 0
    for i in range(3,30,2):
        currSetOfParams = {'n_neighbors': i}
        estimator = KNeighborsClassifier(**currSetOfParams)
        metric, _, _ = clfMetricCalculator(estimator, x_train, y_train,avgMethod)
        # print('with this set:\n',currSetOfParams)
        # print('we achived those scores\n',metric)
        if bestMeasure < metric[measureToCompare]:
            bestMeasure = metric[measureToCompare]
            bestSetOfParams = currSetOfParams

    return bestSetOfParams,bestMeasure


def hyperParamsForSVM(x_train,y_train,avgMethod='weighted'):

    hyperParams = {'C':[-5,10], 'cache_size':[200,201], 'degree':[2,7], 'gamma':[-15,3],'max_iter':[10000,10001]}
    bestSetOfParams = {'C':0, 'cache_size':200, 'degree':3, 'gamma':2,'max_iter':10000}
    kernelParams = ['rbf','linear','poly']
    measureToCompare = 'F1' if avgMethod == 'macro' or avgMethod == 'weighted' else 'Accuracy'
    bestMeasure = 0
    for i in range(50):
        print('Starting iteration number:', i)
        currSetOfParams = randomHyperParamsForClassifier(hyperParams)
        currSetOfParams['C'] = 2 ** currSetOfParams['C']
        currSetOfParams['gamma'] = 2 ** currSetOfParams['gamma']
        currSetOfParams['kernel'] = random.choice(kernelParams)
        print(currSetOfParams)
        estimator = SVC(**currSetOfParams)
        metric, _, _ = clfMetricCalculator(estimator, x_train, y_train, avgMethod)
        # print('with this set:\n',currSetOfParams)
        # print('we achived those scores\n',metric)
        if bestMeasure < metric[measureToCompare]:
            bestMeasure = metric[measureToCompare]
            bestSetOfParams = currSetOfParams

    return bestSetOfParams, bestMeasure




def measuresWithoutKFold(clf,x_train,y_train,x_val,y_val,avgMethod='weighted'):
    """

    :param clf: initialized clf
    :param x_train: for the training
    :param y_train: for the training
    :param x_val: to measure performances
    :param y_val: to measure performance
    :param avgMethod: from ['weighted','macro'] , differs from one classification to another
    :return:
    """
    numOflabels = y_train.iloc[:,0].nunique() # Y should contain only one column - label column
    confusionMatrix = np.zeros((numOflabels, numOflabels))

    partiesLabels = y_train.iloc[:,0].unique()

    # train the clf on X set
    clf.fit(x_train,y_train)
    # test the tree on validation set
    pred = clf.predict(x_val)
    pred = pd.DataFrame(pred, index=y_val.index)  # convert into col vector and match index

    confusionMatrix = confusion_matrix(y_val.values, pred, labels=partiesLabels)

    accuracy = accuracy_score(y_val, pred)
    f1 = f1_score(y_val, pred, average=avgMethod)
    precision = precision_score(y_val, pred, average=avgMethod)
    recall = recall_score(y_val, pred, average=avgMethod)
    metricMap = {'Accuracy': accuracy, 'F1': f1, 'Precision': precision, 'Recall': recall}

    return metricMap,confusionMatrix



def trainWithBestHyperparams(clfType,methodDict,x_train,y_train,x_val,y_val):
    """
    Function that train one clf by his hyperparams and returns his measurements on validation set
    :param clfType: 'Tree' for tree, 'RF' for random forest and 'KNN' for KNN
    :param methodDict: from ['weighted','macro'] , differs from one classification task to another
    :param x_train:
    :param y_train:
    :param x_val
    :param y_val
    :return:
    """
    # f = open('./output','w')
    # line = '\t' + clfType
    # f.write(line)
    print('\t',clfType)
    for method in methodDict:
        if clfType == 'KNN':
            estimator = KNeighborsClassifier(**methodDict[method])
        elif clfType == 'Tree':
            estimator = tree.DecisionTreeClassifier(criterion="entropy", **methodDict[method])
        elif clfType == 'RF':
            estimator = RandomForestClassifier(criterion="entropy", **methodDict[method])
        elif clfType == 'SVM':
            estimator = SVC(**methodDict[method])
        metric,confusionMatrix = measuresWithoutKFold(estimator,x_train,y_train,x_val,y_val,avgMethod=method)
        # line = '\tIn' + method + 'method'
        # f.write(line)
        print('\tIn',method,'method')
        # f.write('\tIn',method,'method')
        # f.write(str(metric))
        print(metric)
        # f.write('achivhed with those hyperparams:')
        print('achieved with those hyperparams:')
        # f.write(str(**methodDict[method]))
        print(methodDict[method])
        # # make nice confusion matrix with labels
        # partiesLabels = y_train.iloc[:, 0].unique()
        # confusionMatrix = pd.DataFrame(confusionMatrix,columns=partiesLabels,index=partiesLabels)
        # f.write(confusionMatrix)
    # f.close()
