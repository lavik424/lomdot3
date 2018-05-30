import numpy as np
from modelTesting import modifiedHistogram

from modelTesting import measuresWithoutKFold


def oneFitAll(clf,x_train,y_train,x_test,y_test):
    """
    Calculate everything by accuracy
    :param clf: clf
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return:
    """
    labels = y_train.iloc[:,0].unique()
    metricMap, confusionMatrix, clf, pred = measuresWithoutKFold(clf,x_train,y_train,x_test,y_test)
    counts = {x: np.sum(confusionMatrix.loc[x, :]) for x in
              labels}  # sum each col separately (predicted num of voters for each party)

    winner = max(counts,key=counts.get)
    print("The party with the most votes is:",winner)
    modifiedHistogram(confusionMatrix,labels)
    print("Every row's prediction:\n",pred)



def diffMethodsSolver(clfForProblems,x_train,y_train,x_test,y_test):
    """
    function the solve the 3 problems by different classifiers
    :param clfForProblems:
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return:
    """
    labels = y_train.iloc[:,0].unique()
    # for problem1 - winning party
    _, confusionMatrix, _, _ = measuresWithoutKFold(
        clfForProblems['Problem1']['clf'],x_train,y_train,x_test,y_test)
    counts = {x: np.sum(confusionMatrix.loc[x, :]) for x in
              labels}  # sum each col separately (predicted num of voters for each party)

    winner = max(counts,key=counts.get)
    print("The party with the most votes is:",winner)

    # for problem2 - distribution
    metricMap, confusionMatrix, clf, pred = measuresWithoutKFold(
        clfForProblems['Problem2']['clf'],x_train,y_train,x_test,y_test)
    modifiedHistogram(confusionMatrix,labels)

    # for problem3 - every row prediction
    metricMap, confusionMatrix, _, pred = measuresWithoutKFold(
        clfForProblems['Problem3']['clf'],x_train,y_train,x_test,y_test)
    print("Every row's prediction:\n",pred)
    print("Confusion Matrix:\n",confusionMatrix)
    print("Problem3 scores:\n",metricMap)
