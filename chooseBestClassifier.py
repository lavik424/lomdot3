
from modelTesting import trainWithBestHyperparams

def chooseBestClfForProblem(hyperParamters,x_train,y_train,x_val,y_val):
    """
    Problem1- winning party by weighted
    Problem2- distribution of voters by macro
    Problem3- each row prediction by accuracy
    :param hyperParamters: for each clf has each best hyperparamters
    :param x_train:
    :param y_train:
    :param x_val:
    :param y_val:
    :return: dict of metric, cm , clf
    """
    clfForProblem = {'Problem1':None,'Problem2':None,'Problem3':None}
    clfTypes = ['NB','Tree', 'SVM', 'KNN', 'RF']
    averageMethodsForMeasures = ['weighted', 'macro', 'Accuracy']
    trainedClf = {type: {method: None for method in averageMethodsForMeasures} for type in clfTypes}

    # trains best hyperparamters on training set and test on validation set
    for method in averageMethodsForMeasures:
        trainedClf['NB'][method] = trainWithBestHyperparams\
                ('NB',{'priors':None},x_train,y_train,x_val,y_val)
        for type in clfTypes:
            if type == 'NB':
                continue
            trainedClf[type][method] = trainWithBestHyperparams\
                (type,hyperParamters[type][method],x_train,y_train,x_val,y_val)

    # print(trainedClf)
    # find best for each problem
    clfForProblem['Problem1'] = trainedClf['Tree']['weighted']
    clfForProblem['Problem2'] = trainedClf['Tree']['macro']
    clfForProblem['Problem3'] = trainedClf['Tree']['Accuracy']
    for type in clfTypes:
        if clfForProblem['Problem1']['metric']['F1'] < trainedClf[type]['weighted']['metric']['F1']:
            clfForProblem['Problem1'] = trainedClf[type]['weighted']
        if clfForProblem['Problem2']['metric']['F1'] < trainedClf[type]['macro']['metric']['F1']:
            clfForProblem['Problem2'] = trainedClf[type]['macro']
        if clfForProblem['Problem3']['metric']['Accuracy'] < trainedClf[type]['Accuracy']['metric']['Accuracy']:
            clfForProblem['Problem3'] = trainedClf[type]['Accuracy']

    return clfForProblem