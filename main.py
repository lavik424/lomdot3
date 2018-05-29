import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import tree

from TreePlot import plotTree
from modelTesting import *
import matplotlib.pyplot as plt


def chooseColumns(currCols,oldCols):
    modeSuffix = 'FillByModeInt'
    onehotSuffix = 'FillByMode_'
    meanSuffix = 'FillByMean'
    meadianSuffix = 'FillByMedian'
    nanSuffix = '_nan'
    # currList= list of original cols name, res= set of cols to continue with
    res = [colName for colName in currCols if onehotSuffix in colName]
    res += [colName for colName in currCols if (meadianSuffix in colName) & (meanSuffix not in colName)]
    currListMode = [colName for colName in oldCols if (colName + modeSuffix) in currCols]
    res += [(colName + modeSuffix) for colName in currListMode]
    currListMean = [colName for colName in oldCols if ((colName + meanSuffix) in currCols) & (colName not in currListMode)]
    res += [(colName + meanSuffix) for colName in currListMean if (colName + meadianSuffix) not in res]
    res = [colName for colName in res if nanSuffix not in colName]
    return res


def chooseRightColumns(colsAfterWork):
    """

    :param colsAfterWork: the names of columns without nan
    :return: list with the relevant features
    """
    rightList = ["Yearly_IncomeKFillByMean" ,"Number_of_valued_Kneset_membersFillByMedian",
                 "Overall_happiness_scoreFillByMean" ,"Garden_sqr_meter_per_person_in_residancy_areaFillByMean",
                 "Most_Important_IssueFillByMode_" ,"Weighted_education_rank",
                 "Will_vote_only_large_partyFillByMode_" ,"Avg_Satisfaction_with_previous_vote"]
    res = []
    for feature in rightList:
        for oldFeat in colsAfterWork:
            if feature in oldFeat:
                res.append(oldFeat)
    return res



def loadEnvirment():
    """

    :return: 6 dataFrames train, validarion and test x/y
    """

    df = pd.read_csv("./ElectionsData.csv")
    oldCols = df.columns

    ## load tables from prev hw
    x_train = pd.read_csv("./input/x_train.csv" ,index_col=0)
    x_val = pd.read_csv("./input/x_val.csv", index_col=0)
    x_test = pd.read_csv("./input/x_test.csv", index_col=0)
    y_train = pd.read_csv("./input/y_train.csv" ,index_col=0)
    y_val = pd.read_csv("./input/y_val.csv", index_col=0)
    y_test = pd.read_csv("./input/y_test.csv", index_col=0)


    # choose the correct set of features
    colsAfterWork = chooseColumns(x_train.columns, oldCols)
    rightFeatures = chooseRightColumns(colsAfterWork)

    x_train = x_train[rightFeatures]
    x_val = x_val[rightFeatures]
    x_test = x_test[rightFeatures]

    return x_train, x_val, x_test, y_train, y_val, y_test


def modifiedHistogram(cm:pd.DataFrame,labels):
    counts = {x:np.sum(cm.loc[x,:]) for x in labels} # sum each col separately (predicted num of voters for each party)

    # plt.plot(np.arange(10))
    # plt.show()
    plt.bar(np.arange(len(labels)), counts.values(), align='center')
    plt.xticks(np.arange(len(labels)), counts.keys())
    # plt.savefig("./sddf.png")
    plt.show()
    print('done')

def saveEnvirment(x_train, x_val, x_test, y_train, y_val, y_test):
    # save after all changes
    x_train_final = x_train.copy()
    x_train_final['Vote'] = y_train.values
    x_val_final = x_val.copy()
    x_val_final['Vote'] = y_val.values
    x_test_final = x_test.copy()
    x_test_final['Vote'] = y_test.values
    # Save labels
    x_train_final.to_csv("./x_train_final.csv")
    x_val_final.to_csv("./x_val_final.csv")
    x_test_final.to_csv("./x_test_final.csv")
    y_train.to_csv("./y_train.csv")
    y_val.to_csv("./y_val.csv")
    y_test.to_csv("./y_test.csv")

def main():

    (x_train, x_val, x_test, y_train, y_val, y_test) = loadEnvirment()
    saveEnvirment(x_train, x_val, x_test, y_train, y_val, y_test)

    # modifiedHistogram(confusionMatrix,partiesLabels)
    import warnings
    warnings.filterwarnings("ignore") # todo fun ignoring that shit

    # clf = GaussianNB(priors=None)
    # metric,cm,clf = clfMetricCalculator(clf,x_train,y_train)
    # print(metric)
    # print(cm)
    #
    # exit(3)

    clfTypes = ['Tree']#,'Tree','KNN','RF']
    hyperParamters = {type:{'weighted':None,'macro':None, 'Accuracy':None} for type in clfTypes}
    averageMethodsForMeasures = ['weighted', 'macro', 'Accuracy']

    # train to find best hyperparameters for classifiers with different avg methods
    for method in averageMethodsForMeasures:
        for type in clfTypes:
            if type in ['Tree', 'RF']:
                hyperParamters[type][method],_ = hyperParamsForTreeOrRF(type,x_train,y_train,method)
            elif type == 'KNN':
                hyperParamters[type][method],_ = hyperParamsForKNN(x_train,y_train,method)
            else:
                hyperParamters[type][method], _ = hyperParamsForSVM(x_train, y_train, method)

    # train the classifers with the best set of hyperparams
    # for type in clfTypes:
    #     for method in averageMethodsForMeasures:
    #         estimator = trainWithBestHyperparams(type,hyperParamters[type][method],x_train,y_train,x_val,y_val)

    estimator = trainWithBestHyperparams("Tree", hyperParamters["Tree"]['Accuracy'], x_train, y_train, x_val, y_val)



    # labelsForTree = ['Blues', 'Browns', 'Greens', 'Greys', 'Oranges', 'Pinks',
    #                  'Purples', 'Reds', 'Turquoises', 'Whites', 'Yellows']
    # plotTree(estimator, x_train.columns, labelsForTree)


if __name__ == '__main__':
    main()

