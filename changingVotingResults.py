import pandas as pd
from ThreeProblemsSolver import oneFitAll

def changeAndItsPrice(clf,x_train,y_train,x_test:pd.DataFrame,y_test):

    # # changing factor "Number_of_valued_Kneset_members" - div by 2
    x_test.loc[:,'Number_of_valued_Kneset_membersFillByMedian'] /= 2
    # x_test.loc[:,'Garden_sqr_meter_per_person_in_residancy_areaFillByMean'] = 0
    # x_test.loc[:,'Will_vote_only_large_partyFillByMode_No'] = 0
    # x_test.loc[:,'Weighted_education_rankFillByMean'] /= 2
    # x_test.loc[:, 'Overall_happiness_scoreFillByMean'] /= 2

    oneFitAll(clf,x_train,y_train,x_test,y_test)

