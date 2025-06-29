#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Includes Machine learning models for Tianjin Cohort
"""
__author__      = "Chengbin Hu" 
__copyright__   = "BGI Copyright 2023"



import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder
#from sklearn.linear_model import LogisticRegression 

    
def forward_regression(X, y,
                       initial_list=[], 
                       threshold_in=0.05, 
                       threshold_out = 1, 
                       verbose=True):
    initial_list = []
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        #print(included)
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded,dtype=float)
        for new_column in excluded:
            #print(new_column)
            #if new_column in newpara:
            #    continue
            #model = LogisticRegression(random_state=0).fit(X[included+[new_column]], y)
            #print(model.pvalues)

            #model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            #X=(X-X.min())/(X.max()-X.min())
            model = sm.MNLogit(y, X[included+[new_column]]).fit()
            #print(new_pval[new_column])
            #print(newpara,new_column)
            #print(model.pvalues)
            new_pval[new_column] = model.pvalues[0][new_column]

        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.index[new_pval.argmin()]
            included.append(best_feature)
            changed=True
            print(best_feature,best_pval)
            #if verbose:
            #    print('Add   with p-value '.format(best_feature, best_pval))

        if not changed:
            break

    return included
data = pd.read_csv("mRNATotal07_group_train.csv",sep=',',header=0,low_memory=False)
data.pop("Row.names")
Y_train = data.pop("group")
X_train=(data-data.min())/(data.max()-data.min())

selected = forward_regression(X_train, Y_train)
with open('output1.txt', 'w') as f:
     print("final result:", selected, file=f)
#print("final result:",selected)
