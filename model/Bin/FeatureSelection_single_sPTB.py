#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Includes Machine learning models for PTB
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
            try:
                model = sm.Logit(y, X[included+[new_column]]).fit()
            except:
                continue
            #print(new_pval[new_column])
            #print(newpara,new_column)
            #print(model.pvalues)
            new_pval[new_column] = model.pvalues[new_column]

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
data = pd.read_csv("train_group_TPM.csv",sep=',',header=0,low_memory=False)
Y_train = data.pop("group")

X_train=(data-data.min())/(data.max()-data.min())

selected = forward_regression(X_train, Y_train)
print("final result:",selected)

def backward_regression(X, y,
                           initial_list=[], 
                           threshold_in=0.01, 
                           threshold_out = 0.05, 
                           verbose=True):
    included=list(X.columns)
    while True:
        changed=False
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop  with p-value '.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

#backward_regression(X_train, Y_train)