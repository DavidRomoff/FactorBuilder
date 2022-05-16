
# Setup
import numpy as np
import pandas as pd
import math

# Create Data
n = 1000
marketCap = np.random.normal(100,10,n)
bookValue = np.random.normal(80,10,n)
returns = np.random.normal(.01,.005,n)
df = pd.DataFrame(data=zip(returns,marketCap,bookValue),columns=['returns','marketCap','bookValue'],index=range(1,n+1))

def CategoryBuilder(X: pd.DataFrame,category_colname: str,cutoffs: str,labels: str,
                    categoryType = 'percentile',categoryTitle = 'categories') -> None: 
    if categoryType == 'value':
        X[categoryTitle] = 'unlabeled'
        lastcutoff = math.inf
        for cutoff,label in zip(cutoffs,labels):
            ind_category = np.logical_and(X[category_colname] > cutoff, X[category_colname] <= lastcutoff)
            X.loc[ind_category,categoryTitle] = label
            lastcutoff = cutoff

def FactorBuilder(X: pd.DataFrame,returns_colname: str,risk_colname: str,category_colname: str):
    ind_highLow = np.logical_or(X[risk_colname]=='high',X[risk_colname]=='low')
    X_aggregated = X.loc[ind_highLow,:].groupby([category_colname,risk_colname]).mean()
    categoryDiffMeans = X_aggregated.loc[:,'high',:][returns_colname] - X_aggregated.loc[:,'low',:][returns_colname]
    factor = categoryDiffMeans.mean()
    return factor

CategoryBuilder(X=df,
                category_colname='marketCap',
                cutoffs=[100,90,0],
                labels=['low','off','high'],
                categoryType='value',
                categoryTitle='riskHighLow')

CategoryBuilder(X=df,
                category_colname='bookValue',
                cutoffs=[105,95,0],
                labels=['bigcap','midcap','smallcap'],
                categoryType='value',
                categoryTitle='marketCapCategory')

factor = FactorBuilder( X=df,
                        returns_colname='returns',
                        risk_colname='riskHighLow',
                        category_colname='marketCapCategory')

print(factor)
