#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 18:18:22 2020

@author: alexandraxue
"""

import os 
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc 
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
from scipy.stats.mstats import winsorize
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.tree import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb





ratings = pd.read_csv("~/Documents/PwC/Rating Analysis/ratings.csv")

# In[4]:


def to_rank(ss):
    ss = np.argsort(np.argsort(ss))
    ss = ss/np.nanmax(ss)
    return ss


def to_tier(ss,k=10):
    return ss
    digit_values = ss.dropna().values
    me = np.nanmedian(digit_values)    
    tiers = []
    for i in range(0,k-1):
        t = 100/k*(i+1)
        x = np.percentile(digit_values,t)
        tiers.append(x)
    me = (k-1)/2
    def func(x):
        if np.isnan(x):
            return 0
        else:
            for i in range(len(tiers)):
                if x< tiers[i]:
                    return i-me
            return len(tiers)-me

    res = ss.apply(func)
    return res


# In[70]:


def nandivide(a,b):
    b = b.apply(lambda x:np.nan if x==0 else x)
    r = a/b 
    return r.fillna(0)


# In[94]:


results = pd.DataFrame()
for i in [2012,2013,2014,2015,2016,2017]:
    df = pd.read_csv("~/Documents/PwC/Rating Analysis/report_{}1231.csv".format(i), index_col=0)
    df['Year'] = i
    df.fillna(0, inplace=True) #fill na with 0
    results = pd.concat([results,df],axis=0)
results['Year'].describe()
print(results)


# In[95]:



# In[96]:


# 1 Net Receivables
results['Net Receivables'] = (results['NOTES_RCV']
                                +results['ACCT_RCV']
                                +results['OTH_RCV']
                                +results['DVD_RCV']
                                +results['INT_RCV']
                                -results['NOTES_PAYABLE']
                                -results['ACCT_PAYABLE']
                                -results['ADV_FROM_CUST'])


# In[97]:


# 2 Total Long Term Debt
results['Total Long-term Debt'] = (results['LT_BORROW']
                                +results['BONDS_PAYABLE']
                                +results['LT_PAYABLE'])


# In[98]:


# 3 PP&E
## Why do we need PP&E2?
results['PP&E'] = results['FIX_ASSETS']
results['PP&E2'] = (results['FIX_ASSETS']
                    +results['CONST_IN_PROG']
                    +results['INVEST_REAL_ESTATE']
                    +results['PROJ_MATL']
                    +results['OIL_AND_NATURAL_GAS_ASSETS']
                    +results['PRODUCTIVE_BIO_ASSETS'])


# In[99]:


# 4 Securities
results['Securities'] = (results['FIN_ASSETS_AVAIL_FOR_SALE']+results['TRADABLE_FIN_ASSETS'])


# In[100]:


# 5 SG&A Expense
results['SG&A Expense'] = (results['LESS_SELLING_DIST_EXP']+results['LESS_GERL_ADMIN_EXP'])


# In[101]:


# Others
results['DS'] = nandivide(results['Net Receivables'],results['TOT_OPER_REV'])
results['GM'] = nandivide((results['TOT_OPER_REV']-results['LESS_OPER_COST']),results['TOT_OPER_REV'])
results['AQ'] = nandivide(results['TOT_ASSETS']-(results['TOT_CUR_ASSETS']+results['PP&E']+results['Securities']),results['TOT_ASSETS'])
results['DE'] = nandivide(results['DEPR_FA_COGA_DPBA'],(results['PP&E']+results['DEPR_FA_COGA_DPBA']))
results['SG'] = nandivide(results['SG&A Expense'],results['TOT_OPER_REV'])
results['LV'] = nandivide((results['TOT_CUR_LIAB']+results['Total Long-term Debt']),results['TOT_ASSETS'])
results['WC'] = results['TOT_ASSETS']-results['TOT_LIAB']
results['CFI'] = to_tier(nandivide(results['NET_CASH_FLOWS_INV_ACT'],results['NET_CASH_FLOWS_OPER_ACT'].apply(abs)))
results['CFF'] = to_tier(nandivide(results['NET_CASH_FLOWS_FNC_ACT'],results['NET_CASH_FLOWS_OPER_ACT'].apply(abs)))
results['LOSS'] = (results['TOT_PROFIT']-results['PLUS_NON_OPER_REV']+results['LESS_NON_OPER_EXP']).apply(lambda x: 1 if x<0 else 0)
results['OTHREC'] = to_tier(nandivide(results['OTH_RCV'],results['TOT_ASSETS']))
results['SIZE'] = to_tier(results['TOT_ASSETS'].apply(float))
results['EQUITY'] = results['TOT_ASSETS']-results['TOT_LIAB']
results['XueIdx_1'] = to_tier(nandivide((results['OPER_PROFIT']+results['LESS_FIN_EXP']+results['DEPR_FA_COGA_DPBA']+results['AMORT_INTANG_ASSETS']),results['TOT_LIAB']))
results['XueIdx_3'] = to_tier(nandivide((results['TOT_PROFIT']- results['OPER_PROFIT']),results['TOT_PROFIT'].apply(lambda x: np.nan if x<=0 else x)))
results['XueIdx_4'] = to_tier(nandivide((results['ACCT_RCV']+results['NOTES_RCV']+results['OTH_RCV']),results['TOT_CUR_ASSETS']))    
results['TATA'] = to_tier(nandivide((results['OPER_PROFIT']-results['NET_CASH_FLOWS_OPER_ACT']),results['TOT_ASSETS']))
results


# In[102]:


#NEW

results['CUR_RATIO'] = nandivide(results['TOT_CUR_ASSETS'],results['TOT_CUR_LIAB'])
#quick ratio = (current assets - >inventory<) / current liabilities
#shareholders equity = >capital< + >retained earnings<
results['LT_LIAB'] = results['TOT_LIAB'] - results['TOT_CUR_LIAB']
results['EBIT'] = results['TOT_PROFIT'] - (results['LESS_GERL_ADMIN_EXP'] + results['LESS_SELLING_DIST_EXP'] + results['DEPR_FA_COGA_DPBA'])
#return on capital employed = ebit / (shareholders equity + long term liab)


# In[103]:


for year in [2012,2013,2014,2015,2016,2017]:
    idx1 = results['Year'] == year
    idx2 = results['Year'] == year - 1

    results.loc[idx1,'DSRI'] = to_tier(nandivide(results.loc[idx1,'DS'],results.loc[idx2,'DS']))
    results.loc[idx1,'GMI'] = to_tier(nandivide(results.loc[idx1,'GM'],results.loc[idx2,'GM']))
    results.loc[idx1,'AQI'] = to_tier(nandivide(results.loc[idx1,'AQ'],results.loc[idx2,'AQ']))
    results.loc[idx1,'SGI'] = to_tier(nandivide(results.loc[idx1,'TOT_OPER_REV']-results.loc[idx2,'TOT_OPER_REV'],results.loc[idx2,'TOT_OPER_REV']))
    results.loc[idx1,'DEPI'] = to_tier(nandivide(results.loc[idx2,'DE'],results.loc[idx1,'DE']))
    results.loc[idx1,'SGAI'] = to_tier(nandivide(results.loc[idx1,'SG'],results.loc[idx2,'SG']))
    results.loc[idx1,'LVGI'] = to_tier(nandivide(results.loc[idx1,'LV'],results.loc[idx2,'LV']))
    


# In[104]:


for year in [2012,2013,2014,2015,2016,2017]:
    idx1 = results['Year'] == year
    idx2 = results['Year'] == year - 1
    
    results.loc[idx1,'AC_CHG'] = (results.loc[idx1,'ACCT_RCV']+results.loc[idx1,'NOTES_RCV']) - (results.loc[idx2,'ACCT_RCV']+results.loc[idx2,'NOTES_RCV'])
    results.loc[idx1,'CASH'] = results.loc[idx1,'MONETARY_CAP']-results.loc[idx2,'MONETARY_CAP']
    results.loc[idx1,'XueIdx_2'] = to_tier(nandivide(results.loc[idx1,'CASH'],
                    results.loc[idx1,'NON_CUR_LIAB_DUE_WITHIN_1Y']+
                    results.loc[idx1,'ST_BORROW']+
                    results.loc[idx1,'OTH_PAYABLE']+
                    results.loc[idx1,'ACCT_PAYABLE']+
                    results.loc[idx1,'NOTES_PAYABLE']))
    results.loc[idx1,'CH_CS'] = to_tier(nandivide(results.loc[idx1,'OPER_PROFIT']-results.loc[idx1,'AC_CHG'],results.loc[idx1,'TOT_OPER_REV']))                
    results.loc[idx1,'FCF'] = to_tier(nandivide(results.loc[idx1,'free_cash_flow'] - results.loc[idx2,'free_cash_flow'],
        results.loc[idx2,'free_cash_flow'].apply(abs)))
    results.loc[idx1,'lin3'] = to_tier(nandivide(results.loc[idx1,'STOT_CASH_INFLOWS_INV_ACT'],results.loc[idx2,'STOT_CASH_INFLOWS_INV_ACT']))
    results.loc[idx1,'lin4'] = to_tier(nandivide(results.loc[idx1,'OPER_PROFIT'],results.loc[idx2,'OPER_PROFIT']))
#NEW
    results.loc[idx1, 'CFO'] = (results.loc[idx1, 'EBIT']+results.loc[idx1, 'AMORT_INTANG_ASSETS']
                                +results.loc[idx1, 'DEPR_FA_COGA_DPBA']
                                +results.loc[idx1, 'WC'] - results.loc[idx2, 'WC'])


# In[105]:


results.fillna(0)
results.to_csv('results.csv')




year = []
year = ratings.ANN_DT.astype(str).str[:4].astype(int)
ratings['Year'] = year
print(ratings)

goodratings = ['AAA','Aa1','Aa2','AAA-','AA+','AA']
ratings = ratings.dropna(how='any')

ratings['B_INFO_CREDITRATING2'] = ratings['B_INFO_CREDITRATING'].apply(lambda x:
    1 if x in goodratings else 0)

ratings['B_INFO_PRECREDITRATING2'] = ratings['B_INFO_PRECREDITRATING'].apply(lambda x:
    1 if x in goodratings else 0)



ratings['label'] = np.where(ratings['B_INFO_CREDITRATING2'] < ratings['B_INFO_PRECREDITRATING2'], 1 , 0)


ratings_downgrade = ratings.loc[ratings['label'] == 1]

print(ratings['label'].value_counts())
print(ratings_downgrade)


def lastrating(company,date):
    x = ratings.loc[ratings['S_INFO_COMPCODE'] == company]
    if len(x) > 0:
        x = x.loc[x['ANN_DT'].apply(lambda a: True if a < date else False)]
        if len(x) > 0:
            x = x.sort_values(by='ANN_DT')
            return x.iloc[-1]['B_INFO_CREDITRATING'] in goodratings
        else:
            return False
    else:
        return False
    
    

# x = pull out the company data if it's been downgraded
# when there exists some data entries
def labeling(company,date):
    x = ratings_downgrade.loc[
        ratings_downgrade['S_INFO_COMPCODE'] == company] 
    if len(x)>0:
        compare = x.loc[x['ANN_DT'].apply(lambda a: True if date+10000 > a >= date else False)]
        if len(compare) > 0:
            return 1
        else:
            return 0
    else:
        return 0





#results_all = results.loc[results.apply(lambda row: lastrating(row['S_INFO_COMPCODE'],20171231), axis=1)]
#print(results_all)
#label = pd.DataFrame()

results_all = results.loc[results.apply(lambda row: lastrating(row['S_INFO_COMPCODE'],row['Year']*10000+1231), axis=1)]
print(results_all)

results_new = results_all

results_new['Label']  = results_all.apply(lambda row: labeling(row['S_INFO_COMPCODE'],row['Year']*10000+1231), axis=1)
print(results_new)

data = results_new
print(data)



data['lin1'] =to_tier(nandivide(data['OTHER_CASH_RECP_RAL_FNC_ACT'],data['TOT_ASSETS']))
data['lin2'] =to_tier(nandivide(data['TOT_CUR_LIAB'],data['NET_CASH_FLOWS_OPER_ACT'].apply(abs)))
data['lin_1'] = to_tier(nandivide(data['TOT_OPER_REV']-data['NET_CASH_FLOWS_OPER_ACT'],data['TOT_OPER_REV']))
data['Turnover'] = to_tier(nandivide(data['TOT_OPER_REV'],data['INVENTORIES']))

data = data.drop(['LESS_FIN_EXP', 'LESS_GERL_ADMIN_EXP', 'LESS_NON_OPER_EXP', 'LESS_OPER_COST', 'LESS_SELLING_DIST_EXP',
                  'OPER_PROFIT', 'PLUS_NON_OPER_REV', 'TOT_OPER_REV', 'TOT_PROFIT', 'ACCT_PAYABLE',
                  'ACCT_RCV', 'ADV_FROM_CUST', 'BONDS_PAYABLE', 'CONST_IN_PROG', 'DVD_RCV', 
                  'FIN_ASSETS_AVAIL_FOR_SALE', 'FIX_ASSETS', 'INT_RCV', 'INVEST_REAL_ESTATE',
                  'LT_BORROW','LT_PAYABLE','MONETARY_CAP', 'NON_CUR_LIAB_DUE_WITHIN_1Y', 'NOTES_PAYABLE',
                  'NOTES_RCV', 'OIL_AND_NATURAL_GAS_ASSETS','OTH_PAYABLE', 'OTH_RCV', 'PRODUCTIVE_BIO_ASSETS',
                  'PROJ_MATL', 'ST_BORROW', 'TOT_ASSETS', 'TOT_CUR_ASSETS', 'TOT_CUR_LIAB',
                  'TOT_LIAB', 'TRADABLE_FIN_ASSETS', 'AMORT_INTANG_ASSETS', 'DEPR_FA_COGA_DPBA',
                  'NET_CASH_FLOWS_FNC_ACT', 'NET_CASH_FLOWS_INV_ACT', 'NET_CASH_FLOWS_OPER_ACT'], axis=1)
print(data)
print(data['Label'].value_counts())
data = data.dropna(how='any')

pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 30)   

def confusion_matrix(pred, real):
    x = pd.DataFrame()
    x['label'] = real
    x['predict'] = pred
    
    TP = sum(x[x['predict']==1]['label'])
    FP = sum(1-x[x['predict']==1]['label'])
    TN = sum(1-x[x['predict']==0]['label'])
    FN = sum(x[x['predict']==0]['label'])
    
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    
    print('Precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    confusion_muatrix_result = pd.DataFrame({
        'Negative':{'True':FN, 'False':TN},
        'Positive':{'True':TP, 'False':FP}})
    print(confusion_muatrix_result)
    return confusion_muatrix_result



def illus_roc_curve(real,pred):
    # Compute ROC curve and ROC area for each class 
    fpr,tpr,threshold = roc_curve(real, pred) 
    roc_auc = auc(fpr,tpr) 
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) 
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()
    
 
#LOGISTIC REGRESSION    
reg_cols = [
        #'PP&E',
        'SGI',
        #'Securities',
        'DEPI',
        'DSRI',
        'LVGI',
        #'CUR_RATIO',
        'TATA',
        'XueIdx_1',
        'XueIdx_4',
        'LOSS',
        'CH_CS',
        'SIZE',
        'lin_1',
        'lin1',
       'lin2',
        'lin3',
        'Turnover',
        'AQI',
    'SGAI',
    'CFO',
        ]

logit = sm.Logit(data['Label'], data[reg_cols])
result = logit.fit ()    
print(result.summary())
pred = result.predict(data[reg_cols])
real = data['Label']
illus_roc_curve(real,pred)


pred = result.predict(data[reg_cols])
for alpha in [1e-5,1e-4,1e-3,1e-2]:
    print(alpha)
    pred2 = pred.apply(lambda x: 1 if x>alpha else 0)
    real = data['Label']
    confusion_matrix(pred2,real)

corr = data[reg_cols].corr()
info = data[reg_cols].describe()
print(data[reg_cols].describe())



fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(data[reg_cols].columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(data[reg_cols].columns)
ax.set_yticklabels(data[reg_cols].columns)
plt.show()


#WINSORIZATION
data['SGI'] = stats.mstats.winsorize(data['SGI'], limits=[0.01, 0.1])
data['CH_CS'] = stats.mstats.winsorize(data['CH_CS'], limits=0.01)
data['lin3'] = stats.mstats.winsorize(data['lin3'], limits=[0.1, 0.2])
data['SGAI'] = stats.mstats.winsorize(data['SGAI'], limits=0.01)
data['Turnover'] = stats.mstats.winsorize(data['Turnover'], limits=[0.01, 0.1])


y = data['Label'].values
X = data[reg_cols].values
n = int(len(X)*0.75)
X_train = X[:n]
y_train = y[:n]
X_test = X[n:]
y_test = y[n:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


#DECISION TREE
model = DecisionTreeClassifier(max_depth=20)
r = model.fit(X_train,y_train)
y_pred_train= model.predict(X_train)
y_pred_test = model.predict(X_test)

confusion_matrix(y_pred_train, y_train)
confusion_matrix(y_pred_test, y_test)

illus_roc_curve(y_pred_train, y_train)
illus_roc_curve(y_pred_test,y_test)

#RANDOM FOREST
model = RandomForestClassifier()
model.fit(X_train,y_train)
y_pred_train= model.predict(X_train)
y_pred_test = model.predict(X_test)
confusion_matrix(y_pred_train, y_train)
confusion_matrix(y_pred_test, y_test)

illus_roc_curve(y_pred_train, y_train)
illus_roc_curve(y_pred_test,y_test)




data_lgb = data
data_lgb = data_lgb.dropna(how='any')


lgb_label = data_lgb['Label']
data_lgb.drop(labels=['Label'], axis=1,inplace = True)
data_lgb['Label'] = lgb_label
lgb_year = data_lgb['Year']
data_lgb.drop(labels=['Year'], axis=1,inplace = True)
data_lgb['Year'] = lgb_label
print(data_lgb)
print(data_lgb['Label'].value_counts())



lgb_reg_cols = ['INVENTORIES', 'free_cash_flow', 'Net Receivables', 'Total Long-term Debt', 'PP&E', 'PP&E2', 
                'Securities', 'SG&A Expense', 'DS', 'GM', 'AQ', 'DE', 'SG', 'LV', 'WC', 'CFI', 'CFF', 'LOSS', 
                'OTHREC', 'SIZE', 'EQUITY', 'XueIdx_1', 'XueIdx_3', 'XueIdx_4', 'TATA', 'CUR_RATIO', 'LT_LIAB', 
                'EBIT', 'DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'LVGI', 'AC_CHG', 'CASH', 'XueIdx_2', 'CH_CS', 
                'FCF', 'lin3', 'lin4', 'CFO', 'lin1', 'lin2', 'lin_1', 'Turnover']



dt_y = data_lgb['Label']
dt_X = data_lgb[lgb_reg_cols]
dt_X_train, dt_X_test, dt_y_train, dt_y_test = train_test_split(dt_X, dt_y, test_size = 0.25, random_state = 0)
lgb_train = lgb.Dataset(dt_X_train, dt_y_train, feature_name = lgb_reg_cols)
lgb_dev = lgb.Dataset(dt_X_test,dt_y_test, reference = lgb_train)
params = {
    'task':'train',
    'boosting_type':'gbdt',
    'metric': {'l2','fair'},
    'num_leaves':20,
    'num_threads':8,
    'learning_rate':0.02,
    'feature_fraction':0.3,
    'bagging_fraction':0.8
}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=lgb_dev,
               early_stopping_rounds=10)

dt_x = dt_X.loc[:,lgb_reg_cols]
dt_y_pred = gbm.predict(dt_x, num_iteration = gbm.best_iteration)
gbm.save_model('lgb_model.txt')

fi = pd.Series(gbm.feature_importance(), index = gbm.feature_name())
fi = fi.sort_values(ascending=False)
fi


gbm_real = data_lgb['Label']
illus_roc_curve(gbm_real,dt_y_pred)


for alpha in [0.046,0.015,0.0125,0.01,0.005, 0.0025]:
    print( '--------- {} ---------'.format(alpha))
    dt_pred = [1 if x>alpha else 0 for x in dt_y_pred]
    real = dt_y
    confusion_matrix(dt_pred,real)


