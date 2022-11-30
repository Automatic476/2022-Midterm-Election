#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import os
get_ipython().run_line_magic('cd', '/Users/jasongreen/Desktop/archive/polls')
os.environ['KAGGLE_CONFIG_DIR'] = "/Users/jasongreen/Desktop/archive/polls"


# In[43]:


# Loading all libraries

#Baysian Optimization
from bayes_opt import BayesianOptimization

#Pandas stack
import pandas as pd
import numpy as np

#FastAI
from fastai.tabular.all import *
from fastai.tabular.core import *

#Plots
import matplotlib.pyplot as plt
import seaborn as sns

#System
import os
import sys
import traceback

#Fit an xgboost model
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score

#Random
import random

#TabNet
from fast_tabnet.core import *

import shutil

#general brute force imports
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
get_ipython().run_line_magic('config', "InlineBackend.figure_format ='retina'")
get_ipython().run_line_magic('matplotlib', 'inline')

from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot

#Fit an xgboost model
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score

import opendatasets as od

#LazyPredict
import lazypredict
from lazypredict.Supervised import LazyClassifier, LazyRegressor

import missingno as msno
get_ipython().run_line_magic('matplotlib', 'inline')

from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

from fbprophet import Prophet


# In[4]:


# Check the versions of libraries
 
# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))

from sklearn.utils import all_estimators


# In[5]:


#Project Variables
PROJECT_NAME = 'elections'
VARIABLE_FILES = False
SAMPLE_COUNT = 20000
FASTAI_LEARNING_RATE = 1e-1
AUTO_ADJUST_LEARNING_RATE = False
ENABLE_BREAKPOINT = True
CONVERT_TO_CAT = False
REGRESSOR = True
SEP_DOLLAR = True
SEP_PERCENT = True
SHUFFLE_DATA = True
Y_COL = 'Total'
DATE_COL = 'Poll Date'


# In[6]:


#redacted code from Data Ranch - https://dataranch.info/ - 
#importing senate polling dataset
param_dir = f'/Users/jasongreen/Desktop/archive/polls'
TARGET = ''
PARAM_DIR = param_dir
print(f'param_dir: {param_dir}')
df = pd.read_csv('senate_polls.csv')
df.rename(columns={0:'poll_id',1:'pollster_id',2:'pollster',3: 'sponsor_ids',4:'sponsors',5: 'display_name',6: 'pollster_rating_id',7: 'pollster_rating_name',8: 'fte_grade,methodology',9: 'state,start_date',10: 'end_date',11: 'sponsor_candidate_id',12: 'sponsor_candidate',13: 'sponsor_candidate_party',14: 'question_id',15: 'sample_size',16: 'population',17: 'subpopulation',18: 'population_full',19: 'tracking',20: 'created_at',21: 'notes',22: 'url', 23: 'source',24: 'internal',25: 'partisan',26: 'race_id,cycle',27: 'office_type',28: 'seat_number', 29: 'seat_name',30: 'election_date',31: 'stage',32: 'nationwide_batch',33: 'ranked_choice_reallocated',34: 'party,answer',35: 'candidate_id',36: 'candidate_name', 37:'pct'}, inplace=True)
df.to_csv('Project backup.csv', index=False) # save to new csv file
df.head()
df.shape


# In[50]:


df.describe()


# In[54]:


len(df)


# In[8]:


df.isna().sum()


# In[9]:


sns.heatmap(df.corr())


# In[10]:


df.shape


# In[11]:


msno.bar(df)


# In[12]:


df = df.drop( ['sponsor_ids','sponsor_candidate_id', 'sponsor_candidate', 'sponsor_candidate_party','subpopulation', 'tracking','notes','source'], axis = 1)


# In[13]:


#getting rid of symbols such as percentage signs or dollar signs
#For every column in in the dataframe where the column contains a % or a $, make a new column with the value without the symbol
if SEP_DOLLAR:
    for col in df.columns:
        if '$' in df[col].to_string():
            df[col + '_no_dollar'] = df[col].str.replace('$', '').str.replace(',', '')


if SEP_PERCENT:
    for col in df.columns:
        if '%' in df[col].to_string():
            df[col + '_no_percent'] = df[col].str.replace('%', '').str.replace(',', '')

target = ''
target_str = ''

targets = []
#loops through the columns 
for i in range(len(df.columns)-1, 0, -1):
    try:
        df[df.columns[i]] = df[df.columns[i]].astype(float)
        target = df.columns[i]
        target_str = target.replace('/', '-')
    except:
        continue
    print(f'Target Variable: {target}')


# In[14]:


#after checking for symbols delete duplicate rows 
df = df.drop_duplicates()
if SHUFFLE_DATA:
    df = df.sample(frac=1).reset_index(drop=True)


# In[15]:


# workaround for fastai/pytorch bug 
for n in df:
    if pd.api.types.is_bool_dtype(df[n]):
        df[n] = df[n].astype('uint8')

with open(f'senate_polls.csv', 'r') as f:
    cols_to_delete = f.read().splitlines()
for col in cols_to_delete:
    try:
        del(df[col])
    except:
        pass
   
    #try to fill in missing values now, otherwise FastAI will fix this farther down 
try:
    df = df.fillna(0)
except:
    pass
df = df_shrink(df)


# In[16]:


#Auto detection of categorical and continuous variables
likely_cat = {}
for var in df.columns:
    likely_cat[var] = 1.*df[var].nunique()/df[var].count() < 0.05 #or some other threshold

cats = [var for var in df.columns if likely_cat[var]]
conts = [var for var in df.columns if not likely_cat[var]]

#remove target from lists
try:
    conts.remove(target)
    cats.remove(target)
except:
    pass
#Convert target to float
df[target] = df[target].astype(float)

print('CATS=====================')
print(cats)
print('CONTS=====================')
print(conts)


# In[17]:


#Populate categorical and continuous lists

if VARIABLE_FILES == True:
    with open(f'/Users/jasongreen/Desktop/archive/polls/cats.txt', 'r') as f:
        cats = f.read().splitlines()

    with open(f'/Users/jasongreen/Desktop/archive/polls/conts.txt', 'r') as f:
        conts = f.read().splitlines()


# In[18]:


procs = [Categorify, FillMissing, Normalize]
#print(df.describe().T)
df = df[0:SAMPLE_COUNT]
splits = RandomSplitter()(range_of(df))

#print((len(cats)) + len(conts))
#dd = 12
#dd1 = float(dd)
#print(dd1)

for var in conts:
    try:
        df[var] = float(df[var])
    except:
        print(f'Could not convert {var} to float.')
        pass
    
#print(dd1)


# In[56]:


#Experimental logic for adding columns one at a time
if ENABLE_BREAKPOINT == True:
   temp_procs = [Categorify, FillMissing]
   print('Looping through continuous variables')
   cont_list = []
   for cont in conts:
       focus_cont = cont
       cont_list.append(cont)
       try:
           to = TabularPandas(df, procs=procs, cat_names=cats, cont_names=cont_list, y_names=target, y_block=RegressionBlock(), splits=splits)
           del(to)
       except:
           print('Error with ', focus_cont)
           cont_list.remove(focus_cont)
           if CONVERT_TO_CAT == True:
               cats.append(focus_cont)
           continue
   for var in cont_list:
       try:
           df[var] = df[var].astype(float)
       except:
           print(f'Could not convert {var} to float.')
           cont_list.remove(var)
           if CONVERT_TO_CAT == True:
               cats.append(var)
           pass
   print('')
   print(f'Continuous variables that made the cut : {cont_list}')
   print('')
   print(f'Categorical variables that made the cut : {cats}')
   df = df_shrink(df)


# In[20]:


#Creating tabular object + quick preprocessing
to = None
if REGRESSOR == True:
    try:
        to = TabularPandas(df, procs, cats, conts, target, y_block=RegressionBlock(), splits=splits)
    except:
        conts = []
        to = TabularPandas(df, procs, cats, conts, target, y_block=RegressionBlock(), splits=splits)
else:
    try:
        to = TabularPandas(df, procs, cats, conts, target, splits=splits)
    except:
        conts = []
        to = TabularPandas(df, procs, cats, conts, target, splits=splits)

dls = to.dataloaders()
print(f'Tabular Object size: {len(to)}')
try:
    dls.one_batch()
except:
    print(f'problem with getting one batch of {PROJECT_NAME}')


# In[21]:


#Extracting train and test sets from tabular object
X_train, y_train = to.train.xs, to.train.ys.values.ravel()
X_test, y_test = to.valid.xs, to.valid.ys.values.ravel()

if target in X_train and target in X_test:
    del(X_train[target])
    del(X_test[target])
#create dataframe from X_train and y_train and export 
pd.DataFrame(X_train).to_csv(f'/Users/jasongreen/Desktop/archive/polls/X_train_{target_str}.csv', index=False)
pd.DataFrame(X_test).to_csv(f'/Users/jasongreen/Desktop/archive/polls/X_test_{target_str}.csv', index=False)
pd.DataFrame(y_train).to_csv(f'/Users/jasongreen/Desktop/archive/polls/y_train_{target_str}.csv', index=False)
pd.DataFrame(y_test).to_csv(f'/Users/jasongreen/Desktop/archive/polls/y_test_{target_str}.csv', index=False)


# In[22]:


msno.bar(df)


# In[55]:


df.shape


# In[23]:


#Model Selection May Finally Begin
if REGRESSOR == True:
    try:
        reg = LazyRegressor(verbose=2, ignore_warnings=False, custom_metric=None)
        models, predictions = reg.fit(X_train, X_test, y_train, y_test)
        print(f'Project: {PROJECT_NAME}')
        print(PROJECT_NAME)
        print(f'Target: {target}')
        print(target)
        target_std = y_train.std()
        print(f'Target Standard Deviation: {target_std}')
        print(models)
        models['project'] = PROJECT_NAME
        models['target'] = target
        models['target_std'] = target_std
        #rename index of 
        models.to_csv(f'{PARAM_DIR}/regression_results_{target_str}.csv', mode='a', header=True, index=True)
    except:
        print('Issue during lazypredict analysis')
else:
    try:
        clf = LazyClassifier(verbose=2, ignore_warnings=False, custom_metric=None)
        models, predictions = clf.fit(X_train, X_test, y_train, y_test)
        print(f'Project: {PROJECT_NAME}')
        print(PROJECT_NAME)
        print(f'Target: {target}')
        print(target)
        print(f'Target Standard Deviation: {y_train.std()}')
        print(models)
        models.to_csv(f'{PARAM_DIR}/classification_results.csv', mode='a', header=False)
    except:
        print('Issue during lazypredict analysis')

model_name = 'tabnet'


# In[24]:


# FastAI + pre-trained TabNet
#==============================================================================
learn = None
i = 0
while True:
   try:
       del learn
   except:
       pass
   try:
       learn = 0
       model = TabNetModel(get_emb_sz(to), len(to.cont_names), dls.c, n_d=64, n_a=64, n_steps=5, virtual_batch_size=256)
       # save the best model so far, determined by early stopping
       cbs = [SaveModelCallback(monitor='_rmse', comp=np.less, fname=f'{model_name}_{PROJECT_NAME}_{target_str}_best'), EarlyStoppingCallback()]
       learn = Learner(dls, model, loss_func=MSELossFlat(), metrics=rmse, cbs=cbs)
       #learn = get_learner(to)
       if(learn != 0):
           break
       if i > 50:
           break
   except:
       i += 1
       print('Error in FastAI TabNet')
       traceback.print_exc()
       continue
   try:
       #display learning rate finder results
       x = learn.lr_find()
   except:
       pass
   if AUTO_ADJUST_LEARNING_RATE == True:
       FASTAI_LEARNING_RATE = x.valley
   print(f'LEARNING RATE: {FASTAI_LEARNING_RATE}')
   try:
       if i < 50:
           learn.fit_one_cycle(20, FASTAI_LEARNING_RATE)
           plt.figure(figsize=(10, 10))
           try:
               ax = learn.show_results()
               plt.show(block=True)
           except:
               print('Could not show results')
               pass
   except:
       print('Could not fit model')
       traceback.print_exc()
       pass


# In[25]:


#==============================================================================
#fit an xgboost model
#==============================================================================
if REGRESSOR == True:
    xgb = XGBRegressor()
else:
    xgb = XGBClassifier()
try:
    xgb = XGBRegressor()
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    print('XGBoost Predictions vs Actual==========')
    print(pd.DataFrame({'actual': y_test, 'predicted': y_pred}).head())
    print('XGBoost RMSE: ', np.sqrt(mean_squared_error(y_test, y_pred)))
    #save feature importance plot to file
    plot_importance(xgb)
    plt.title(f'XGBoost Feature Importance for {PROJECT_NAME} | Target : {target}', wrap=True)
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{PARAM_DIR}/xgb_feature_importance_{target_str}.png')
    fi_df = pd.DataFrame([xgb.get_booster().get_score()]).T
    fi_df.columns = ['importance']
    #create a column based off the index called feature
    fi_df['feature'] = fi_df.index
    #create a dataframe of feature importance
    fi_df = fi_df[['feature', 'importance']]
    fi_df.to_csv(f'{PARAM_DIR}/xgb_feature_importance_{target_str}.csv', index=False)
    #xgb_fi = pd.DataFrame(xgb.feature_importances_, index=X_train.columns, columns=['importance'])
    #xgb_fi.to_csv(f'{PARAM_DIR}/xgb_feature_importance_{target_str}.csv')
    #print('XGBoost AUC: ', roc_auc_score(y_test, y_pred))
except:
    traceback.print_exc()
    print('XGBoost failed')


# In[26]:


df.describe()


# In[27]:


df.isna().sum()


# In[28]:


# Cross Validation and print out a confusion matrix

#Get position recall

#get kappa score

sns.heatmap(df.corr())


# In[29]:


out_dir = f'/Users/jasongreen/Desktop/archive/polls'
xgb_feature_importance_csvs = []

for file in os.listdir(out_dir):
    if 'xgb_feature_importance' in file and '.csv' in file:
        xgb_feature_importance_csvs.append(pd.read_csv(os.path.join(out_dir, file)))

xgb_feature_importance = pd.concat(xgb_feature_importance_csvs,axis=0)
xgb_feature_importance.rename(columns={'Unnamed: 0': 'feature'}, inplace=True)
print(xgb_feature_importance.head())
xgb_feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False).plot(kind='bar', title='XGBoost Overall Feature Importance', figsize=(6, 6))



# In[49]:


#'poll_id','pollster','sponsor_ids','sponsors','display_name','pollster_rating_id','pollster_rating_name','start_date','end_date','sponsor_candidate_id','sponsor_candidate','sponsor_candidate_party','question_id','sample_size','population','subpopulation','population_full','tracking','created_at','notes','url', 'source','internal','partisan','race_id,cycle','office_type','seat_number', 'seat_name','election_date','stage','nationwide_batch','ranked_choice_reallocated','party,answer', 'candidate_id','candidate_name','pct'])
model = Prophet()
ph_df = df
#ph_df = ph_df.drop(['poll_id','pollster','sponsors','display_name','pollster_rating_id','pollster_rating_name'])

ph_df = ph_df.reset_index()
ph_df = ph_df.rename(columns = {'state':'y', 'fte_grade':'ds'})

model.fit(ph_df)

#I've run out of time to finish the forecasting however the feature significance is complete but please use the below link:
#https://www.kaggle.com/code/aks2411/time-series-with-prophet-s-p-500

#This shows how to use facebooks prophet prediction modeling 


# In[45]:


ph_df.head()


# In[41]:


msno.bar(ph_df)


# In[ ]:




