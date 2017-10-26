# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 13:58:55 2017

@author: Winson.Liao
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.feature_selection import SelectFromModel
from sklearn.externals import joblib
import warnings
warnings.filterwarnings('ignore')

def plot_corr(dataframe):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    corr = dataframe.corr()
    sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
    return corr
    
def FeatureImportancePlot(X, Xtest, y, k, threshold, featureSelection, plot):
    forest = RandomForestRegressor(n_estimators=50)
    forest.fit(X, y.reshape((-1,)))
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]
    
    
    if plot:
        print("Feature ranking:")
    
        for f in range(min(k,len(importances))):
            print("%d. feature %s (%f)" % (f + 1, X.columns[indices[f]], importances[indices[f]]))
        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(X.shape[1]), importances[indices],
               color="r", yerr=std[indices], align="center")
        plt.xticks(range(X.shape[1]), indices)
        plt.xlim([-1, X.shape[1]])
        plt.savefig('feature importance')
    
    # feature selection
    if featureSelection:
        model = SelectFromModel(forest, prefit=True, threshold=threshold)
        X = model.transform(X)
        Xtest = model.transform(Xtest)
    return X, Xtest
    
    
    
def StandardizeData(XscaleSwitch, YscaleSwitch, X, Xtest, y):
    scaler = None
    scalery = None
    if XscaleSwitch:
        scaler = preprocessing.StandardScaler().fit(X)
        X = scaler.transform(X)  
        Xtest = scaler.transform(Xtest)  
    
    if YscaleSwitch:
        y = y.values.reshape(-1,1)
        scalery = preprocessing.StandardScaler().fit(y)
        y = scalery.transform(y)
        
    return scaler, scalery, X, Xtest, y

def Loaddata(train, testX, testy):
    data = pd.read_csv(train, header=0)
    X = data[sorted(list(set(list(data.columns.values))-set(['y'])))]
    Xtest = pd.read_csv(testX, header=0)[sorted(list(set(list(data.columns.values))-set(['y'])))]
    ytest = pd.read_csv(testy, header=0)['y']
    y = data['y']  
    return X, Xtest, y, ytest

def EncodeCatFeature(X, Xtest):
    Xsize = X.shape[0]
    Xtestsize = Xtest.shape[0]
    df = pd.concat([X, Xtest], keys=X.columns, ignore_index=True)
    Cat_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            Cat_columns.append(col)
    
    df_cat = df[Cat_columns]
    for col in Cat_columns:
        del X[col]
        del Xtest[col]
    df_cat_onehot = pd.get_dummies(df_cat)
    
    X = pd.concat([X, df_cat_onehot.loc[0:Xsize-1]], axis = 1, join_axes = [X.index])
    
    Xtest_cat_onehot = df_cat_onehot.loc[Xsize:]
    Xtest_cat_onehot.index = pd.RangeIndex(Xtestsize)
    Xtest = pd.concat([Xtest, Xtest_cat_onehot], axis = 1, join_axes = [Xtest.index])
    
    return X, Xtest

         
if __name__ == '__main__':
    #__spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    plot = False
    
    XscaleSwitchs = [True, True, False, False, True, True, False, False, True, True, False, False, True, True, False, False]
    YscaleSwitchs = [True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False]
    versions = range(1,17)
    retrains = [False, False, False, False, True, True, True, True, False, False, False, False, True, True, True, True]
    featureSelections = [True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False]
    results = [[] for i in range(16)]
    for i in range(16):
        XscaleSwitch = XscaleSwitchs[i]
        YscaleSwitch = YscaleSwitchs[i]
        version = versions[i]
        retrain = retrains[i]
        featureSelection = featureSelections[i]
        
        print('Xscale:' + str(XscaleSwitch))
        print('Yscale:' + str(YscaleSwitch))
        print('retrain:' + str(retrain))
        print('featureSelection:' + str(featureSelection))
        
        print('load data')
        X, Xtest, y, ytest = Loaddata('C:/Users/Winson.Liao/Downloads/trainingData.csv', 
                 'C:/Users/Winson.Liao/Downloads/testData.csv',
                 'C:/Users/Winson.Liao/Downloads/MASTER.csv')
        
        
        print('Encode categorical data')
        X, Xtest = EncodeCatFeature(X, Xtest)
        
        dataframe = pd.concat([X, y], axis = 1, join_axes = [X.index])
        #corr = plot_corr(dataframe)
        X, Xtest = FeatureImportancePlot(X, Xtest, y, 20, 0.001, featureSelection, plot)
        
        scalerX, scalery, X, Xtest, y = StandardizeData(XscaleSwitch, YscaleSwitch, X, Xtest, y)
    
        
        print('start train test')
        
        para_grid1 = {
                'max_features': ['auto'],
                'max_depth': [50, 200, 300] 
                }                    
        forest = RandomForestRegressor(n_estimators=500)
        
        forestcv = GridSearchCV(forest, para_grid1, cv = 5, n_jobs = 4)
        forestcv.fit(X, y.reshape((-1,)))
 
        forestbest = RandomForestRegressor(**forestcv.best_params_)
        forestbest.fit(X,y.reshape((-1,)))
        if retrain:
            ypred1 = forestbest.predict(Xtest)
        else:
            ypred1 = forestcv.best_estimator_.predict(Xtest)
        ypred1 = ypred1.reshape((-1, 1))
        if YscaleSwitch:
            ypred1 = scalery.inverse_transform(ypred1)
        rmse1 = sqrt(mean_squared_error(ytest, ypred1))
        print('randomforest:' + str(rmse1))
        results[i].append(rmse1)
        joblib.dump(forestbest, 'forestcv' + str(version) +'.pkl') 
        
        
        lm = LinearRegression()
        lm.fit(X,y.reshape((-1,)))
        ypred2 = lm.predict(Xtest)
        ypred2 = ypred2.reshape((-1,1))
        if YscaleSwitch:
            ypred2 = scalery.inverse_transform(ypred2)
        rmse2 = sqrt(mean_squared_error(ytest, ypred2))
        results[i].append(rmse2)
        print('lm:' + str(rmse2))
        joblib.dump(lm, 'lm' + str(version) +'.pkl') 
        
          
        alphalist = np.logspace(-3, 4, num=20)
        ridgereg = Ridge()
        ridgecv = GridSearchCV(ridgereg, {'alpha':alphalist}, cv = 5, n_jobs = 4)
        ridgecv.fit(X,y)
        ridgebest = Ridge(**ridgecv.best_params_)
        ridgebest.fit(X,y)
        if retrain:
            ypred3 = ridgebest.predict(Xtest)
        else:
            ypred3 = ridgecv.best_estimator_.predict(Xtest)
        
        ypred3 = ypred3.reshape((-1,1))
        if YscaleSwitch:
            ypred3 = scalery.inverse_transform(ypred3)
        rmse3 = sqrt(mean_squared_error(ytest, ypred3))
        print('ridge:' + str(rmse3))
        results[i].append(rmse3)
        joblib.dump(ridgecv, 'ridgecv' + str(version) +'.pkl') 
        
        
        para_grid = {
                'activation':['logistic', 'relu'],
                'alpha': np.logspace(-3, 4, num=10),
                'hidden_layer_sizes' : [(100,50,), (100, 10,), (50,10,), (20, 5,)]
                }
        nn = MLPRegressor(learning_rate = 'adaptive', learning_rate_init  = 0.1, max_iter  = 1000)
        nncv = GridSearchCV(nn, para_grid, cv = 5, n_jobs = 4)
        nncv.fit(X, y.reshape((-1,)))
        nnbest = MLPRegressor(learning_rate = 'adaptive', learning_rate_init  = 0.1, max_iter  = 1000, **nncv.best_params_)
        nnbest.fit(X, y.reshape((-1,)))
        
        if retrain:
            ypred4 = nnbest.predict(Xtest)
        else:
            ypred4 = nncv.best_estimator_.predict(Xtest)
            
        ypred4 = nnbest.predict(Xtest)
        ypred4 = ypred4.reshape((-1,1))
        if YscaleSwitch:
            ypred4 = scalery.inverse_transform(ypred4)
        rmse4 = sqrt(mean_squared_error(ytest, ypred4))
        print('nn:' + str(rmse4))
        results[i].append(rmse4)
        joblib.dump(nncv, 'nncv' + str(version) +'.pkl') 
        
        svr = SVR()
        svrcv = GridSearchCV(svr, param_grid={"C": np.logspace(-3, 4, num=10),
                                   "gamma": np.logspace(-2, 2, 5)}, cv = 5, n_jobs = 4)
        svrcv.fit(X, y.reshape((-1,)))
        svrbest = SVR(**svrcv.best_params_)
        svrbest.fit(X, y.reshape((-1,)))
        
        if retrain:
            ypred5 = svrbest.predict(Xtest)
        else:
            ypred5 = svrcv.best_estimator_.predict(Xtest)
        
        ypred5 = ypred5.reshape((-1,1))
        if YscaleSwitch:
            ypred5 = scalery.inverse_transform(ypred5)
        rmse5 = sqrt(mean_squared_error(ytest, ypred5))
        print('svm:' + str(rmse5))
        results[i].append(rmse5)
        joblib.dump(svrcv, 'svrcv' + str(version) +'.pkl') 
        
        ypred_ensemble = np.mean([ypred1, ypred2, ypred3, ypred4, ypred5],axis=0)
        rmse_ensemble = sqrt(mean_squared_error(ytest, ypred_ensemble))
        print('ensemble:' + str(rmse_ensemble))
        results[i].append(rmse_ensemble)
    print ('All results')
    print (results)
        
        
    
        
