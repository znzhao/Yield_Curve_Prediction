import os
import numpy as np
import datetime
import copy
import scipy
import pandas as pd
from helper.lsctsplitter import loadAllYC, LSCTsplitter
from helper.utils import dropCommonNan, maturities
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from predictors.predclass import BaselinePredClass, LagLSCTPredClass, LinearLSCTPredClass, LinearPredClass

def createYieldPredictor(model):
    registerlist = {'Baseline': BaselinePredClass, 
                    'Lagged LSCT': LagLSCTPredClass, 
                    'Linear LSCT': LinearLSCTPredClass, 
                    'Linear Model': LinearPredClass}
    if model not in registerlist.keys():
        raise(NotImplementedError)
    class YieldPredictor(registerlist[model]):
        pass
    return YieldPredictor

class ModelAssessor(object):
    def __init__(self, names = ['Baseline'], models = ['Baseline'], Xs = [None], starts = [None], ends = [None]):
        if not len(names) == len(models) == len(Xs) == len(starts) == len(ends):
            raise Exception("lens of the parameters have to be the same.")
        
        self.models = []
        for name, model, X, start, end in zip(names, models, Xs, starts, ends):
            self.models.append({'name': name, 'model': model, 'input': X, 'start': start, 'end': end})
        for mod in self.models:
            YieldPredictor = createYieldPredictor(mod['model'])
            mod['predictor'] = YieldPredictor(name = mod['model'], X = mod['input'], start = mod['start'], end = mod['end'])
    
    def fit(self):
        for mod in self.models:
            mod['pred'] = mod['predictor'].fit()
    
    def evaluate(self):
        for mod in self.models:
            mod['rmse'], mod['rmses'] = mod['predictor'].evaluate()
    
    def summary(self, metric = 'All'):
        print('#' + '-'*60 + '#')      
        if metric == 'Agg' or 'All':
            rmse = {'model': [], 'start date': [],'end date': [] , 'rmse': []}
            for mod in self.models:
                rmse['model'].append(mod['name'])
                rmse['start date'].append(mod['start'])
                rmse['end date'].append(mod['end'])
                rmse['rmse'].append(mod['rmse'])
            rmse = pd.DataFrame(rmse).set_index('model')
            print('Aggregate RMSE Summary:')
            print(rmse.to_string())
        if metric == 'Detail' or 'All':
            rmses = {'model': [], 'rmses': []}
            for mod in self.models:
                rmses['model'].append(mod['name'])
                rmses['rmses'].append(mod['rmses'])
            index = rmses['model']
            rmses = pd.concat(rmses['rmses'], axis=0)
            rmses.index = index
            print('Detailed RMSE Summary:')
            print(rmses.to_string())
        else:
            raise Exception('metric not allowed. Potential metrics: [All, Agg, Detail]')
        print('#' + '-'*60 + '#')

if __name__ == "__main__":
    YieldPredictor = createYieldPredictor('Linear Model')
    rwpredor = YieldPredictor(Xlist=['Inflation'], start='20190101')

    rwpredor.fit(log = False)
    print(rwpredor.forecast(5))

    '''
    assessor = ModelAssessor(
        names  = ['Long Baseline', 'Short Baselines', ], 
        models = ['Baseline', 'Baseline', ], 
        Xs     = [None, None, ], 
        starts = [None, '20130701', ], 
        ends   = [None, '20180701', ], 
        )
    assessor.fit()
    assessor.evaluate()
    assessor.summary()
    '''
    