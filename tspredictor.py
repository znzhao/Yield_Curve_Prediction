import os
import numpy as np
import datetime
import copy
import pandas as pd
from lsctsplitter import loadAllYC, LSCTsplitter
from utils import dropCommonNan
from sklearn.metrics import mean_squared_error

class YieldPredictor(object):
    def __init__(self, name = 'Baseline', X = [], start = None, end = None, display = False):
        # initialize the output to be predicted
        self.Y = loadAllYC()
        self.model = name
        if start is not None:
            start = datetime.datetime.strptime(start, '%Y%m%d')
            self.Y = self.Y[self.Y.index >= start]
        if end is not None:
            end = datetime.datetime.strptime(end, '%Y%m%d')
            self.Y = self.Y[self.Y.index < end]            
        self.columns = self.Y.columns
        self.index = self.Y.index

        Xdata = []
        if 'LSCT' in X:
            path = './MktData/factors/factors.csv'
            factors = pd.read_csv(path, index_col = 'date')
            factors.index = pd.to_datetime(factors.index)
            Xdata.append(factors)
        
        if len(Xdata) == 1:
            self.X = Xdata[0]
        elif len(Xdata) > 1:
            self.X = Xdata[0]
            for i in range(1, len(self.X)):
                self.X = pd.merge(self.X, Xdata[i], left_index = True, right_index = True)

    def __fitBaseline(self):
        self.pred = copy.deepcopy(self.Y.shift(1))
        return self.pred
    
    def __fitLagLSCT(self, *args, **kwargs):
        if 'lambdas' in kwargs.keys():
            lambdas = kwargs['lambdas']
        else:
            lambdas = 0.498
        lscsplitter = LSCTsplitter(lambdas)
        lscsplitter.fit(data=self.Y, refit=False)
        self.pred = copy.deepcopy(lscsplitter.inverse().shift(1))
        return self.pred
    
    def fit(self, *args, **kwargs):
        if self.model == 'Baseline':
            return self.__fitBaseline()
        if self.model == 'Lagged LSCT':
            return self.__fitLagLSCT(*args, **kwargs)
        else:
            raise Exception("model type [{}] not allowed. Potential models: [Baseline, ]")

    def evaluate(self):
        true_y, pred_y = dropCommonNan(self.Y.values, self.pred.values)
        self.rmse = np.sqrt(mean_squared_error(true_y, pred_y))
        
        self.rmses = []
        for i in range(len(self.columns)):
            true_y, pred_y = dropCommonNan(self.Y.iloc[:, i].values, self.pred.iloc[:, i].values)
            if len(true_y) == 0:
                self.rmses.append(np.nan)
            else:
                self.rmses.append([mean_squared_error(true_y, pred_y, squared=False)])
        self.rmses = pd.DataFrame(dict(zip(self.columns, self.rmses)))
        self.rmses.index = [self.model]
        return self.rmse, self.rmses
    
    def save(self):
        dirpath = './Output/'
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        dirpath = './Output/PredData/'
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        pred = self.fit()
        pred.to_csv(dirpath + str(self.model) + '.csv', index=True, header=True)

class ModelAssessor(object):
    def __init__(self, names = ['Baseline'], models = ['Baseline'], Xs = [None], starts = [None], ends = [None]):
        if not len(names) == len(models) == len(Xs) == len(starts) == len(ends):
            raise Exception("lens of the parameters have to be the same.")
        
        self.models = []
        for name, model, X, start, end in zip(names, models, Xs, starts, ends):
            self.models.append({'name': name, 'model': model, 'input': X, 'start': start, 'end': end})
        for mod in self.models:
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
    rwpredor = YieldPredictor()
    rwpredor.save()
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
    