import os
import numpy as np
import datetime
import copy
import scipy
import pandas as pd
from lsctsplitter import loadAllYC, LSCTsplitter
from utils import dropCommonNan, maturities
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error

class YieldPredictor(object):
    def __init__(self, name = 'Baseline', Xlist = [], start = None, end = None, display = False, *args, **kwargs):
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
        self.Xlist = Xlist

        if 'lambdas' in kwargs.keys():
            self.lambdas = kwargs['lambdas']
        else:
            self.lambdas = 0.498            
        if 'lags' in kwargs.keys():
            self.lags = kwargs['lags']
        else:
            self.lags = 5
        if 'log' in kwargs.keys():
            self.log = kwargs['log']
        else:
            self.log = False

        if ('LSCT' in self.Xlist) or (self.model == 'Lagged LSCT') or (self.model == 'Linear LSCT'):
            self.lscsplitter = LSCTsplitter(self.lambdas)
            self.lscsplitter.fit(data=self.Y, refit=False)
        
        if ('Stock' in self.Xlist):
            self.lscsplitter = LSCTsplitter(self.lambdas)
            self.lscsplitter.fit(data=self.Y, refit=False)
        self.rawX = []

    def __logTransform(self):
        self.minY = np.nanmin(self.Y.values)-1e-4
        self.Y = self.Y.applymap(lambda x: np.log(x+self.minY))

    def __expTransform(self, Y = None):
        if Y is None:
            self.Y = self.Y.apply(lambda x: np.exp(x)-self.minY)
            return self.Y
        else:
            return Y.apply(lambda x: np.exp(x)-self.minY)

    def __fitBaseline(self):
        self.pred = copy.deepcopy(self.Y.shift(1))
        # deal with NaN values
        self.pred = self.pred.T
        maturity_names = self.pred.index
        self.pred.index = maturities
        self.pred = self.pred.interpolate(method='index', axis=0, limit_direction = 'both')
        self.pred.index = maturity_names
        self.pred = self.pred.T
        return self.pred
    
    def __forecastBaseline(self, periods):
        oos_pred = []
        for p in range(periods):
            oos_pred.append(copy.deepcopy(self.Y.tail(1)))
        oos_pred =  pd.concat(oos_pred, axis=0).reset_index(drop=True)
        # deal with NaN values
        oos_pred = oos_pred.T
        maturity_names = oos_pred.index
        oos_pred.index = maturities
        oos_pred = oos_pred.interpolate(method='index', axis=0, limit_direction = 'both')
        oos_pred.index = maturity_names
        oos_pred = oos_pred.T
        return oos_pred

    def __fitLagLSCT(self, *args, **kwargs):
        self.pred = copy.deepcopy(self.lscsplitter.inverse().shift(1))
        return self.pred
    
    def __forecastLagLSCT(self, periods):
        oos_pred = []
        for p in range(periods):
            oos_pred.append(copy.deepcopy(self.lscsplitter.inverse().tail(1)))
        return pd.concat(oos_pred, axis=0).reset_index(drop=True)

    def __fitLinearLSCT(self, *args, **kwargs):
        if self.log:
            self.__logTransform()
        
        self.predmodel = LinearRegression()
        self.X = []
        for lag in range(self.lags):
            shifted_factors = self.lscsplitter.factors.shift(lag+1)
            shifted_factors.columns = ['L{}'.format(lag+1) + x for x in shifted_factors.columns]
            self.X.append(shifted_factors)
        self.X = pd.concat(self.X, axis=1)

        # deal with NaN values
        self.Y = self.Y[~self.X.isna().any(axis=1)]
        self.Y = self.Y.T
        maturity_names = self.Y.index
        self.Y.index = self.lscsplitter.maturities
        self.Y = self.Y.interpolate(method='index', axis=0, limit_direction = 'both')
        self.Y.index = maturity_names
        self.Y = self.Y.T
        self.X = self.X[~self.X.isna().any(axis=1)]
        
        self.predmodel = self.predmodel.fit(X = self.X, y = self.Y)
        self.pred = self.predmodel.predict(self.X)    
        self.pred = pd.DataFrame(self.pred, columns=self.Y.columns)
        if self.log:
            self.pred = self.__expTransform(self.pred)
        return self.pred
    
    def __forecastLinearLSCT(self, periods):
        oosX = pd.concat([self.lscsplitter.factors.tail(1), self.X.tail(1).iloc[:, :-4]], axis=1)
        oosX.columns = self.X.columns
        oosX = oosX.reset_index(drop=True)
        
        oos_pred = []
        for p in range(periods):
            print(oosX)
            pred = self.predmodel.predict(oosX)
            pred = pd.DataFrame(pred, columns=self.Y.columns)
            if self.log:
                pred = self.__expTransform(pred)
            oos_pred.append(pred)
            oosX = pd.concat([LSCTsplitter(lambdas = self.lambdas).fit(data=pred).factors, oosX.iloc[:, :-4]], axis=1)
            oosX.columns = self.X.columns
            oosX = oosX.reset_index(drop=True)
        return pd.concat(oos_pred, axis=0)

    def __fitLinear(self, *args, **kwargs):
        if 'kernel' in kwargs.keys():
            kernel = kwargs['kernel']
        else:
            kernel = 1.0
        if 'alpha' in kwargs.keys():
            alpha = kwargs['alpha']
        else:
            alpha = 1.0

        if self.log:
            self.__logTransform()

        if kernel == 'Linear':
            self.predmodel = LinearRegression()
        elif kernel == 'LASSO':
            self.predmodel = Lasso(alpha = alpha)
        elif kernel == 'Ridge':
            self.predmodel = Ridge(alpha = alpha)
        
        self.X = []
        if 'LSCT' in self.Xlist:
            self.rawX.append(self.lscsplitter.factors)
            self.rawX = pd.concat(self.rawX, axis=1)
            for lag in range(self.lags):
                shiftXs = self.rawX.shift(lag+1)
                shiftXs.columns = ['L{}'.format(lag+1) + x for x in shiftXs.columns]
                self.X.append(shiftXs)
        if self.Xlist == []:
            self.X = [pd.DataFrame({'Const': [0]*self.Y.shape[0]}, index=self.Y.index)]
        print(self.X)
        self.X = pd.concat(self.X, axis=1)
        # deal with NaN values
        self.Y = self.Y[~self.X.isna().any(axis=1)]
        self.Y = self.Y.T
        maturity_names = self.Y.index
        self.Y.index = maturities
        self.Y = self.Y.interpolate(method='index', axis=0, limit_direction = 'both')
        self.Y.index = maturity_names
        self.Y = self.Y.T
        self.X = self.X[~self.X.isna().any(axis=1)]
        
        self.predmodel = self.predmodel.fit(X = self.X, y = self.Y)
        self.pred = self.predmodel.predict(self.X)    
        self.pred = pd.DataFrame(self.pred, columns=self.Y.columns)
        if self.log:
            self.pred = self.__expTransform(self.pred)
        return self.pred
    
    def __forecastLinear(self, periods):
        if 'LSCT' in self.Xlist:
            oosX = pd.concat([self.lscsplitter.factors.tail(1), self.X.tail(1).iloc[:, :-4]], axis=1)
            oosX.columns = self.X.columns
            oosX = oosX.reset_index(drop=True)
        if  self.Xlist == []:
            oosX = self.X.tail(1)

        oos_pred = []
        for p in range(periods):
            print(oosX)
            pred = self.predmodel.predict(oosX)
            pred = pd.DataFrame(pred, columns=self.Y.columns)
            if self.log:
                pred = self.__expTransform(pred)
            oos_pred.append(pred)
            if 'LSCT' in self.Xlist:
                oosX = pd.concat([LSCTsplitter(lambdas = self.lambdas).fit(data=pred).factors, oosX.iloc[:, :-4]], axis=1)
                oosX.columns = self.X.columns
                oosX = oosX.reset_index(drop=True)
        return pd.concat(oos_pred, axis=0)


    def fit(self, *args, **kwargs):
        if self.model == 'Baseline':
            return self.__fitBaseline()
        if self.model == 'Lagged LSCT':
            return self.__fitLagLSCT(*args, **kwargs)
        if self.model == 'Linear LSCT':
            return self.__fitLinearLSCT(*args, **kwargs)
        if self.model == 'Linear Model':
            return self.__fitLinear(*args, **kwargs)
        else:
            raise Exception("model type [{}] not allowed. Potential models: [Baseline, Lagged LSCT, Linear LSCT, Linear Model]")

    def forecast(self, periods = 1):
        if self.model == 'Baseline':
            return self.__forecastBaseline(periods)
        if self.model == 'Lagged LSCT':    
            return self.__forecastLagLSCT(periods)
        if self.model == 'Linear LSCT':
            return self.__forecastLinearLSCT(periods)
        if self.model == 'Linear Model':
            return self.__forecastLinear(periods)
        else:
            raise Exception("model type [{}] not allowed. Potential models: [Baseline, Lagged LSCT, Linear LSCT, Linear Model]")

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
    rwpredor = YieldPredictor('Linear LSCT' ,start='20190101')
    rwpredor.fit(log = True)
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
    