
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

class PredClass(object):
    def __init__(self, Xlist = ['LSCT'], start = None, end = None, display = False, *args, **kwargs):
        # initialize the output to be predicted
        self.Y = loadAllYC()
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

        self.rawX = []

        if 'Stock' in self.Xlist:
            stock_data = pd.read_csv('https://raw.githubusercontent.com/znzhao/Yield_Curve_Prediction/main/MktData/stock/SP500.csv')
            stock_data = stock_data.rename(columns={'Date': 'date'})
            stock_data['date'] = stock_data.apply(lambda x: x['date'][:10], axis=1)
            stock_data = stock_data[['date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            self.stockcolumns = ['stock_open', 'stock_high', 'stock_low', 'stock_close', 'stock_volume']
            stock_data = self.__dataClean(stock_data, columns=self.stockcolumns)
            self.rawX.append(stock_data)

        if 'Inflation' in self.Xlist:
            tyie_data = pd.read_csv('https://raw.githubusercontent.com/znzhao/Yield_Curve_Prediction/main/MktData/tyie/tyie.csv', na_values='.')
            self.tyiecolumns = ['T5YIE', 'T10YIE']
            tyie_data = self.__dataClean(tyie_data)
            self.rawX.append(tyie_data)
            
        if 'LSCT' in self.Xlist:
            self.lscsplitter = LSCTsplitter(self.lambdas)
            self.lscsplitter.fit(data=self.Y, refit=False)

    def __dataClean(self, dataset, columns=None):
        dataset['date'] = pd.to_datetime(dataset.date)
        dataset = dataset.set_index('date')
        if columns is not None:
            dataset.columns = columns
        dataset = dataset.applymap(lambda x: np.log(x) if x>0 else np.nan)
        dataset = dataset.dropna(axis=0)
        dataset = pd.merge(dataset, pd.DataFrame({'ToDrop': [1]*self.Y.shape[0]}, index=self.Y.index), 
                      how = 'right', left_index=True, right_index=True)
        dataset = dataset.drop(columns=['ToDrop'])
        dataset = dataset.fillna(method = 'ffill')
        return dataset
    
    def logTransform(self):
        self.minY = np.nanmin(self.Y.values)-1e-4
        self.Y = self.Y.applymap(lambda x: np.log(x+self.minY))

    def expTransform(self, Y = None):
        if Y is None:
            self.Y = self.Y.apply(lambda x: np.exp(x)-self.minY)
            return self.Y
        else:
            return Y.apply(lambda x: np.exp(x)-self.minY)

    def fit(self, *args, **kwargs):
        raise(NotImplementedError)
    
    def forecast(self, periods = 1):
        raise(NotImplementedError)
    
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

class BaselinePredClass(PredClass):
    def fit(self):
        self.pred = copy.deepcopy(self.Y.shift(1))
        # deal with NaN values
        self.pred = self.pred.T
        maturity_names = self.pred.index
        self.pred.index = maturities
        self.pred = self.pred.interpolate(method='index', axis=0, limit_direction = 'both')
        self.pred.index = maturity_names
        self.pred = self.pred.T
        return self.pred
    
    def forecast(self, periods):
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

class LagLSCTPredClass(PredClass):
    def fit(self, *args, **kwargs):
        self.pred = copy.deepcopy(self.lscsplitter.inverse().shift(1))
        return self.pred
    
    def forecast(self, periods):
        oos_pred = []
        for p in range(periods):
            oos_pred.append(copy.deepcopy(self.lscsplitter.inverse().tail(1)))
        return pd.concat(oos_pred, axis=0).reset_index(drop=True)
    
class LinearLSCTPredClass(PredClass):
    def fit(self, *args, **kwargs):
        if self.log:
            self.logTransform()
        
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
            self.pred = self.expTransform(self.pred)
        return self.pred
    
    def forecast(self, periods):
        oosX = pd.concat([self.lscsplitter.factors.tail(1), self.X.tail(1).iloc[:, :-4]], axis=1)
        oosX.columns = self.X.columns
        oosX = oosX.reset_index(drop=True)
        
        oos_pred = []
        for p in range(periods):
            pred = self.predmodel.predict(oosX)
            pred = pd.DataFrame(pred, columns=self.Y.columns)
            if self.log:
                pred = self.expTransform(pred)
            oos_pred.append(pred)
            oosX = pd.concat([LSCTsplitter(lambdas = self.lambdas).fit(data=pred).factors, oosX.iloc[:, :-4]], axis=1)
            oosX.columns = self.X.columns
            oosX = oosX.reset_index(drop=True)
        return pd.concat(oos_pred, axis=0)

class LinearPredClass(PredClass):
    def fit(self, *args, **kwargs):
        if 'kernel' in kwargs.keys():
            kernel = kwargs['kernel']
        else:
            kernel = 1.0
        if 'alpha' in kwargs.keys():
            alpha = kwargs['alpha']
        else:
            alpha = 1.0

        if kernel == 'Linear':
            self.predmodel = LinearRegression()
        elif kernel == 'LASSO':
            self.predmodel = Lasso(alpha = alpha)
        elif kernel == 'Ridge':
            self.predmodel = Ridge(alpha = alpha)
        else:
            self.predmodel = LinearRegression()

        if self.log:
            self.logTransform()

        self.X = []

        if 'LSCT' in self.Xlist:
            self.rawX.append(self.lscsplitter.factors)
            self.rawX = pd.concat(self.rawX, axis=1)
            for lag in range(self.lags):
                shiftXs = self.rawX.shift(lag+1)
                shiftXs.columns = ['L{}'.format(lag+1) + x for x in shiftXs.columns]
                self.X.append(shiftXs)

        elif self.Xlist != []:
            self.rawX = pd.concat(self.rawX, axis=1)
            for lag in range(self.lags):
                shiftXs = self.rawX.shift(lag+1)
                shiftXs.columns = ['L{}'.format(lag+1) + x for x in shiftXs.columns]
                self.X.append(shiftXs)
            
        else:
            self.X = [pd.DataFrame({'Const': [1]*self.Y.shape[0]}, index=self.Y.index)]        

        self.X = pd.concat(self.X, axis=1)

        # deal with NaN values
        self.Y = self.Y[~self.X.isna().any(axis=1)]
        self.Y = self.Y.T
        maturity_names = self.Y.index
        self.Y.index = maturities
        self.Y = self.Y.interpolate(method='index', axis=0, limit_direction = 'both')
        self.Y.index = maturity_names
        self.Y = self.Y.T

        if 'Stock' in self.Xlist:
            stock_data = self.rawX[self.stockcolumns][~self.X.isna().any(axis=1)]
            self.Y = pd.concat([self.Y, stock_data], axis=1)
        if 'Inflation' in self.Xlist:
            tyie_data = self.rawX[self.tyiecolumns][~self.X.isna().any(axis=1)]
            self.Y = pd.concat([self.Y, tyie_data], axis=1)

        self.X = self.X[~self.X.isna().any(axis=1)]
        
        self.predmodel = self.predmodel.fit(X = self.X, y = self.Y)
        self.pred = self.predmodel.predict(self.X)    
        self.pred = pd.DataFrame(self.pred, columns=self.Y.columns)
        self.pred = self.pred[self.columns]
        if self.log:
            self.pred = self.expTransform(self.pred)
        return self.pred
    
    def forecast(self, periods):
        if  self.Xlist == []:
            oosX = self.X.tail(1)
        else:
            oosX = pd.concat([self.rawX.tail(1), self.X.tail(1).iloc[:, :-self.rawX.shape[1]]], axis=1)
            print(oosX.columns)
            oosX.columns = self.X.columns
            oosX = oosX.reset_index(drop=True)

        oos_pred = []
        for p in range(periods):
            pred = self.predmodel.predict(oosX)
            pred = pd.DataFrame(pred, columns=self.Y.columns)
            ycpred = pred[self.columns]
            if self.log:
                ycpred = self.expTransform(ycpred)      
            oos_pred.append(ycpred)

            newoosX = []
            if 'Stock' in self.Xlist:
                stockpred = pred[self.stockcolumns]
                newoosX.append(stockpred)
                
            if 'Inflation' in self.Xlist:
                stockpred = pred[self.tyiecolumns]
                newoosX.append(stockpred)

            if 'LSCT' in self.Xlist:
                newoosX.append(LSCTsplitter(lambdas = self.lambdas).fit(data=ycpred).factors)

            if self.Xlist != []:
                oosX = pd.concat(newoosX + [oosX.iloc[:, :-self.rawX.shape[1]]], axis=1)
            print(self.X.columns)
            print(oosX.columns)
            
            oosX.columns = self.X.columns
            oosX = oosX.reset_index(drop=True)
        result = pd.concat(oos_pred, axis=0)
        return result
