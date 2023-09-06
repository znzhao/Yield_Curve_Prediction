import os
import pandas as pd
import numpy as np
import tqdm
import datetime
from scipy.optimize import minimize
from matplotlib import pyplot
from helper.utils import Timer

def loadYC(display = False):
    '''
    Load the yield curve data from the saved drive
    '''
    dirpath = './MktData/ycg/'
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    dirlist = os.listdir(dirpath)
    
    datas = []
    if display: print('Starting Loading Yield Curve Data...')
    with Timer('Data Loader', display):
        for dir in dirlist:
            if dir == 'ycg.csv':
                continue
            subdir = dirpath + dir
            filelist = os.listdir(subdir)
            for file in filelist:
                if display: print('Loading Data: {}...'.format(file), end="\r")
                data = pd.read_csv(subdir + '/' + file, index_col = 'date')
                datas.append(data)
        datas = pd.concat(datas, axis = 0)
        datas.index = pd.to_datetime(datas.index)
        datas = datas.sort_index()
    datas.to_csv(dirpath+'ycg.csv', index=True, header=True)
    return datas

def loadAllYC(display = False):
    '''
    Load the yield curve data from the saved drive
    '''
    path = 'https://raw.githubusercontent.com/znzhao/Yield_Curve_Predition/main/MktData/ycg/ycg.csv'
    data = pd.read_csv(path, index_col = 'date')
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    return data

class LSCTsplitter(object):
    def __init__(self, lambdas = 0.498, maturities = [1/12, 2/12, 3/12, 4/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]):
        self.lambdas = lambdas
        self.maturities = maturities
        loadL = np.ones(len(maturities)).tolist()
        loadS = [(1-np.exp(-self.lambdas*t)) / (self.lambdas*t) for t in self.maturities]
        loadC = [(1-np.exp(-self.lambdas*t)) / (self.lambdas*t) - np.exp(-self.lambdas*t) for t in self.maturities]
        loadT = [2*(1-np.exp(-self.lambdas*t)) / (self.lambdas*t) - np.exp(-self.lambdas*t)*(self.lambdas*t + 2) for t in self.maturities]
        
        self.loadings = np.array([loadL, loadS, loadC, loadT])
    
    def fit(self, data, refit = False):
        self.data = data
        dirpath = 'https://raw.githubusercontent.com/znzhao/Yield_Curve_Predition/main/MktData/factors'
        if os.path.exists(dirpath + '/factors.csv'):
            print('Factors exist. Read from driver-saved data.')
            lambdas = self.loadfactors()
        
        if refit or (not os.path.exists(dirpath+'/factors.csv')) or lambdas != self.lambdas:
            print('Factors not exist. Run LSCT splitter optimization.')
            with Timer('LSCT Optimization'):
                values = data.values
                self.factors = []
                for row in tqdm.tqdm(values):
                    res = minimize(lambda x: self.__objfunc(x, row), x0 = np.zeros(4))
                    self.factors.append(pd.DataFrame(res.x).T)
                self.factors = pd.concat(self.factors, axis = 0)
                self.factors.columns = ['level', 'slope', 'curvature', 'twist']
                self.factors.index = data.index
                self.savefactors()
        return self
        
    
    def __objfunc(self, factors, trueys):
        factors = np.expand_dims(factors, axis=0)
        return np.nansum((trueys - self.__pred(factors))**2)

    def __pred(self, factors):
        return factors.dot(self.loadings)
    
    def savefactors(self, path = './MktData/factors/factors.csv'):
        self.factors['lambdas'] = self.lambdas
        self.factors.to_csv(path, index=True, header=True)
        self.factors = self.factors.drop(columns=['lambdas'])

    def loadfactors(self, path = './MktData/factors/factors.csv'):
        self.factors = pd.read_csv(path, index_col = 'date')
        lambdas = self.factors.lambdas[0]
        self.factors = self.factors.drop(columns=['lambdas'])
        self.factors.index = pd.to_datetime(self.factors.index)
        return lambdas
    
    def inverse(self, factors = None):
        if factors == None:
            factors = self.factors
        self.preds = pd.DataFrame(factors.values.dot(self.loadings))
        self.preds.columns = self.data.columns
        self.preds.index = self.data.index
        return self.preds

    def plotfactors(self, *args, **kwargs):
        if 'end' not in kwargs.keys():
            end = datetime.datetime.now()
        else:
            end = kwargs['end']

        if 'start' not in kwargs.keys():
            start = end - datetime.timedelta(weeks=52*10)
        else:
            start = kwargs['start']
        
        data_to_plot = self.factors[(self.factors.index >= start) & (self.factors.index <= end)]

        fig = pyplot.figure(figsize=(16,9))
        axfactors  = []
        axloadings = []
        titles = ['level', 'slope', 'curvature', 'twist']
        for i in range(4):
            axfactors.append(pyplot.subplot2grid((4,3), (i,0), colspan = 2))
            axloadings.append(pyplot.subplot2grid((4,3), (i,2)))
            axfactors[i].plot(data_to_plot.index, data_to_plot.iloc[:,i].values)
            axfactors[i].set_title(titles[i])
            [axfactors[i].spines[loc_axis].set_visible(False) for loc_axis in ['top','right', 'bottom']]
            axloadings[i].plot(self.maturities, self.loadings[i], marker = 'o')
            axloadings[i].set_title(titles[i] +' loading')
            [axloadings[i].spines[loc_axis].set_visible(False) for loc_axis in ['top','right', 'bottom']]
        fig.tight_layout()
        pyplot.show()

    def plot(self, columns = [[0, 2, 4], [5, 8, 10]], *args, **kwargs):
        if 'end' not in kwargs.keys():
            end = datetime.datetime.now()
        else:
            end = kwargs['end']

        if 'start' not in kwargs.keys():
            start = end - datetime.timedelta(weeks=52*10)
        else:
            start = kwargs['start']
        
        data_to_plot = self.data[(self.data.index >= start) & (self.data.index <= end)]
        pred_to_plot = self.preds[(self.preds.index >= start) & (self.preds.index <= end)]
        columns = np.array(columns)
        fig, ax = pyplot.subplots(columns.shape[0], columns.shape[1], figsize = (16,9))
        for i in range(columns.shape[0]):
            for j in range(columns.shape[1]):
                
                ax[i,j].plot(data_to_plot.index, data_to_plot.iloc[:,columns[i][j]], label = 'True Data')
                ax[i,j].plot(pred_to_plot.index, pred_to_plot.iloc[:,columns[i][j]], label = 'Pred Data')
                [ax[i,j].spines[loc_axis].set_visible(False) for loc_axis in ['top','right', 'bottom']]
                ax[i,j].set_title(data_to_plot.columns[columns[i][j]])
        pyplot.legend(loc = 'best', frameon = False)
        fig.tight_layout()
        pyplot.show()

if __name__ == "__main__":
    '''
    lscsplitter = LSCTsplitter()
    lscsplitter.fit(data=data, refit=False)
    lscsplitter.inverse()
    lscsplitter.plotfactors()
    lscsplitter.plot()'''
    data = loadAllYC()
    print(data)
    