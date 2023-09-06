import os
import re
import copy
import requests
import xmltodict
import datetime
import numpy as np
import pandas as pd
from helper.utils import Timer, getYahooTicker
import yfinance as yahooFinance


def parseMonthlyYieldCurve(value = 'daily_treasury_yield_curve', *args, **kwargs):
    '''
    Parse the monthly yield curve into a pandas DataFrame.
    '''
    if 'yyyymm' not in kwargs.keys():
        yyyy = str(datetime.datetime.now().year)
        mm = str(datetime.datetime.now().month)
        mm = mm if len(mm) == 2 else '0'+ mm
        yyyymm = yyyy+mm
    else:
        yyyymm = kwargs['yyyymm']
    
    # download data from the US treasury
    treasury_html = 'https://home.treasury.gov/resource-center/data-chart-center'
    treasury_html += '/interest-rates/pages'
    treasury_html += '/xml?data={value}&field_tdr_date_value_month={yyyymm}'.format(value = value, yyyymm = yyyymm)
    response = requests.get(treasury_html)
    data = xmltodict.parse(response.content)

    # parse data into dataframe
    yc = {
          'date':      [],
          'BC_1MONTH': [],
          'BC_2MONTH': [],
          'BC_3MONTH': [],
          'BC_4MONTH': [],
          'BC_6MONTH': [],
          'BC_1YEAR':  [],
          'BC_2YEAR':  [],
          'BC_3YEAR':  [],
          'BC_5YEAR':  [],
          'BC_7YEAR':  [],
          'BC_10YEAR': [],
          'BC_20YEAR': [],
          'BC_30YEAR': [],
          }
    
    for entry in data['feed']['entry']:
        yc['date'] += [entry['content']['m:properties']['d:NEW_DATE']['#text']]
        for col in list(yc.keys())[1:]:
            try:
                yc[col] += [float(entry['content']['m:properties']['d:'+col]['#text'])]
            except:
                yc[col] += [np.nan]
            
    yc = pd.DataFrame(yc)
    yc['date'] = pd.to_datetime(yc.date)
    yc = yc.set_index('date')
    return yc

def parseYieldCurve(value = 'daily_treasury_yield_curve', start = '199001', *args, **kwargs):
    '''
    Parse the yield curve into a pandas DataFrame and keep the data up to date.
    '''
    startyyyy = int(start[:4])
    startmm = int(start[4:6])
    if 'end' in kwargs.keys():        
        endyyyy = int(kwargs['end'][:4])
        endmm = int(kwargs['end'][4:6])
    else:
        endyyyy = datetime.datetime.now().year
        endmm = datetime.datetime.now().month
    
    # obtain the last year and month saved in the drive
    dirpath = './MktData/ycg'
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    if os.listdir(dirpath) == []:
        lastyyyymm = startyyyy
    else:
        lastyyyy = max([int(x) for x in os.listdir(dirpath)])
        lastyyyy = str(lastyyyy)
        lastyyyymm = max([int(re.findall(r'[\d]+', x)[0]) for x in os.listdir(dirpath + '/' + lastyyyy)])
    ycg = []
    with Timer('DataParser'):
        for yyyy in range(startyyyy, endyyyy+1):
            if not os.path.exists(dirpath + '/' + str(yyyy)):
                os.mkdir(dirpath + '/' + str(yyyy))
            # obtain beginning month and stop month for each year
            if yyyy == startyyyy:
                beginmm = startmm
            else:
                beginmm = 1
            if yyyy == endyyyy:
                stopmm = endmm
            else:
                stopmm = 12
            yyyy = str(yyyy)
            
            # download data for yyyymm
            for mm in range(beginmm, stopmm+1):
                mm = str(mm)
                mm = mm if len(mm) == 2 else '0'+ mm
                yyyymm = yyyy+mm
                path = dirpath + '/{}/ycg{}.csv'.format(yyyy, yyyymm)
                if os.path.isfile(path) and (yyyymm != str(lastyyyymm)):
                    print('History Data {} Exist.   '.format(yyyymm), end="\r")
                    ycm = pd.read_csv(path, index_col='date')
                    ycm.index = pd.to_datetime(ycm.index)
                    ycg.append(ycm)
                    continue

                print('Dowloading Data {}...      '.format(yyyymm), end="\r")
                ycm = parseMonthlyYieldCurve(value = value, yyyymm = yyyymm)
                ycm.to_csv(path, index=True, header=True)
                ycg.append(ycm)
        print('Data Parser ended.' + ' '*20)
    ycg = pd.concat(ycg, axis = 0)
    return ycg

def updateYC():
    data = parseYieldCurve()
    print(data)

def parseFredData(series_id, observation_start = '2003-01-02', observation_end = '9999-12-31', *args, **kwargs):
    '''
    Parse the FRED series data into a pandas DataFrame.
    '''
    
    # download data from the US treasury
    api_key = '5bf337418324db94a2989c2312d61079'
    fred_html = 'https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&observation_start={observation_start}&observation_end={observation_end}'.format(series_id = series_id, api_key = api_key, observation_start = observation_start, observation_end = observation_end)
    response = requests.get(fred_html)
    data = xmltodict.parse(response.content)
    fred_data = {'date':[], series_id:[]}
    for entry in data['observations']['observation']:
        fred_data['date'].append(entry['@date'])
        fred_data[series_id].append(entry['@value'])
    fred_data = pd.DataFrame(fred_data)
    fred_data['date'] = pd.to_datetime(fred_data['date'])
    fred_data = fred_data.set_index('date')
    return fred_data

def parseBreakEvenInflation():
    '''
    Parse the break even inflation rate into a pandas DataFrame.
    '''
    t5yie = parseFredData('T5YIE')
    t10yie = parseFredData('T10YIE')   
    tyie = pd.merge(t5yie, t10yie, left_index=True, right_index=True, how='outer')
    dirpath = './MktData/tyie'
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    tyie.to_csv(dirpath + '/tyie.csv', index=True, header=True)
    return tyie

def parseYahooData(ticker):
    name = copy.deepcopy(ticker)
    ticker = getYahooTicker(ticker)
    ticker_info = yahooFinance.Ticker(ticker)
    data = ticker_info.history(period='max')
    dirpath = './MktData/stock'
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    data.to_csv(dirpath + '/'+ name +'.csv', index=True, header=True)
    return data

def parseSP500():
    sp500 = parseYahooData('SP500')
    return sp500

if __name__ == "__main__":
    sp500 = parseBreakEvenInflation()