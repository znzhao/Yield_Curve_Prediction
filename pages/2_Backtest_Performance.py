import datetime
import copy
import numpy as np
import streamlit as st
import plotly.express as px
import pandas as pd
from helper.utils import maxDrawDown, maxDrawUp, calSortino
from helper.lsctsplitter import loadAllYC
from predictors.modelcontroller import *
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

def createDisplayer():
    if 'stage' not in st.session_state:
        st.session_state.stage = 0
    def set_state():
        st.session_state.stage = 1

    st.title("Backtesting Performance")
    data = loadAllYC()
    mindate = min(data.index).to_pydatetime()
    maxdate = max(data.index).to_pydatetime()
    startdate = max(data.index).to_pydatetime() - datetime.timedelta(weeks=4)
    enddate = max(data.index).to_pydatetime()

    periods = st.sidebar.select_slider('Forecast Period', [1, 2, 3, 4, 5], 1)
    DateRange = st.sidebar.date_input('Test Range', value=(startdate, enddate), min_value=mindate, max_value=maxdate)
    params = createSideBar()
    submitted = st.sidebar.button("Run Backtest", type="primary", on_click=set_state)
    st.sidebar.divider()
    st.sidebar.markdown('''Created by Zhenning Zhao :link:[Website](https://sites.google.com/utexas.edu/zhenning-zhao/about)''')

    if submitted or 'truedatas' not in st.session_state:
        data = data[(data.index > pd.to_datetime(DateRange[0])) & (data.index <= pd.to_datetime(DateRange[1]))]
        refitbar = st.progress(0, text='Refitting Progress')
        
        truedatas = []
        preddatas = []
        basedatas = []

        for i, ycdate in enumerate(data.index):
            truedata = data[data.index<=ycdate]
            truedata = truedata.tail(1)
            truedata.columns = maturities
            truedata = truedata.T
            truedata.columns = [ycdate]
            if ycdate > maxdate:
                truedata = truedata.applymap(lambda x: np.nan)
            truedatas.append(truedata)
            try:
                preddata = createModel(ycdate, periods, params)
                preddata.columns = maturities
                preddata = preddata.T
                preddata.columns = [ycdate]
            except Exception as error:
                print(error)
                preddata = copy.deepcopy(truedata).applymap(lambda x: np.nan)
            preddatas.append(preddata)
            try:
                baseparams = copy.deepcopy(params)
                baseparams['model'] = 'Random Walk'
                basedata = createModel(ycdate, periods, baseparams)
                basedata.columns = maturities
                basedata = basedata.T
                basedata.columns = [ycdate]
            except Exception as error:
                print(error)
                basedata = copy.deepcopy(truedata).applymap(lambda x: np.nan)
            basedatas.append(basedata)
            refitbar.progress(min(i/len(data.index) + 0.01, 1.00), text='Refitting Progress {:.2f}%...'.format(100*min(i/len(data.index) + 0.01, 1.00)))
        refitbar.progress(1.0, text='Refitting Progress Completed.')
        truedatas = pd.concat(truedatas, axis = 1).T
        preddatas = pd.concat(preddatas, axis = 1).T
        basedatas = pd.concat(basedatas, axis = 1).T
        st.session_state['truedatas'] = truedatas
        st.session_state['preddatas'] = preddatas
        st.session_state['basedatas'] = basedatas


    metric_type = st.selectbox('Metrics', ['Direction Accuracy', 'Mean Squared Error', 'R Squared', 'Profit and Losses (PnLs)', 'Sortino Ratio', 'Max Drawdown', 'Max Drawup'])
    
    if metric_type == 'Direction Accuracy' or metric_type == 'Mean Squared Error' or metric_type == 'R Squared':
        modelmetric = []
        basermetric = []
        for col in st.session_state.truedatas.columns:
            if metric_type == 'Direction Accuracy':
                truedatas = np.sign(st.session_state.truedatas[col] - st.session_state.basedatas[col])
                preddatas = np.sign(st.session_state.preddatas[col] - st.session_state.basedatas[col])
                basedatas = np.sign(st.session_state.basedatas[col] - st.session_state.basedatas[col])    
                modelmetric.append(accuracy_score(truedatas, preddatas))
                basermetric.append(accuracy_score(truedatas, basedatas))
            elif metric_type == 'Mean Squared Error':
                modelmetric.append(mean_squared_error(st.session_state.truedatas[col], st.session_state.preddatas[col]))
                basermetric.append(mean_squared_error(st.session_state.truedatas[col], st.session_state.basedatas[col]))
            elif metric_type == 'R Squared':
                truedatas = (st.session_state.truedatas[col] - st.session_state.basedatas[col])
                preddatas = (st.session_state.preddatas[col] - st.session_state.basedatas[col])
                basedatas = (st.session_state.basedatas[col] - st.session_state.basedatas[col])  
                modelmetric.append(r2_score(truedatas, preddatas))
                basermetric.append(r2_score(truedatas, basedatas))
        
        modelmetric = pd.DataFrame({'Metrics':modelmetric}, index = maturities)
        modelmetric['Model Type'] = 'Model'
        basermetric = pd.DataFrame({'Metrics':basermetric}, index = maturities)
        basermetric['Model Type'] = 'Baseline'
        resdata = pd.concat([basermetric, modelmetric], axis = 0)
        resdata = resdata.reset_index(names = 'Maturities')
        
        colorlist = ['#183D3D','#93B1A6', ]
        fig = px.bar(resdata, x = 'Maturities', y = 'Metrics', color_discrete_sequence = colorlist[:len(resdata.columns)], color='Model Type', barmode='group',)
        fig.update_layout(legend=dict(title = None, orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x = 0.5))
        st.plotly_chart(fig)

    elif metric_type == 'Profit and Losses (PnLs)':
        col = st.selectbox('Maturities', maturities)
        truedatas = st.session_state.truedatas[col] - st.session_state.basedatas[col]
        preddatas = np.sign(st.session_state.preddatas[col] - st.session_state.basedatas[col])
        preddatas = pd.DataFrame({'Model': np.cumsum(preddatas * truedatas.values)})
        preddatas.index = st.session_state.truedatas.index
        truedatas = pd.DataFrame({'Baseline': np.cumsum(truedatas.values)})
        truedatas.index = st.session_state.truedatas.index
        resdata = pd.concat([truedatas, preddatas], axis = 1)
        colorlist = ['#183D3D','#93B1A6', ]
        fig = px.line(resdata, 
                      color_discrete_sequence = colorlist[:len(resdata.columns)],
                      labels={
                              "value": "Percent (%)",
                              "date": "Date",
                             })
        fig.update_layout(legend=dict(title = None, orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x = 0.5))
        st.plotly_chart(fig)

    elif metric_type == 'Sortino Ratio' or metric_type == 'Max Drawdown' or metric_type == 'Max Drawup':
        resdatas = {}
        for col in st.session_state.truedatas.columns:
            truedatas = st.session_state.truedatas[col] - st.session_state.basedatas[col]
            preddatas = np.sign(st.session_state.preddatas[col] - st.session_state.basedatas[col])
            preddatas = pd.DataFrame({'Model': np.cumsum(preddatas * truedatas.values)})
            preddatas.index = st.session_state.truedatas.index
            truedatas = pd.DataFrame({'Baseline': np.cumsum(truedatas.values)})
            truedatas.index = st.session_state.truedatas.index
            resdata = pd.concat([truedatas, preddatas], axis = 1)
            if metric_type == 'Sortino Ratio':
                metricfunc = calSortino
            elif metric_type == 'Max Drawdown':
                metricfunc = maxDrawDown
            elif metric_type == 'Max Drawup':
                metricfunc = maxDrawUp
            resdatas[col] = [metricfunc(resdata['Baseline'].tolist()), metricfunc(resdata['Model'].tolist())]
        resdata = pd.DataFrame(resdatas, index=['Baseline', 'Model']).T

        basedatas = pd.DataFrame(resdata['Baseline'])
        basedatas.columns = ['Metrics']
        basedatas['Model Type'] = 'Baseline'
        preddatas = pd.DataFrame(resdata['Model'])
        preddatas.columns = ['Metrics']
        preddatas['Model Type'] = 'Model'
        resdata = pd.concat([basedatas,preddatas], axis = 0)
        resdata = resdata.reset_index(drop=False, names='Maturities')
        colorlist = ['#183D3D','#93B1A6', ]
        fig = px.bar(resdata, x = 'Maturities', y = 'Metrics', color_discrete_sequence = colorlist[:len(resdata.columns)], color='Model Type', barmode='group',)
        fig.update_layout(legend=dict(title = None, orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x = 0.5))
        st.plotly_chart(fig)

if __name__ == "__main__":
    createDisplayer()