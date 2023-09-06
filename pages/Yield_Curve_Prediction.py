import datetime
import copy
import numpy as np
import streamlit as st
import plotly.express as px
import pandas as pd
from helper.lsctsplitter import loadAllYC
from helper.ycpredictor import createYieldPredictor
import matplotlib.pyplot as pyplot
maturities = ['1 Month', '2 Month', '3 Month', '4 Month', '6 Month', '1 Year', '2 Year', '3 Year', '5 Year', '7 Year', '10 Year', '20 Year', '30 Year',]

def createSideBar(showpred, ycdate, periods):
    pred = st.sidebar.selectbox('Prediction Model', ['Random Walk', 'Lagged LSCT', 'Linear LSCT', 'Linear Model'], disabled = not showpred)
    if pred == 'Random Walk':
        YieldPredictor = createYieldPredictor('Baseline')
        predictor = YieldPredictor(start = (ycdate - datetime.timedelta(days=periods+7)).strftime("%Y%m%d"), 
                                end = (ycdate - datetime.timedelta(days=periods-1)).strftime("%Y%m%d"))
        with st.sidebar:
            with st.spinner('Model Refitting...'):
                predictor.fit()
                preddata = predictor.forecast(periods=periods).tail(1)

    if pred == 'Lagged LSCT':
        lambdas = st.sidebar.slider('Decomposition Kernel ($\lambda$)', 0.0, 1.0, 0.498, disabled = not showpred)
        with st.sidebar:
            with st.spinner('Model Refitting...'):
                YieldPredictor = createYieldPredictor('Lagged LSCT')
                predictor = YieldPredictor(lambdas = lambdas,
                                           start = (ycdate - datetime.timedelta(days=periods+7)).strftime("%Y%m%d"), 
                                           end = (ycdate - datetime.timedelta(days=periods-1)).strftime("%Y%m%d"))
                predictor.fit()
                preddata = predictor.forecast(periods=periods).tail(1)

    if pred == 'Linear LSCT':
        lambdas = st.sidebar.slider('Decomposition Kernel ($\lambda$)', 0.0, 1.0, 0.498, disabled = not showpred)
        days = st.sidebar.select_slider('Training Window (Days)', range(1,1+252*10), 252, disabled = not showpred)
        lags = st.sidebar.select_slider('Lags (Days)', range(1, 11), 5, disabled = not showpred)
        logTrans = st.sidebar.toggle('Log Transform', value = False, disabled = not showpred)
        with st.sidebar:
            with st.spinner('Model Refitting...'):
                YieldPredictor = createYieldPredictor('Linear LSCT')
                predictor = YieldPredictor(lambdas = lambdas,
                                           lags = lags,
                                           log = logTrans,
                                           start = (ycdate - datetime.timedelta(days=periods+7+days)).strftime("%Y%m%d"), 
                                           end = (ycdate - datetime.timedelta(days=periods-1)).strftime("%Y%m%d"))
                predictor.fit()
                preddata = predictor.forecast(periods=periods).tail(1)

    if pred == 'Linear Model':
        lambdas = st.sidebar.slider('Decomposition Kernel ($\lambda$)', 0.0, 1.0, 0.498, disabled = not showpred)
        days = st.sidebar.select_slider('Training Window (Days)', range(1,1+252*10), 252, disabled = not showpred)
        lags = st.sidebar.select_slider('Lags (Days)', range(1, 11), 5, disabled = not showpred)
        logTrans = st.sidebar.toggle('Log Transform', value = False, disabled = not showpred)
        
        Xlist = st.sidebar.multiselect('Input Variables', ['LSCT'], default = ['LSCT'], disabled = not showpred)
        kernel = st.sidebar.selectbox('Regression Kernel', ['Linear', 'LASSO', 'Ridge'], disabled = not showpred)
        alpha = st.sidebar.slider('Panelty ($log \\alpha$)', -5.0, 5.0, 0.0, disabled = (not showpred) or (kernel == 'Linear'))
        alpha = 10.0 ** alpha
        with st.sidebar:
            with st.spinner('Model Refitting...'):
                YieldPredictor = createYieldPredictor('Linear Model')
                predictor = YieldPredictor(Xlist = Xlist,
                                           lambdas = lambdas,
                                           lags = lags,
                                           log = logTrans,
                                           start = (ycdate - datetime.timedelta(days=periods+7+days)).strftime("%Y%m%d"), 
                                           end = (ycdate - datetime.timedelta(days=periods-1)).strftime("%Y%m%d"))
                predictor.fit(kernel = kernel, alpha = alpha)
                preddata = predictor.forecast(periods=periods).tail(1)
    st.sidebar.divider()
    st.sidebar.markdown('''Created by Zhenning Zhao :link:[Website](https://znzhao.github.io/)''')
    return preddata

def createDisplayer():
    st.title("Yield Curve Prediction")
    data = loadAllYC()
    mindate = min(data.index).to_pydatetime()
    maxdate = max(data.index).to_pydatetime()
    enddate = max(data.index).to_pydatetime()

    showpred = st.sidebar.toggle('Show Predictions', value = True)
    periods = st.select_slider('Forecast Period', [1, 2, 3, 4, 5], 1)
    ycdate = st.slider('Yield Curve Date', mindate, maxdate, enddate)
    
    truedata = data[data.index<=ycdate]
    truedata = truedata.tail(1)

    truedata.columns = maturities
    truedata = truedata.T
    truedata.columns = ['True Data']

    try:
        preddata = createSideBar(showpred, ycdate, periods)
        preddata.columns = maturities
        preddata = preddata.T
        preddata.columns = ['Predicted Data']
    except Exception as error:
        print(error)
        showpred = False
        preddata = copy.deepcopy(truedata).applymap(lambda x: np.nan)

    if showpred:
        dispdata = pd.merge(truedata, preddata, left_index=True, right_index=True)
    else:
        dispdata = truedata
    colorlist = ['#2196F3', '#F44336', ]
    fig = px.line(
        dispdata,
        markers='o',
        color_discrete_sequence = colorlist[:len(dispdata.columns)],
        labels={
            "value": "Percent (%)",
            "index": "Maturities"
                 }
        )
    fig.update_layout(showlegend=showpred)
    fig.update_layout(legend=dict(title = None, orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x = 0.5))
    st.plotly_chart(fig)
    st.markdown('''
                ##### Reference
                - Diebold, Francis X., and Canlin Li. "Forecasting the Term Structure of Government Bond Yields." Journal of Econometrics :link:[link](https://www.sciencedirect.com/science/article/abs/pii/S0304407605000795)
                - Xuyang Ma, "The Twist Factor of Yields", Advances in Economics and Business :link:[link](https://www.hrpub.org/journals/article_info.php?aid=6167)
                ''')
    
    
if __name__ == "__main__":
    createDisplayer()