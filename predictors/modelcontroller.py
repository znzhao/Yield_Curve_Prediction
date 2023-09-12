import datetime
import streamlit as st
from helper.ycpredictor import createYieldPredictor
maturities = ['1 Month', '2 Month', '3 Month', '4 Month', '6 Month', '1 Year', '2 Year', '3 Year', '5 Year', '7 Year', '10 Year', '20 Year', '30 Year',]

def createSideBar(showpred = True):
    pred = st.sidebar.selectbox('Prediction Model', ['Random Walk', 'Lagged LSCT', 'Linear LSCT', 'Linear Model'], disabled = not showpred)
    params = {}
    if pred == 'Random Walk':
        pass
        params['model'] = 'Random Walk'
    if pred == 'Lagged LSCT':
        lambdas = st.sidebar.slider('Decomposition Kernel ($\lambda$)', 0.0, 1.0, 0.498, disabled = not showpred)
        params['model'] = 'Lagged LSCT'
        params['lambdas'] = lambdas
    if pred == 'Linear LSCT':
        lambdas = st.sidebar.slider('Decomposition Kernel ($\lambda$)', 0.0, 1.0, 0.498, disabled = not showpred)
        days = st.sidebar.select_slider('Training Window (Days)', range(1,1+252*10), 252, disabled = not showpred)
        lags = st.sidebar.select_slider('Lags (Days)', range(1, 11), 5, disabled = not showpred)
        logTrans = st.sidebar.toggle('Log Transform', value = False, disabled = not showpred)
        params['model'] = 'Linear LSCT'
        params['lambdas'] = lambdas
        params['days'] = days
        params['lags'] = lags
        params['logTrans'] = logTrans

    if pred == 'Linear Model':
        lambdas = st.sidebar.slider('Decomposition Kernel ($\lambda$)', 0.0, 1.0, 0.498, disabled = not showpred)
        days = st.sidebar.select_slider('Training Window (Days)', range(1,1+252*10), 252, disabled = not showpred)
        lags = st.sidebar.select_slider('Lags (Days)', range(1, 11), 5, disabled = not showpred)
        logTrans = st.sidebar.toggle('Log Transform', value = False, disabled = not showpred)
        Xlist = st.sidebar.multiselect('Input Variables', ['LSCT', 'Stock', 'Inflation'], default = ['LSCT'], disabled = not showpred)
        kernel = st.sidebar.selectbox('Regression Kernel', ['Linear', 'LASSO', 'Ridge'], disabled = not showpred)
        alpha = st.sidebar.slider('Panelty ($log \\alpha$)', -5.0, 5.0, 0.0, disabled = (not showpred) or (kernel == 'Linear'))
        alpha = 10.0 ** alpha
        
        params['model'] = 'Linear LSCT'
        params['lambdas'] = lambdas
        params['days'] = days
        params['lags'] = lags
        params['logTrans'] = logTrans
        params['Xlist'] = Xlist
        params['kernel'] = kernel
        params['alpha'] = alpha
    return params

def createModel(ycdate, periods, params):
    pred = params['model']
    if pred == 'Random Walk':
        YieldPredictor = createYieldPredictor('Baseline')
        predictor = YieldPredictor(start = (ycdate - datetime.timedelta(days=periods+7)).strftime("%Y%m%d"), 
                                end = (ycdate - datetime.timedelta(days=periods-1)).strftime("%Y%m%d"))
        with st.spinner('Model Refitting...'):
            predictor.fit()
            preddata = predictor.forecast(periods=periods).tail(1)

    if pred == 'Lagged LSCT':
        lambdas = params['lambdas']
        with st.spinner('Model Refitting...'):
            YieldPredictor = createYieldPredictor('Lagged LSCT')
            predictor = YieldPredictor(lambdas = lambdas,
                                       start = (ycdate - datetime.timedelta(days=periods+7)).strftime("%Y%m%d"), 
                                       end = (ycdate - datetime.timedelta(days=periods-1)).strftime("%Y%m%d"))
            predictor.fit()
            preddata = predictor.forecast(periods=periods).tail(1)

    if pred == 'Linear LSCT':
        lambdas = params['lambdas']
        days = params['days']
        lags = params['lags']
        logTrans = params['logTrans']
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
        lambdas = params['lambdas']
        days = params['days']
        lags = params['lags']
        logTrans = params['logTrans'] 
        Xlist = params['logTrans']  
        kernel = params['kernel']  
        alpha = params['kernel']

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
    return preddata
