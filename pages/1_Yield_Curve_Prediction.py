import datetime
import copy
import numpy as np
import streamlit as st
import plotly.express as px
import pandas as pd
from helper.utils import nextKBusinessDay
from helper.lsctsplitter import loadAllYC
from helper.ycpredictor import createYieldPredictor
from predictors.modelcontroller import *

def createDisplayer():
    st.title("Yield Curve Prediction")
    data = loadAllYC()
    mindate = min(data.index).to_pydatetime()
    maxdate = max(data.index).to_pydatetime()

    periods = st.select_slider('Forecast Period', [1, 2, 3, 4, 5], 1)
    ycdate = st.slider('Yield Curve Date', mindate, nextKBusinessDay(maxdate, periods), nextKBusinessDay(maxdate, periods))
    showpred = st.sidebar.toggle('Show Predictions', value = True, disabled = ycdate > maxdate)
    showpred = showpred if ycdate <= maxdate else True

    truedata = data[data.index<=ycdate]
    truedata = truedata.tail(1)
    truedata.columns = maturities
    truedata = truedata.T
    truedata.columns = ['True Data']
    if ycdate > maxdate:
        truedata = truedata.applymap(lambda x: np.nan)
    try:
        params = createSideBar(showpred)
        preddata = createModel(ycdate, periods, params)
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
    colorlist = ['#183D3D', '#5C8374', ]
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
    
    st.sidebar.divider()
    st.sidebar.markdown('''Created by Zhenning Zhao :link:[Website](https://sites.google.com/utexas.edu/zhenning-zhao/about)''')
    
if __name__ == "__main__":
    createDisplayer()