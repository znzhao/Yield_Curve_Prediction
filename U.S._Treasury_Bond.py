import datetime
import numpy as np
import streamlit as st
import plotly.express as px
import pandas as pd
from helper.utils import interpolateColor
from helper.lsctsplitter import loadAllYC
from helper.dataparser import parseMonthlyYieldCurve, parseBreakEvenInflation, parseSP500
from helper.sensitive_utils import getLog, pushToGithub, logGithub


maturities = ['1 Month', '2 Month', '3 Month', '4 Month', '6 Month', '1 Year', '2 Year', '3 Year', '5 Year', '7 Year', '10 Year', '20 Year', '30 Year',]

def updateGit():
    logdate = getLog()
    startyyyy = logdate.year
    startmm = logdate.month
    endyyyy = datetime.datetime.now().year
    endmm = datetime.datetime.now().month
    if logdate.date() <= datetime.datetime.now().date():
        # update the yield curve data
        ycg = loadAllYC()
        for yyyy in range(startyyyy, endyyyy+1):
            if yyyy == startyyyy:
                beginmm = startmm
                stopmm = 13
            if yyyy == endyyyy:
                beginmm = 1
                stopmm = endmm
            if startyyyy == endyyyy:
                beginmm = startmm
                stopmm = endmm

            for mm in range(beginmm, stopmm+1):
                mm = str(mm) if len(str(mm)) == 2 else '0'+ str(mm)
                ycm = parseMonthlyYieldCurve(yyyymm = str(yyyy) + str(mm))
                ycg = pd.concat([ycg, ycm], axis = 0)
                print(ycm)
        ycg = ycg.drop_duplicates()
        pushToGithub('MktData/ycg/ycg.csv', ycg.to_csv(index=True, header=True))
        # update the tyie data
        parseBreakEvenInflation(git = True)
        parseSP500(git = True)
    logGithub(timestamp = max(ycg.index).to_pydatetime())

def createDisplayer():
    with st.spinner('Check Data Updates...'):
        updateGit()
    st.title("U.S. Treasury Bond Time Series")
    data = loadAllYC()
    mindate = min(data.index).to_pydatetime()
    maxdate = max(data.index).to_pydatetime()
    startdate = max(data.index).to_pydatetime() - datetime.timedelta(weeks=52*10)
    middate = max(data.index).to_pydatetime() - datetime.timedelta(weeks=52*5)
    enddate = max(data.index).to_pydatetime()
    
    DateRange = st.slider('Yield Curve Date Range', mindate, maxdate, (startdate, enddate))

    st.sidebar.markdown('## Maturities')
    columns = []
    for m in maturities:
        columns.append(st.sidebar.checkbox(m, value = True))
    
    st.sidebar.divider()
    st.sidebar.markdown('''Created by Zhenning Zhao :link:[Website](https://znzhao.github.io/)''')
    
    dispdata = data[(data.index >= DateRange[0]) & (data.index <= DateRange[1])]
    dispdata.columns = maturities
    dispcol = [dispdata.columns[i] for i, x in enumerate(columns) if x]
    colorlist = [interpolateColor('#FFC0C0','#E84444', i) for i in np.linspace(0,1,sum(columns))]
    fig = px.line(
        dispdata, 
        y = dispcol,
        color_discrete_sequence = colorlist,
        labels={
            "value": "Percent (%)",
            "date": "Date",
            "variable": "Maturities"
                 }
        )
    st.plotly_chart(fig)
    st.markdown('''
                ##### Data Source
                - U.S. Department of the Treasury :link: [link](https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics)
                ''')
                
if __name__ == "__main__":
    createDisplayer()