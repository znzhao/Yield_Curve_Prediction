import datetime
import copy
import numpy as np
import streamlit as st
import plotly.express as px
import pandas as pd
from helper.utils import nextKBusinessDay
from helper.lsctsplitter import loadAllYC
from helper.ycpredictor import createYieldPredictor
from pages.Yield_Curve
def createDisplayer():
    pass