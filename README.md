# Yield Curve Predictor
Created by Zhenning Zhao, Contact the creator at [znzhao@utexas.edu](mailto:znzhao@utexas.edu).

Visit the website [link](https://yield-curve-prediction.streamlit.app/) to run the yield curve predictor. Please sign in with a streamlit cloud account to visit the website. Streamlit cloud service is free.

The yield curve predictor cannot only plot the US treasury bond yield curve but also use customized models to predict the yield curve and run back tests. The current supported model includes:

1. Random Walk Model: Assume the yield curve is a random walk and use the past yield curve to directly predict the future yield curve. The random walk model also work as the baseline for all the other models.

2. Lagged LSCT Model: Decompose the yield curve into level, slope, curvature and twist factors and use lagged factors to predict the future yield curve.

3. Linear LSCT Model: Basic linear model using only lagged LSCT factors. The linear LSCT model work as a second baseline for more complicated models.

4. Linear Model: Linear models with panelties. More factors can be added than the basic linear LSCT models.