#%%
import numpy as np
import plotly as py
import plotly.express as px


#%%
# Basics

arr_1 = np.random.rand(50,4)
print(arr_1)
px.line_3d(arr_1)


#%%
# Line Plots
import plotly.graph_objects as go
df_stocks = px.data.stocks()
px.line(df_stocks, x='date', y='GOOG', labels={'x': 'Date', 'y': 'Price'})

px.line(df_stocks, x='date', y=['GOOG', 'AAPL'], title='Apple vs. Google')