import pandas as pd
from pandas_datareader import data
import numpy as np
import matplotlib.pyplot as plt
from warnings import filterwarnings
from cycler import cycler

plt.style.use(['dark_background', 'fivethirtyeight'])
plt.rcParams['axes.facecolor'] = '#121212'
plt.rcParams['figure.facecolor'] = '#121212'
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['figure.autolayout'] = True ## This makes sure that the subplots are sufficiently far apart
plt.rcParams['axes.prop_cycle'] = cycler('color',['#bb86fc', '#c7fc86', '#ff7697', '#ffffff'])
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['boxplot.boxprops.linewidth'] = 2 
plt.rcParams['boxplot.flierprops.markeredgecolor'] = 'white'

filterwarnings('ignore')



try:
    tickers = pd.read_csv('tickers_sp100.csv')
except FileNotFoundError:    
    website_data = pd.read_html('https://en.wikipedia.org/wiki/Sget_ipython().run_line_magic("26P_100')", "")
    tickers = website_data[2].Symbol # this is the ticker table
    tickers.to_csv('tickers_sp100.csv', index=False)
    
tickers = tickers.Symbol.to_list()


start_date = '2015-05-31'
end_date = '2020-05-31'

try:
    raw_data = pd.read_pickle('sp100_adj_close.pkl')
except FileNotFoundError:
    raw_data = pd.DataFrame()
    
    print('Downloading started:')
    print('-'*50)
    
    for i, ticker in enumerate(tickers):
        try:
            col = data.DataReader(ticker, 'yahoo', start_date, end_date)['Adj Close']
            raw_data[ticker] = col
        except:
            print(f'Could not retrieve the data for {ticker}')
      
    print('Downloading completeget_ipython().getoutput("')")
    print('-'*50)
    
    raw_data.to_pickle('sp100_adj_close.pkl')
        


raw_data.columns[raw_data.isna().any()]


data = raw_data[raw_data.columns[raw_data.isna().any() == False]]
data.isna().sum().sum()


returns = data.diff().dropna()
returns.to_pickle('sp100_clean_returns.pkl')
returns.shape


fig, axs = plt.subplots(3,3, figsize = (15,15))
axs = axs.ravel()

for i, elt in enumerate(returns.columns[:9]):
    returns[elt].plot(ax=axs[i], title = elt)
    axs[i].set_xlabel('')
    


import seaborn as sns

from sklearn import preprocessing
from sklearn.decomposition import PCA 


returns.corr().describe()


standard_returns = pd.DataFrame(preprocessing.scale(returns), index = returns.index, columns = returns.columns)


fig, axs = plt.subplots(3,3, figsize = (15,15))
axs = axs.ravel()

for i, elt in enumerate(standard_returns.columns[:9]):
    standard_returns[elt].plot(ax=axs[i], title = elt)
    axs[i].set_xlabel('')
    


pca_model = PCA(n_components=20)
pca_model.fit(standard_returns)


plt.bar(range(20),pca_model.explained_variance_ratio_)
plt.title('Variance explained')


reduced_returns = pd.DataFrame(pca_model.transform(standard_returns), index=standard_returns.index)


fig, axs = plt.subplots(5,1, figsize = (15,15))
axs = axs.ravel()

for i in range(5):
    reduced_returns[i].plot(ax=axs[i])
    axs[i].set_title(f'Singular value number {i}')


import denoising_detoning as dd


x = dd.mpPDF(1, 1258/97, 1000)


x.plot()


eVal, eVec = dd.getPCA(standard_returns.corr())


pdf1 = dd.fitKDE(np.diag(eVal))


pdf1.plot()


fig, ax = plt.subplots(1,1,figsize=(12,12))
pdf1.plot(ax=ax)
x.plot(ax=ax)






