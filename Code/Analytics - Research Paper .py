
# coding: utf-8

# In[296]:


import datetime  
import numpy as np  
import pandas as pd
import numpy.random as random
#import statsmodels.api as sm
import statsmodels.tsa as tsa
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6


# In[297]:


xl = pd.ExcelFile("C:/Users/jayant.malik/Desktop/MMMProject_Data.xlsx")
xl.sheet_names
[u'modelingData']
df = xl.parse("modelingData")
df.head()


# In[ ]:


print(df.dtypes)


# In[369]:


list(df)


# In[301]:


df.columns = ['geography','period','sales','total_spend','fathers_day','mothers_day','thanksgiving_day','display','spot_tv','email','events','local_mag','machine_unit','national_mag','newspaper','online_video','OOH','paid_search','paid_social','national_tv','august_dummy']


# In[376]:


df.describe(percentiles=[])


# In[382]:


def correlation_matrix(df):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Correlation')
    labels=[]
    ax1.set_xticklabels(labels,fontsize=6)
    ax1.set_yticklabels(labels,fontsize=6)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
    plt.show()


# In[383]:


correlation_matrix(df)


# In[384]:


df.corr()


# In[367]:


plt.plot(df['period'],df['sales'])


# In[368]:


plt.plot(df['period'],df['national_tv'])


# In[303]:


modelfit1 = smf.ols(formula='sales ~ total_spend + display + spot_tv + email + events + local_mag + machine_unit + national_mag + newspaper + online_video + OOH + paid_search + paid_social + national_tv + august_dummy + fathers_day + mothers_day + thanksgiving_day',data=df).fit()


# In[304]:


print(modelfit1.summary())


# In[305]:


# Multivariate regression model with significant variables in the data
modelfit2 = smf.ols(formula='sales ~ display + machine_unit + national_mag + newspaper + online_video + OOH + paid_search + paid_social + national_tv',data=test).fit()


# In[306]:


print(modelfit2.summary())


# In[313]:


ts = df.groupby(['period'])['sales'].sum()
ts.head(10)


# In[314]:


plt.plot(ts)


# In[327]:


from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)  
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


# In[329]:


test_stationarity(ts)


# In[330]:


ts_log = np.log(ts)
plt.plot(ts_log)


# In[332]:


#Moving average
moving_avg = ts_log.rolling(12).mean()
plt.plot(ts_log)
plt.plot(moving_avg, color='red')


# In[333]:


ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.head(12)


# In[334]:


ts_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(ts_log_moving_avg_diff)


# In[343]:


expwighted_avg = ts_log.ewm(12).mean()
plt.plot(ts_log)
plt.plot(expwighted_avg, color='red')


# In[344]:


ts_log_ewma_diff = ts_log - expwighted_avg
test_stationarity(ts_log_ewma_diff)


# In[345]:


#Eliminating Trend and Seasonality
#Differencing – taking the differece with a particular time lag
#Decomposition – modeling both trend and seasonality and removing them from the model.
#1. Differencing
ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff)


# In[346]:


ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)


# In[347]:


#Decomposing
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()


# In[348]:


ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
test_stationarity(ts_log_decompose)


# In[349]:


#forecasting time series
#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf


# In[350]:


lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')


# In[351]:


#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')


# In[352]:


#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()


# In[353]:


from statsmodels.tsa.arima_model import ARIMA


# In[354]:


#AR Model
model = ARIMA(ts_log, order=(2, 1, 0))  
results_AR = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_diff)**2))


# In[355]:


#MA Model
model = ARIMA(ts_log, order=(0, 1, 2))  
results_MA = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-ts_log_diff)**2))


# In[356]:


#Combined Model
model = ARIMA(ts_log, order=(2, 1, 2))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))


# In[358]:


#taking it back to original scale
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head())


# In[360]:


predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()


# In[361]:


predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(ts)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))

