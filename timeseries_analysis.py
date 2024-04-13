#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import scipy.stats
import pylab
from statsmodels.tsa.stattools import adfuller
import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARMA
from scipy.stats.distributions import chi2
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from math import sqrt
sns.set()
%matplotlib inline
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import warnings


# In[34]:


file=r'dblp_data.txt'


# Importing of the dataset, converting in dataframe and showing details about null values

# In[35]:


data=pd.read_csv(file, delimiter="\t",index_col="Years",parse_dates=True, header=0)
data = data.asfreq('AS')
data = data.shift(periods = 1, freq= 'A')
data


# In[4]:


data.isna().sum()


# Fill null values with the previous value

# In[5]:


data=data.fillna(method='ffill')


# In[6]:


data.index


# In[7]:


data=data[:'2020']


# Visualize the Number of Publications

# In[8]:


data.plot(label="Number of Publications")
plt.title('Number of Publications of Computational Linguistics', size=10)
plt.legend(loc='best')
plt.savefig('data.png')
plt.show()


# Check stationarity of the time serie

# In[9]:


#stationary test
stationary=sts.adfuller(data["Number of Publications"])
print("ADFuller Statistics: ", stationary[0])
print("P value: ", stationary[1])
print("Lags: ", stationary[2])
print("Critical values: ", stationary[4])


# In[10]:


# Determing rolling statistics
data['Number of Publications'].plot(figsize=(8,4))
holt_winters = ExponentialSmoothing(data, trend = 'add').fit().fittedvalues.plot(label="Holt Winters")
rollstd = data['Number of Publications'].rolling(window=4).std().plot(label='Rolling Std')
plt.title('Rolling Mean & Standard Deviation', size=10)
plt.legend(loc='best')


# Multiplicative decomposition of Trend, Seasonality and Residuals

# In[11]:


s_dec_multiplicative = seasonal_decompose(data['Number of Publications'], model = 'multiplicative',period=1)
s_dec_multiplicative.plot();


# Transformation of the time serie in logarithmic

# In[12]:


series_log = np.log(data['Number of Publications'])
plt.plot(series_log)
plt.show()


# Double exponational smoothing with holt winters

# In[13]:


holt_winters = ExponentialSmoothing(series_log, trend = 'add').fit().fittedvalues
holt_winters.plot(label="Holt Winters")


# Making the logarithmic time serie stationary and checking with stationary test (Dickey-Fuller test)

# In[14]:


series_log_stat = series_log - holt_winters
print (series_log_stat.head(15))


# In[15]:


series_log_stat.dropna(inplace=True)
print (series_log_stat)
series_log_stat.plot(label='series log',figsize=(8,4))
holt_winters = ExponentialSmoothing(series_log_stat, trend = 'add').fit().fittedvalues.plot(label="Holt Winters")
rollstd = series_log.rolling(window=4).std().plot(label='Rolling Std')
plt.title('stationary time serie & Standard Deviation', size=13)
plt.legend(loc='best')
plt.show()
plt.savefig('stationaryTimeSerie.png')


# In[16]:


#stationary test
stat_log=sts.adfuller(series_log_stat)
print("ADFuller Statistics: ", stat_log[0])
print("P value: ", stat_log[1])
print("Lags: ", stat_log[2])
print("Critical values: ", stat_log[4])


# In[17]:


decomposition = seasonal_decompose(series_log)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(series_log, label='timeSerie log')
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
plt.show()


# Plot the ACF and PACF to find the significant number of Lags

# In[18]:


sgt.plot_acf(series_log_stat,lags=40,zero=False)
plt.title("ACF", size=24)
plt.show()
sgt.plot_pacf(series_log_stat,lags=25,zero=False)
plt.title("PACF", size=24)
plt.show()


# Split the time serie set in train and test set

# In[19]:


train_holt_winters=series_log[:'2015']
test_holt_winters=series_log['2016':]


# Making predictions with Holt Winters model and evaluating it with errors

# In[20]:


model = ExponentialSmoothing(train_holt_winters, trend='add')
fitted = model.fit()


# In[21]:


print(fitted.summary())


# In[22]:


series_log_forecast = fitted.forecast(steps=5)


# In[23]:


fig = plt.figure()
fig.suptitle('....')
past, = plt.plot(train_holt_winters.index, train_holt_winters, 'b.-', label='Train')
future, = plt.plot(test_holt_winters.index, test_holt_winters, 'r.-', label='Test')
predicted_future, = plt.plot(test_holt_winters.index, series_log_forecast, 'g.-', label='Series log forecast')
plt.legend(handles=[past, future, predicted_future])
plt.show()


# In[24]:


print('test log data:\n' + str(test_holt_winters)+'\n')

print('predicted log data:\n' + str(series_log_forecast))


# In[25]:


rms = sqrt(mean_squared_error(test_holt_winters,series_log_forecast))
print('RMSE: ' + str(rms))
MAE=mean_absolute_error(test_holt_winters,series_log_forecast)
print('MAE: ' +str(MAE))
MSE = np.square(np.subtract(test_holt_winters,series_log_forecast)).mean()
print('MSE: ' + str(MSE))


# Making predictions with ARIMA model and evaluating it with errors

# In[26]:


# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train, test = X[:'2015'], X['2016':]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error


# In[27]:


# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order, mse))
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))


# In[28]:


# evaluate parameters
p_values = [0,1,2,3,4,5,6,7,8]
d_values = [0,1,2,3]
q_values = [0,1,2,3,4,5]
warnings.filterwarnings("ignore")
evaluate_models(series_log, p_values, d_values, q_values)


# In[51]:


rms = sqrt(mean_squared_error(test, predictions))
print('RMSE: ' + str(rms))
MAE=mean_absolute_error(test, predictions)
print('MAE: ' +str(MAE))
MSE = round(mean_squared_error(test, predictions),3)
print('MSE: ' + str(MSE))


# In[48]:


predictions


# In[49]:


model = ARIMA(series_log, order=(1, 1, 0))
results_ARIMA = model.fit(disp=0)
series_log_stat.plot(figsize=(8,4), label="Time serie log")
results_ARIMA.fittedvalues.plot(label="Results Arima")
plt.title('ARIMA(1,1,0)', size=10)
plt.legend(loc='best')
plt.savefig('arima.png')


# In[59]:


predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_log = pd.Series(series_log, index=series_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff,fill_value=0)
predictions_ARIMA_log.head()


# In[36]:


test=series_log['2016':]
train=series_log[:'2015']

test


# In[72]:


predictions_ARIMA_log.plot(figsize=(8,4), label="ARIMA")
test.plot(figsize=(8,4), label="Test")
train.plot(figsize=(8,4), label="Train")
plt.title('Prediction of ARIMA model', size=10)
plt.legend(loc='best')
plt.savefig('predictionArima.png')
plt.show()


# Converting the ARIMA values to non-logarithmic

# In[55]:


predictions_ARIMA = np.exp(predictions_ARIMA_log)
print(predictions_ARIMA.tail(5))
print(data.tail(5))


# In[75]:


data.plot(figsize=(8,4), label="Number of Publications")
predictions_ARIMA.plot(figsize=(8,4), label="Prediction")
plt.title('Number of Publications & Prediction', size=10)
plt.legend(loc='best')
plt.savefig('predictionPubs.png')
plt.show()


# Making forecast about the trend of the number of publications untill 2030

# In[81]:


results_ARIMA.plot_predict(1,78);

