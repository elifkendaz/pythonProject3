from typing import Any, Union

import pandas as pd
import numpy as np
import datetime, math
from matplotlib import style
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
from pandas.core.arrays import ExtensionArray
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn import preprocessing
from sklearn import preprocessing, svm
style.use('ggplot')



df_reversed= pd.read_csv('BJKAS.csv')
df =df_reversed[::-1]
print(df.isnull().sum().sum())

df.drop('Fark %', axis=1, inplace=True)

"""for i in df['Tarih']:
    i_temp=i
    datetime_obj=datetime.datetime.strptime(i, "%b %d, %Y")
    df['Tarih'].loc[df['Tarih'].values == i_temp] = i"""

df = df.set_index('Tarih')
for i in df['Şimdi']:
    i_temp=i
    i=str(i).replace(',','.')
    i=float(i)
    df['Şimdi'].loc[df['Şimdi'].values == i_temp] = i

for i in df['Açılış']:
    i_temp=i
    i=str(i).replace(',','.')
    i=float(i)
    df['Açılış'].loc[df['Açılış'].values == i_temp] = i

for i in df['Yüksek']:
    i_temp=i
    i=str(i).replace(',','.')
    i=float(i)
    df['Yüksek'].loc[df['Yüksek'].values == i_temp] = i

for i in df['Düşük']:
    i_temp=i
    i=str(i).replace(',','.')
    i=float(i)
    df['Düşük'].loc[df['Düşük'].values == i_temp] = i

for i in df['Hac.']:
    i_temp=i
    if str(i).endswith('K'):
        i = str(i)[:-1]
        i=str(i).replace(',', '.')
        i=float(i)
        i=i*1000.0
        df['Hac.'].loc[df['Hac.'].values == i_temp] = i
    elif str(i).endswith('M'):
        i = str(i)[:-1]
        i = str(i).replace(',', '.')
        i = float(i)
        i = i * 1000000.0
        df['Hac.'].loc[df['Hac.'].values == i_temp] = i
"""dfreg[‘HL_PCT’] = (df[‘High’] — df[‘Low’]) / df[‘Close’] * 100.0
dfreg[‘PCT_change’] = (df[‘Close’] — df[‘Open’]) / df[‘Open’] * 100.0"""

"""df['HL_PCT']= (df['Yüksek'] — df['Düşük'] ) / df['Şimdi'] * 100.0
df['PCT_change'] = ( df['Şimdi'] — df['Açılış']) / df['Açılış'] *100.0"""

df['HL_PCT']= (df['Yüksek'] -df['Düşük']) / df['Şimdi']*100.0
df['PCT_change'] = (df['Şimdi']-df['Açılış']) / df['Açılış']*100.0
df=df[['Şimdi','Hac.','HL_PCT','PCT_change']]


forecast_col='Şimdi'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.001 * len(df)))

df['Tahmin']=df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

X=np.array(df.drop(['Tahmin'],1))
X = X[:-forecast_out]
X= preprocessing.scale(X)
X_lately=X[-forecast_out:]


df.dropna(inplace=True)
y=np.array(df['Tahmin'])
y=np.array(df['Tahmin'])

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=0)


regressor = LinearRegression(n_jobs=-1)
regressor.fit(X_train, y_train)
Accuracy=regressor.score(X_test,y_test)

print('Accuracy:', Accuracy)

forecast_set=regressor.predict(X_lately)
df['Tahmin']=np.nan

last_date=df.iloc[-1].name
lastDatetime= last_date.timestamp()
one_day=86400
nextDatetime=lastDatetime+one_day
print("elielfo")

for i in forecast_set:
    next_date=datetime.datetime.fromtimestamp(nextDatetime)
    nextDatetime+= one_day
    df.loc[next_date]=[np.nan for x in range(len(df.columns)-1)]+[i]


df['Şimdi'].plot(color='r')
df['Tahmin'].plot(color='b')
plt.legend(loc=4)
plt.xlabel('Tarih')
plt.ylabel('Tahmin')
plt.title('BJK')
plt.show()


print("lol")



"""for i in df['Fark %']:
           i_temp1 = i
           i = str(i)[:-1]
           i = str(i).replace(',', '.')
           i = float(i)
           i = i * 0.01
           df['Fark %'].loc[df['Fark %'].values == i_temp1] = i"""















