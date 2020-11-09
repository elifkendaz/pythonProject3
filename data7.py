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
from sklearn import preprocessing, svm
style.use('ggplot')


df_reversed= pd.read_csv('BJKAS.csv')
df =df_reversed[::-1]

"""df = df.set_index('Tarih')"""
df.drop('Fark %', axis=1, inplace=True)

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



df.plot(x='Tarih', y=['Yüksek','Hac.'])
df.plot(x='Tarih', y='Yüksek')
plt.show()
