import pandas as pd
import numpy as np
import datetime, math
from matplotlib import style
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
from pandas.core.arrays import ExtensionArray
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv('BJKAS.csv')

def temizle(df):
    df = df.set_index('Tarih')
    df.drop('Fark %', axis=1, inplace=True)

    df = df[::-1]

    for i in df['Şimdi']:
        i_temp = i
        i = str(i).replace(',', '.')
        i = float(i)
        df['Şimdi'].loc[df['Şimdi'].values == i_temp] = i

    for i in df['Açılış']:
        i_temp = i
        i = str(i).replace(',', '.')
        i = float(i)
        df['Açılış'].loc[df['Açılış'].values == i_temp] = i

    for i in df['Yüksek']:
        i_temp = i
        i = str(i).replace(',', '.')
        i = float(i)
        df['Yüksek'].loc[df['Yüksek'].values == i_temp] = i

    for i in df['Düşük']:
        i_temp = i
        i = str(i).replace(',', '.')
        i = float(i)
        df['Düşük'].loc[df['Düşük'].values == i_temp] = i

    for i in df['Hac.']:
        i_temp = i
        if str(i).endswith('K'):
            i = str(i)[:-1]
            i = str(i).replace(',', '.')
            i = float(i)
            i = i * 1000.0
            df['Hac.'].loc[df['Hac.'].values == i_temp] = i
        elif str(i).endswith('M'):
            i = str(i)[:-1]
            i = str(i).replace(',', '.')
            i = float(i)
            i = i * 1000000.0
            df['Hac.'].loc[df['Hac.'].values == i_temp] = i

    if df.isnull().sum().sum() != 0:
        df = df.dropna()

    return df

temizle(df)
