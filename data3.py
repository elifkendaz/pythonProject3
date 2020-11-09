import pandas as pd
import numpy as np
import datetime, math
from matplotlib import style
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import calendar
from datetime import timedelta, time

style.use('ggplot')


def temizle(df):
    """df = df.set_index('Tarih')"""

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



def cevir(df):

    df['HL_PCT']= (df['Yüksek'] - df['Düşük']) / df['Şimdi'] * 100.0
    df['PCT_change'] = (df['Şimdi'] - df['Açılış']) / df['Açılış'] * 100.0
    df = df[['Şimdi', 'Hac.', 'HL_PCT', 'PCT_change']]
    forecast_out = int(math.ceil(0.01 * len(df)))
    df['Tahmin']=df['Şimdi'].shift(-forecast_out)


    X=df.drop(columns='Tahmin')
    y=df.iloc[:,-1]


    scaler= MinMaxScaler()
    scaler.fit(X)
    X_max=scaler.data_max_
    X_min=scaler.data_min_
    X=scaler.transform(X)

    X_toPredict = X[-forecast_out:]
    X = X[:-forecast_out]
    y = y[:-forecast_out]


    X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=0)


    regressor = LinearRegression()
    regressor.fit(X_train, y_train)


    Accuracy=regressor.score(X_test,y_test)

    print('Accuracy:', Accuracy)


    predset=regressor.predict(X_toPredict)
    df['Tahmin']=np.nan

    last_date=df.iloc[-1].name

    print("date",last_date)

    """lastDatetime=last_date.timestamp()"""
    lastDatetime = datetime.datetime.strptime(last_date, '%d.%m.%Y')
    """lastDatetime = datetime.timedelta(last_date)"""
    print("elielfo")
    one_day=86400
    nextDatetime=lastDatetime+timedelta(days=1)
    print("datecd",nextDatetime)
    print("elielfo")
    for i in predset:
        next_date =nextDatetime
        nextDatetime+= timedelta(days=1)
        df.loc[next_date]=[np.nan for x in range(len(df.columns)-1)]+[i]


    df['Şimdi'].plot(color='r')
    df['Tahmin'].plot(color='b')
    plt.legend(loc=4)
    plt.xlabel('Tarih')
    plt.ylabel('Tahmin')
    plt.title('BJK')
    plt.show()

df= pd.read_csv(r'C:\Users\Elif\Desktop\DataSets\BANKACILIK\AKBNK.csv')
df = df.set_index('Tarih')
temizle(df)
cevir(df)














