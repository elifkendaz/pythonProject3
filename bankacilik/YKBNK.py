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
import calendar
from datetime import timedelta, time


from sklearn import preprocessing, svm
style.use('ggplot')

x = datetime.datetime.now()
print(x)

df_reversed= pd.read_csv(r'C:\Users\Elif\Desktop\DataSets\BANKACILIK\YKBNK.csv')
df =df_reversed[::-1]
print(df.isnull().sum().sum())

df.drop('Fark %', axis=1, inplace=True)

for i in df['Tarih']:
    i_temp=i
    datetime_obj=datetime.datetime.strptime(i, '%d.%m.%Y')
    df['Tarih'].loc[df['Tarih'].values == i_temp] = i


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


df['HL_PCT']= (df['Yüksek'] -df['Düşük']) / df['Şimdi']*100.0
df['PCT_change'] = (df['Şimdi']-df['Açılış']) / df['Açılış']*100.0
df=df[['Şimdi','Hac.','HL_PCT','PCT_change']]



forecast_out = int(math.ceil(0.005 * len(df)))
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


lastDatetime = datetime.datetime.strptime(last_date, '%d.%m.%Y')

print("elielfo")
one_day=86400
nextDatetime=lastDatetime+timedelta(days=1)



for i in predset:
    next_date =nextDatetime
    nextDatetime+= timedelta(days=1)
    df.loc[next_date]=[np.nan for x in range(len(df.columns)-1)]+[i]


df['Şimdi'].plot(color='r')
df['Tahmin'].plot(color='b')
plt.legend(loc=4)
plt.xlabel('Tarih')
plt.ylabel('Tahmin')
plt.title('Yapı Kredi')
plt.show()
df['Tahmin'].dropna()

df.dropna(subset=['Tahmin'],inplace=True)
print(df['Tahmin'])

df1= pd.DataFrame(df, columns= ['Tahmin'])
df1.to_csv('ykbnk.csv')
