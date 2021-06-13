import pandas as pd
import numpy as np
import datetime, math
from matplotlib import style
from seaborn import countplot, distplot, boxenplot
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
from pandas.core.arrays import ExtensionArray
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import calendar
from datetime import timedelta, time
from pandas.plotting import scatter_matrix


from sklearn import preprocessing, svm
style.use('ggplot')

x = datetime.datetime.now()
print(x)

df_reversed= pd.read_csv('AKBNK.csv')
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


"""df['Şimdi'].plot(color='r')
df['Tahmin'].plot(color='b')
plt.legend(loc=4)
plt.xlabel('Tarih')
plt.ylabel('Tahmin')
plt.title('Akbank')
countplot(x='Tahmin', data=df)
plt.scatter(x,y)
distplot(df['Tahmin'])
distplot(df['Hac.'])
df['Hac.'].plot(color='b')
plt.legend(loc=4)

plt.xlabel('Tarih')
plt.ylabel('Tahmin')
plt.title('Akbank Hacim')

df.plot(x = 'year', y = 'Tahmin')
df.plot(x = 'year', y = 'Hac.', secondary_y = True)
plt.xlabel('Veriler')
plt.ylabel('Tahmin')
plt.title('Akbank')
data =df['Şimdi']
plt.bar(range(len(data)), data)
plt.show()
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df.plot(kind='density', subplots=True, layout=(5,1), sharex=False)
df1=df['Şimdi']
scatter_matrix(df1)
pd.plotting.scatter_matrix(df, diagonal='kde')
plt.show()"""
df0 = df[df['Şimdi'].between(0.0, 5.0)]
df00=print(len(df0))
df1 = df[df['Şimdi'].between(5.0, 6.0)]
df11=print(len(df1))
df2 = df[df['Şimdi'].between(6.0, 7.0)]
df22=print(len(df2))
df3 = df[df['Şimdi'].between(7.0, 8.0)]
df33=print(len(df3))
df4 = df[df['Şimdi'].between(8.0, 9.0)]
df44=print(len(df4))
labels = ['Less than5.0 tl', 'Between 5.0 and 6.0 tl', 'Between 6.0 and 7.0 tl ',
        'Between 7.0 and 8.0 tl', 'More than 8.0 tl']

#data = [39, 303, 661, 949, 526]
data = [df00, df11, df22, df33, df44]
# Creating plot
fig = plt.figure(figsize=(10, 7))
plt.title('Pie Chart of Akbank Stock Prices')
plt.pie(data, labels=labels)
plt.legend()
plt.show()

"""plt.xlabel('Days of between 23.10.2010 - 23.10.2020')
plt.ylabel('Stock Price')
plt.title('Akbank')
data =df['Şimdi']
plt.bar(range(len(data)), data)
plt.show()


df['Tahmin'].dropna()

df.dropna(subset=['Tahmin'],inplace=True)
print(df['Tahmin'])

df1= pd.DataFrame(df, columns= ['Tahmin'])
df1.to_csv('akb.csv')"""