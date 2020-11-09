import pandas as pd
import numpy as np
import datetime, math
from matplotlib import style
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from openpyxl import load_workbook
style.use('ggplot')
df_reversed = pd.read_excel (r'FENER1.xlsx', sheet_name='Worksheet')

df = df_reversed[::-1]
"""for i in df['Hac.']:
    i_temp=i
    if str(i).endswith('K'):
        i=float(str(i)[:-1])*1000.0
        df['Hac.'].loc[df['Hac.'].values==i_temp] =i
    elif str(i).endswith('M'):
        i = float(str(i)[:-1]) * 1000000.0
        df['Hac.'].loc[df['Hac.'].values == i_temp] = i"""
"""df['Hac.'] = (df['Hac.'].str.split()).apply(lambda x: float(x[0].replace(',', '.')))
df['Hac.'].astype(float).str.replace(',', '.')"""

