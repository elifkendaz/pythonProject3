import pandas as pd
import numpy as np
import datetime, math
from matplotlib import style
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



style.use('ggplot')

df_reversed= pd.read_csv('BJKAS.csv')
df =df_reversed[::-1]




"""files=sorted(glob(r'C:\Users\Elif\Desktop\DataSets\BANKACILIK\*.csv'))"""

"""extension = 'csv'
all_filenames = [i for i in glob.glob('C:\Users\Elif\Desktop\DataSets\BANKACILIK\*.{}'.format(extension))]
#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
#export to csv
combined_csv.to_csv( "combined_csv.csv", index=False)"""

