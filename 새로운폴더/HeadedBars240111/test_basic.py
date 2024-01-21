import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import torch.optim as optim
import os
import openpyxl
from torch.utils.data import TensorDataset, DataLoader,random_split
import math
#import tensorflow as tf

df = pd.read_excel("Data for headed bars_for DataFrame_220725.xlsx", skiprows = 17, engine = 'openpyxl', sheet_name= 'headed (2)' )
df = pd.DataFrame(df, columns = ["No.", "Author", "Year", "Test type", "Remark", "Specimen", "fy", "Ld", "fcm", "db", "b", "cos,avg",
                                 "cth", "ch", "Nh", "Bottom cover", "Ah/Ab", "Fsu at La, test", "dtr", "Ntr", "st"]) # st 제거시

df = df.dropna(subset=['Fsu at La, test'])
df = df[df['Ld'] != 0]

y= df["Fsu at La, test"] 

df['fcm_sqrt']= df['fcm'].apply(lambda x:math.sqrt(x))
df['origin_y'] = (df['Ld']*df['fcm_sqrt'])/(0.19*df['db'])
y_origin = df['origin_y']

from sklearn.metrics import r2_score, mean_squared_error
R2 = r2_score(y, y_origin)
mse = mean_squared_error(y,y_origin)

division =y / y_origin
cov = np.std(division) / np.mean(division)

print(f'cov:{cov}')
print(f'r2_score:{R2}')
print(f'mse:{mse}')
print(f'rmse:{np.sqrt(mse)}')
mape = np.mean(np.abs((y_origin - y)/y_origin))*100
mae = np.mean(np.abs(y_origin - y))
print(f'mape:{mape}')
print(f'mae:{mae}')

import time
now = time
new_data ={
    'NAME':["y vs origin_y"],  # 수정 후 작업 시 이름 변경 필요
    'cov' :[cov],
    'r2_score':[R2],
    'mse':[mse],
    'rmse':[np.sqrt(mse)],
    'mape':[mape],
    'mae' : [mae],
    'Time' :[now.strftime('%Y-%m-%d %H:%M:%S')]
}

existing_data = pd.read_csv('test.csv')
new_df = pd.DataFrame(new_data)
result_df = existing_data.append(new_df, ignore_index=True)
result_df.to_csv('test.csv', index=False)