#!/usr/bin/env python
# coding: utf-8

# In[2]:


from tensorflow.python.client import device_lib   # 이는 나중에 GPU를 사용하고자 할때 필요함.
print(device_lib.list_local_devices())


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns


# In[4]:


import tensorflow as tf
tf.__version__


# In[5]:


import os
currentpath = os.getcwd()
print(currentpath)
newpath = os.chdir(currentpath + '')
print(newpath)


# In[6]:


import pandas as pd

df = pd.read_excel("Data for headed bars_for DataFrame_220725.xlsx", skiprows = 17, engine = 'openpyxl')
df = pd.DataFrame(df, columns = ["No.", "Author", "Year", "Test type", "Remark", "Specimen", "fy", "Ld", "fcm", "db", "b", "cos,avg",
                                 "cth", "ch", "Nh", "Bottom cover", "Ah/Ab", "Fsu at La, test", "dtr", "Ntr", "st"]) # st 제거시
df


# In[ ]:


df = df[df["Test type"] == "Joint type"]  # 실험방법이 Joint type일 경우: 성능이 크게 개선됨.

df


# In[ ]:


# original_Fsu =  df["Fsu at La, test"]
# df["Fsu at La, test"] = np.log1p(df["Fsu at La, test"]) # 스케일링을 한다면 굳이 로그 함수를 사용하지 않아도 됩니다.


# In[ ]:


y= df["Fsu at La, test"]
y


# In[ ]:


df.info()   # dtr, Ntr 데이터가 상대적으로 모자라니, 이 둘을 feature에서 제거함.


# In[ ]:


#X = df[["fy", "Ld", "fcm", "db", "b", "cos,avg", "cth", "ch", "Nh", "Bottom cover", "Ah/Ab", "dtr", "Ntr", "st"]] # 최대변수 사용
#X = df[["fy", "Ld", "fcm", "db", "b", "cos,avg", "cth", "ch", "Nh", "Bottom cover", "Ah/Ab", "st"]] # 누락값 많은 변수 미사용
# Test type - One-hot encoding
X = df[["Test type", "fy", "Ld", "fcm", "db", "b", "cos,avg", "cth", "ch", "Nh", "Bottom cover", "Ah/Ab", "st"]] # 범주형 데이터: Test type 포함
print(X["Test type"].value_counts()) # 범주 갯수 확인


# In[ ]:


X = pd.get_dummies(data = X, columns = ["Test type"], prefix = "Test_type") # One-hot Endcoding 실행 -> (0, 0, 0, 1, 0), (1, 0, 0, 0, 0), ...
X


# In[ ]:


y2 = y[~y.isnull()]
y2


# In[ ]:


X = X[~y.isnull()]
X


# In[ ]:


# X.fillna(0, inplace = True)   # 빈 데이터를을 모두 0으로 채우는 것은 잘못된 결과를 주어서 위험합니다. 차라리 다음과 같이 데이터를 없애는 게 낫습니다.
pd.options.display.max_rows=None

X.dropna(inplace = True)
X


# In[ ]:


y2 = y2[X.index] # 목표값도 X와 동일하게 indexing
y2


# In[ ]:


# y2 = 0인 값을 제거. 
y2 = y2.loc[(y2 != 0)]
y2


# In[ ]:


X = X.loc[y2.index] # y=0인 index의 X값도 제거
X


# In[ ]:


##학습과 실험 데이터를 분류하고 train과 valid 데이터를 분류한뒤


# In[ ]:


# Train, Valid, Test Set으로 분류


# In[ ]:


# 1. Train + Valid : Test  = 0.9 : 0.1 --> 먼저 9:1로 나누고


# In[ ]:


from sklearn.model_selection import train_test_split

tf.random.set_seed(142)

X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.20, random_state=142)

X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:


# 2. Train : Valid = 8: 2  --> 9중 20%를 valid로 가져옴


# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.20,random_state=142)

X_train.shape, X_valid.shape, y_train.shape, y_valid.shape


# In[ ]:


## X변수들을 MinMaxScaler로 스케일링을 진행하였습니다.
## --> Y도 스케일링 하여야 합니다.


# In[ ]:





# In[ ]:


print(type(X_train))
print(type(X_valid))
print(type(X_test))
print(type(y_train))
print(type(y_valid))
print(type(y_test))


# In[ ]:


from sklearn.preprocessing import MinMaxScaler

scX = MinMaxScaler()                    # X의 scaler 정의
X_train_scaled = scX.fit_transform(X_train)    
X_valid_scaled = scX.transform(X_valid)
X_test_scaled = scX.transform(X_test)

scY = MinMaxScaler()                    # y의 scaler 정의
y_train_scaled = scY.fit_transform(y_train.values.reshape(-1,1)) # y_train.values.reshape(-1,1) 는 Pandas Series를 조작하여 차원조절
y_valid_scaled = scY.transform(y_valid.values.reshape(-1,1))
y_test_scaled = scY.transform(y_test.values.reshape(-1,1))


# In[ ]:


X_train


# In[ ]:


sns.pairplot(data=X_train)


# In[ ]:


sns.distplot(y_train_scaled)


# In[ ]:


##층을 여기서 더 추가하거나 하여도 오히려 성능이 더 떨어지는 결과가 나옵니다.
# Set random seed
tf.random.set_seed(42)

# 1. Create a model  --> 일단 현재 네트워크 사용: 최종 네트워크는 실험을 통해 결정(성능 vs 비용 비교하여 효율적인 방향으로 설정)
model_1 = tf.keras.Sequential([
           tf.keras.layers.Dense(10000, activation='relu'),
           tf.keras.layers.Dense(5000, activation='relu'),
           tf.keras.layers.Dense(2500, activation='relu'),
           tf.keras.layers.Dense(1250, activation='relu'),
           tf.keras.layers.Dense(625, activation='relu'),
           tf.keras.layers.Dense(50, activation='relu'), 
           tf.keras.layers.Dense(1, activation='linear')
])

# 2. Comile the model
model_1.compile(loss=tf.keras.losses.mse,
                 optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                 metrics=['mse'])


# In[ ]:


# 3. Fit the model
history = model_1.fit(X_train_scaled, 
                      y_train_scaled, 
                      epochs=1000,    # Load가 크지 않은 문제이므로, 충분히 학습할 것(즉, Underfitting이 되지 않도록 할 것). 
                      verbose = 1,
                      # 단, Overfitting이 발생하면 더이상 학습할 필요가 없음. 
                      #validation_split = 0.1)
                      validation_data=(X_valid_scaled, y_valid_scaled))


# In[ ]:


pd.DataFrame(history.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs")

# 그림의 저장
from datetime import datetime
now = datetime.now()
plt.savefig("modelsave/Model"+ str(now.month) + str(now.day) + str(now.hour) + str(now.minute)+ "epochs.jpg")


# In[ ]:





# In[ ]:





# ### Prediciton

# In[ ]:


y_p = model_1.predict(X_test_scaled)


# In[ ]:


y_p[:10], y_test_scaled[:10]


# In[ ]:


y_test_unscaled = scY.inverse_transform(y_test_scaled)  # scaler.inverse_transform(): scaling을 환원(unscaling)
print(y_test_unscaled)
y_p_unscaled = scY.inverse_transform(y_p)
print(y_p_unscaled)


# In[ ]:


# from keras.models import load_model
# model_1 = load_model('modelsave/Model7251545file.h5')


# In[ ]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6,6))

ax.plot(y_test_unscaled, y_p_unscaled, 'r.')
ax.set_xlabel("Tested tensile stress, ft_test (MPa)", fontsize = 14)
ax.set_ylabel("Predicted tensile stress, ft_pred (MPa)", fontsize = 14)
x = np.linspace(0, 1000, 100)
y = x
ax.plot(x, y, 'b')
fig.show()
# 그림의 저장
from datetime import datetime
now = datetime.now()
plt.savefig("modelsave/Model"+ str(now.month) + str(now.day) + str(now.hour) + str(now.minute)+ "comparison.jpg")


# In[ ]:


# 그림의 저장
from datetime import datetime
now = datetime.now()
plt.savefig("modelsave/Model"+ str(now.month) + str(now.day) + str(now.hour) + str(now.minute)+ "comparison.jpg")


# In[ ]:


X_test_unscaled = scX.inverse_transform(X_test_scaled)  # scaler.inverse_transform(): scaling을 환원(unscaling)
print(X_test_unscaled)


# In[ ]:


score = model_1.evaluate(X_test_scaled, y_test_scaled)
print("test loss, test MAE:", score)


# In[ ]:


from sklearn.metrics import r2_score
r2_score(y_test_unscaled, y_p_unscaled)


# In[ ]:


division = y_p_unscaled / y_test_unscaled
cov = np.std(division) / np.mean(division)
print(cov)


# In[ ]:


y_test_unscaled


# In[ ]:


# 모델의 저장
from datetime import datetime
now = datetime.now()
model_1.save("modelsave/Model"+ str(now.month) + str(now.day) + str(now.hour) + str(now.minute)+ "file.h5")
plt.savefig("modelsave/Model"+ str(now.month) + str(now.day) + str(now.hour) + str(now.minute)+ "file.jpg")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




