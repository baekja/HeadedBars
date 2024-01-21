#knn_regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
 
 
df = pd.read_excel("Data for headed bars_for DataFrame_220725.xlsx", skiprows = 17, engine='openpyxl', sheet_name= 'headed (2)' )
df = pd.DataFrame(df, columns = ["No.", "Author", "Year", "Test type", "Remark", "Specimen", "fy", "Ld", "fcm", "db", "b", "cos,avg",
                                 "cth", "ch", "Nh", "Bottom cover", "Ah/Ab", "Fsu at La, test", "dtr", "Ntr", "st"]) # st 제거시

df = df[df["Test type"] == "Joint type"]
y= df["Fsu at La, test"] 

pd.set_option('display.max_rows',None)

y2=y[~y.isnull()]
X = df[["Test type", "fy", "Ld", "fcm", "db", "b", "cos,avg", "cth", "ch", "Nh", "Bottom cover", "Ah/Ab", "st"]] 

X = pd.get_dummies(data = X, columns = ["Test type"], prefix = "Test_type")
X = X[~y.isnull()]

pd.options.display.max_rows = None
X.dropna(inplace = True)

y2 = y2[X.index]
y2 = y2.loc[(y2 != 0)] # series
X = X.loc[y2.index] # DataFrame

from sklearn.model_selection import train_test_split
np.random.seed(0)
X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.20, random_state=142)
#print(X_test.shape, y_test.shape) => (54,13) (54,)

from sklearn.preprocessing import MinMaxScaler
scX = MinMaxScaler() #형태는 넘파이
x_train_scaled = scX.fit_transform(X_train)   
x_test_scaled = scX.transform(X_test)

scY = MinMaxScaler()
y_train_scaled = scY.fit_transform(y_train.values.reshape(-1,1)).ravel() 
y_test_scaled = scY.transform(y_test.values.reshape(-1,1)).ravel()

from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor(n_neighbors=1) #모델을 조금 더 복잡하게 만들기  
knr.fit(x_train_scaled,y_train_scaled)
y_pred = knr.predict(x_test_scaled)


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test_scaled, y_pred)
r2_test = r2_score(y_test_scaled, y_pred)

print(f'test r2 score: {r2_test}')
print(f'Mean Squared Error: {mse}')

division = y_pred / y_test_scaled
cov = np.std(division) / np.mean(division)

print(f'cov:{cov}')
print(f'rmse:{np.sqrt(mse)}')
mape = np.mean(np.abs((y_test_scaled - y_pred)/y_test_scaled))*100
mae = np.mean(np.abs(y_test_scaled - y_pred))
print(f'mape:{mape}')
print(f'mae:{mae}')


y_test_unscaled = scY.inverse_transform(y_test_scaled.reshape(1,-1))  
y_test_pred_unscaled = scY.inverse_transform(y_pred.reshape(1,-1))

fig, ax = plt.subplots(figsize = (6,6))
ax.plot(y_test_unscaled, y_test_pred_unscaled, 'r.')
ax.set_xlabel("Tested tensile stress, ft_test (MPa)", fontsize = 14,color = 'white')
ax.set_ylabel("Predicted tensile stress, ft_pred (MPa)", fontsize = 14, color = 'white')
x = np.linspace(0, 1000, 100)
y = x
ax.plot(x, y, 'b')
fig.show()
#fig.savefig('KNR_image.png')

import time
now = time

new_data ={
    'NAME':["test_knr"],  # 수정 후 작업 시 이름 변경 필요
    'cov' :[cov],
    'r2_score':[r2_test],
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
