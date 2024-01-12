# GaussianNB
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

#GaussianNB()
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, r2_score
#from sklearn.preprocessing import KBinsDiscretizer

gnb = GaussianNB()
gnb.fit(x_train_scaled, y_train_scaled)
y_pred = gnb.predict(x_test_scaled)

mse = mean_squared_error(y_test_scaled, y_pred)
r2_test = r2_score(y_test_scaled, y_pred)

print(f'Mean Squared Error:{mse}')
print(f'r2 score:{r2_test}')

