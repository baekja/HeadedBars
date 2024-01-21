# Pullout
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
#import tensorflow as tf

df = pd.read_excel("Data for headed bars_for DataFrame_220725.xlsx", skiprows = 17, engine = 'openpyxl', sheet_name= 'headed (2)' )
df = pd.DataFrame(df, columns = ["No.", "Author", "Year", "Test type", "Remark", "Specimen", "fy", "Ld", "fcm", "db", "b", "cos,avg",
                                 "cth", "ch", "Nh", "Bottom cover", "Ah/Ab", "Fsu at La, test", "dtr", "Ntr", "st"]) # st 제거시

df = df[df["Test type"] == "Pullout"] # 
y= df["Fsu at La, test"] 

pd.set_option('display.max_rows',None) # 모든 열 볼 수 있음 

y2=y[~y.isnull()] # shear failure 값 제거

X = df[["Test type", "fy", "Ld", "fcm", "db", "b", "cos,avg", "cth", "Nh", "Ah/Ab", "st"]] 

X = pd.get_dummies(data = X, columns = ["Test type"], prefix = "Test_type")
X = X[~y.isnull()]

pd.options.display.max_rows = None
X.dropna(inplace = True)

y2 = y2[X.index] # 
y2 = y2.loc[(y2 != 0)] # series
X = X.loc[y2.index] # DataFrame


pd.options.display.max_rows = None
X.dropna(inplace = True)

y2 = y2[X.index]
y2 = y2.loc[(y2 != 0)] # series
X = X.loc[y2.index] # DataFrame

from sklearn.model_selection import train_test_split
np.random.seed(0)
X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.20, random_state=142)

from sklearn.preprocessing import MinMaxScaler
scX = MinMaxScaler() #형태는 넘파이
x_train_scaled = scX.fit_transform(X_train)   
x_test_scaled = scX.transform(X_test)

scY = MinMaxScaler()
y_train_scaled = scY.fit_transform(y_train.values.reshape(-1,1)) 
y_test_scaled = scY.transform(y_test.values.reshape(-1,1))

#스케일링->텐서로
x_test_tensor = torch.FloatTensor(x_test_scaled)
y_test_tensor = torch.FloatTensor(y_test_scaled)
x_train_tensor = torch.FloatTensor(x_train_scaled)
y_train_tensor = torch.FloatTensor(y_train_scaled)

class TorchModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, output_size):
        super(TorchModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.layer3 = nn.Linear(hidden_size2, hidden_size3)
        self.layer4 = nn.Linear(hidden_size3, hidden_size4)
        self.layer5 = nn.Linear(hidden_size4, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x = self.layer5(x)
        return x
    
input_size = 11
hidden_size1 = 22 #
hidden_size2 = 44 #
hidden_size3 = 22 #
hidden_size4 = 11 #
output_size = 1
    
#커스텀 데이터 셋
class CustomDataset(TensorDataset): 
    def __init__(self):
        self.x = x_train_tensor
        self.y = y_train_tensor
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x[idx])
        y = torch.FloatTensor(self.y[idx]) 
        return x, y
    
    def __len__(self): 
        return len(self.x)

#데이터 셋에서 train, validation 나누기 
dataset = CustomDataset()
dataset_size = len(dataset)
train_size = int(dataset_size * 0.8)
validation_size = dataset_size - train_size

train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
#print(f"Training Data Size : {len(train_dataset)}")
#print(f"Validation Data Size : {len(validation_dataset)}")

# 데이터 로더
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=5, shuffle=True)

train_losses = []
val_losses = []
train_acc = []
val_acc = []

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TorchModel(input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, output_size).to(device)
criterion = nn.MSELoss().to(device) 
optimizer = torch.optim.Adam(model.parameters(), lr = 0.03) #
nb_epochs = 1000

# 모델 학습
for epoch in range(nb_epochs+1):
    train_loss = 0       
    train_accuracy = 0
    accuracy = 0
    for _,data in enumerate(train_dataloader):
        x,y =data
        x = x.to(device)
        y_train = y.to(device)
        p_train = model(x)
        train_cost =criterion(p_train, y_train)
        
        optimizer.zero_grad()
        train_cost.backward()
        optimizer.step()
        
        train_loss += train_cost.item()
      
    # 모델 검증
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for _,data in enumerate(validation_dataloader):
            x,y = data
            x = x.to(device)
            y_val = y.to(device)
            p_val = model(x)
            val_cost = criterion(p_val, y_val)
            
            val_loss += val_cost.item()
           
    # calculate mean for each batch
    train_losses.append(train_loss / len(train_dataloader))
    val_losses.append(val_loss / len(validation_dataloader))
   
#    if epoch % 100== 0:
#        print("Epoch:{:4d}/{}".format(epoch, nb_epochs),
#              "Train Loss: {:.6f}".format(train_loss / len(train_dataloader)),
#              "Val Loss: {:.6f}".format(val_loss / len(validation_dataloader)))
              

history = {'train_loss': train_losses, 'val_loss': val_losses,
            'y_train':y_train,'p_train': p_train ,'y_val': y_val, 'p_val':p_val}

plt.figure(figsize = (8,4))
plt.plot(history['train_loss'],label = "Train loss")
plt.plot(history['val_loss'],label = "Valid loss")
plt.title(f'Loss', color='white', fontweight = 'bold')
plt.ylabel('Loss', color='white')
plt.xlabel('epoch', color='white')
plt.legend(), plt.grid()

x_test_tensor = x_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)
x_train_tensor = x_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)

p_test = model(x_test_tensor)
y_test_unscaled = scY.inverse_transform(y_test_scaled)  
p_test_unscaled = scY.inverse_transform(p_test.cpu().detach().numpy()) # numpy로 만들기 위해 .cpu()로 변환

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6,6))

ax.plot(y_test_unscaled, p_test_unscaled, 'r.')
ax.set_xlabel("Tested tensile stress, ft_test (MPa)", fontsize = 14,color = 'white')
ax.set_ylabel("Predicted tensile stress, ft_pred (MPa)", fontsize = 14, color = 'white')
x = np.linspace(0, 1000, 100)
y = x
ax.plot(x, y, 'b')
fig.show()

x_test_unscaled = scX.inverse_transform(x_test_scaled)  
#print(x_test_unscaled)

from sklearn.metrics import r2_score, mean_squared_error
R2 = r2_score(y_test_unscaled, p_test_unscaled)
mse = mean_squared_error(y_test_scaled, p_test.detach().numpy())

division = p_test_unscaled / y_test_unscaled
cov = np.std(division) / np.mean(division)

print(f'cov:{cov}')
print(f'r2_score:{R2}')
print(f'mse:{mse}')
print(f'rmse:{np.sqrt(mse)}')
mape = np.mean(np.abs((y_test_unscaled - p_test_unscaled)/y_test_unscaled))*100
mae = np.mean(np.abs(y_test_scaled - p_test.detach().numpy()))
print(f'mape:{mape}')
print(f'mae:{mae}')

import time
now = time

new_data ={
    'NAME':["Pullout"],  # 수정 후 작업 시 이름 변경 필요
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
