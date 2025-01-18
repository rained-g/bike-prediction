import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm.notebook import tqdm
import math
from sklearn.preprocessing import StandardScaler
import random
import wandb
import os
device = torch.device("cuda:0")
#读取数据集
train_dataset = pd.read_csv("train_data.csv")
test_dataset = pd.read_csv("test_data.csv")
#处理数据集

#通过sin和cos编码周期性特征
#季节编码
train_dataset['season_sin'] = np.sin(2 * np.pi * train_dataset['season'] / 4)
train_dataset['season_cos'] = np.cos(2 * np.pi * train_dataset['season'] / 4)
test_dataset['season_sin'] = np.sin(2 * np.pi * test_dataset['season'] / 4)
test_dataset['season_cos'] = np.cos(2 * np.pi * test_dataset['season'] / 4)
#月份编码
train_dataset['month_sin'] = np.sin(2 * np.pi * train_dataset['mnth'] / 12)
train_dataset['month_cos'] = np.cos(2 * np.pi * train_dataset['mnth'] / 12)
test_dataset['month_sin'] = np.sin(2 * np.pi * test_dataset['mnth'] / 12)
test_dataset['month_cos'] = np.cos(2 * np.pi * test_dataset['mnth'] / 12)
#工作日编码
train_dataset['weekday_sin'] = np.sin(2 * np.pi * train_dataset['weekday'] / 7)
train_dataset['weekday_cos'] = np.cos(2 * np.pi * train_dataset['weekday'] / 7)
test_dataset['weekday_sin'] = np.sin(2 * np.pi * test_dataset['weekday'] / 7)
test_dataset['weekday_cos'] = np.cos(2 * np.pi * test_dataset['weekday'] / 7)
#小时编码
train_dataset['hour_sin'] = np.sin(2 * np.pi * train_dataset['hr'] / 24)
train_dataset['hour_cos'] = np.cos(2 * np.pi * train_dataset['hr'] / 24)
test_dataset['hour_sin'] = np.sin(2 * np.pi * test_dataset['hr'] / 24)
test_dataset['hour_cos'] = np.cos(2 * np.pi * test_dataset['hr'] / 24)

# 对训练集特征进行标准化
scaler = StandardScaler()
features = ['yr', 'holiday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 
            'season_sin', 'season_cos', 'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos', 'hour_sin', 'hour_cos']
train_dataset[features] = scaler.fit_transform(train_dataset[features])
#创建滑动窗口
def create_timeseries_data(df, window_size=96, forecast_horizon=96):
    X, y = [], []       #两个数组用于保存值
    for i in range(window_size, len(df) - forecast_horizon + 1):
        # X.append(df.iloc[i-window_size:i, 2:].values)  
        # 过去96小时的数据
        #['mnth','holiday','workingday','weathersit','temp', 'atemp', 'hum', 'windspeed']   8特征值
        X.append(df.iloc[i-window_size:i][['yr', 'holiday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'season_sin', 'season_cos', 'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos', 'hour_sin', 'hour_cos']].values)
        y.append(df.iloc[i:i + forecast_horizon]['cnt'].values)    #待预测的未来96小时的cnt
    return np.array(X), np.array(y)

# 创建96小时的时间窗口数据
X, y = create_timeseries_data(train_dataset, window_size=96, forecast_horizon=240)
X_test, y_test =  create_timeseries_data(test_dataset, window_size=96, forecast_horizon=240)


train_size = int(len(X) * 0.99)
val_size = int(len(X) * 0.05)

X_train, X_val = X[:train_size], X[train_size:train_size+val_size]
y_train, y_val = y[:train_size], y[train_size:train_size+val_size]

X_test = X_test[:train_size]
y_test = y_test[:train_size]
# X_train = X
# y_train = y

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)

if np.any(np.isnan(y_test)):
    print("y_true contains NaN values")

#构建Dataloader
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)


X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)


X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 创建DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

b_size = 128
train_loader = DataLoader(train_dataset, batch_size=b_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=b_size)
test_loader = DataLoader(test_dataset, batch_size=b_size)

#定义Transformer结构
class TransformerModel(nn.Module):
    def __init__(self, input_dim, seq_len, output_dim, nhead=4, num_encoder_layers=6, dim_feedforward=512):
        super(TransformerModel, self).__init__()
        
        # Transformer Encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward, 
                batch_first=True  # 启用 batch_first
            ), 
            num_layers=num_encoder_layers
        )
        
        # Fully connected layer for output prediction
        self.fc = nn.Linear(input_dim, output_dim)  # output_dim should be the forecast horizon (96 in your case)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim), batch_first=True, so no need for permute
        # Pass through the transformer encoder
        transformer_out = self.encoder(x)
        
        # Taking the output of the last time step for prediction
        # Here, we want to get predictions for all future 96 hours
        output = transformer_out[:, -1, :]  # This is the representation of the last time step
        
        # Fully connected layer to produce the output for all 96 hours (forecast_horizon)
        output = self.fc(output)  # The output should now have shape (batch_size, forecast_horizon)
        return output
model = TransformerModel(input_dim=X_train.shape[2], seq_len=X_train.shape[1], output_dim=y_train.shape[1]).to(device)


# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=2e-4)

save_path = "./results/checkpoints_240/5"
os.makedirs(save_path, exist_ok=True)
# 训练循环
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    # 在每个epoch开始时创建一个进度条
    progress_bar_epoch = tqdm(total=1, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True, mininterval=1.0)
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.float().to(device), targets.float().to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        # 记录每个batch的loss
        wandb.log({"train_loss": loss.item(), "step": batch_idx})
    
    # 记录并更新每个epoch的总loss
    wandb.log({"total_train_loss": running_loss / len(train_loader), "epoch": epoch+1})

    # 每过40个epoch保存一次模型
    if epoch % 40 == 0:
        checkpoint_path = os.path.join(save_path, f"checkpoint_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

    # 更新 epoch 进度条的状态
    progress_bar_epoch.update(1)  # 完成当前 epoch

    # 打印当前 epoch 的损失
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

results = []  # List to store individual metrics (MSE, MAE)


# Assuming `test_loader` and `model` are already defined
model.load_state_dict(torch.load("./results/checkpoints_240/3/checkpoint_epoch921.pth"))
model.eval()  # Set model to evaluation mode
criterion = nn.MSELoss()

y_true = []  # List to store true values (targets)
y_pred = []  # List to store predicted values


with torch.no_grad():  # No need to track gradients during inference
    test_loss = 0.0
    pre_loss = 60000
    index = -1
    num = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.float().to(device), targets.float().to(device)
        
        # Forward pass
        outputs = model(inputs)  # Get model outputs

        # Compute the loss
        loss = criterion(outputs.squeeze(), targets)  # Assuming targets have shape (batch_size, 96)
        if(loss<pre_loss):
            index =batch_idx
        test_loss += loss.item()  # Accumulate the test loss
        # print(loss)
        # Store true and predicted values for metric calculation
        y_true.append(targets.cpu().numpy())  # Move targets to CPU for numpy conversion
        y_pred.append(outputs.cpu().numpy())  # Move predictions to CPU for numpy conversion
        # # Store results
        # if num<6:
        #     num+=1
        #     mse = mean_squared_error(targets.cpu().numpy(), outputs.squeeze().cpu().numpy())
        #     mae = mean_absolute_error(targets.cpu().numpy(), outputs.squeeze().cpu().numpy())
        #     results.append((mse, mae))

# Concatenate all batches to compute final metrics
y_true = np.concatenate(y_true, axis=0)  # Shape: (num_samples, 96)
y_pred = np.concatenate(y_pred, axis=0)  # Shape: (num_samples, 96)

print(f"最终结果的shape:{y_pred.shape},真实结果的shape{y_true.shape},最佳结果为第{index}组")
# Calculate MSE and MAE for the entire test set
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)

# Print MSE and MAE for the test set
print(f"Test Loss (MSE): {test_loss / len(train_loader)}")
print(f"Test MSE: {mse}")
print(f"Test MAE: {mae}")


results.append((mse, mae))


# Optionally, save predictions to a CSV file
predictions_df = pd.DataFrame(y_pred, columns=[f"hour_{i+1}" for i in range(96)])  # 96 hours prediction
predictions_df.to_csv('predictions.csv', index=False)

# Optionally, save the model's predictions along with the true values
true_values_df = pd.DataFrame(y_true, columns=[f"hour_{i+1}" for i in range(96)])
true_values_df.to_csv('true_values.csv', index=False)

#   画图
sample_idx = 115
# 使用 Seaborn 风格
sns.set(style="whitegrid")
# 绘制真实值
plt.plot(y_true[sample_idx], label='True Values', color='b', linestyle='-', marker=',', markersize=6, linewidth=2)

# 绘制预测值
plt.plot(y_pred[sample_idx], label='Predicted Values', color='r', linestyle='-', marker=',', markersize=6, linewidth=2)

# 添加标题和标签
plt.title(f"True vs Predicted Values for Sample {sample_idx + 1}", fontsize=16, fontweight='bold')
plt.xlabel("Hours", fontsize=14)
plt.ylabel("Rental Count", fontsize=14)

# 设置 x 轴刻度（如果需要）
plt.xticks(np.arange(0, y_train.shape[1], 20), fontsize=12)  # 每 8 个小时一个刻度
# 设置 y 轴刻度
plt.yticks(fontsize=12)

# 添加图例
plt.legend(loc='upper right', fontsize=12, frameon=False)

# 网格线设置（如果需要）
plt.grid(True, linestyle='--', alpha=0.6)

# 保存图像
plt.tight_layout()  # 自动调整布局
plt.savefig(f'true_vs_predicted_sample{sample_idx + 1}_optimized.png')
# 显示图像
plt.show()
