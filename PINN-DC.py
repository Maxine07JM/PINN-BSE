#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

plt.ion()

# STEP 1

NEW_BATTERY_CSV = r"location.csv"   #location
df_new = pd.read_csv(NEW_BATTERY_CSV)

df_new["Voltage/V"] = pd.to_numeric(df_new["Voltage/V"], errors='coerce')
df_new["Current/A"] = pd.to_numeric(df_new["Current/A"], errors='coerce')   # mA/A
df_new = df_new.dropna(subset=["Voltage/V", "Current/A"])

v_np = df_new["Voltage/V"].values.astype(np.float32)
i_np = df_new["Current/A"].values.astype(np.float32)
num_points = len(v_np)


SAMPLING_INTERVAL = 2.0    # time

CURRENT_CAPACITY_AH = 2.07 
cap_as = CURRENT_CAPACITY_AH * 3600.0

max_v_idx = np.argmax(v_np)

delta_ah = (i_np * SAMPLING_INTERVAL) / 3600.0
cumulative_ah = np.cumsum(delta_ah)

# Highest
total_ah_trend = float(cumulative_ah[max_v_idx]) - float(cumulative_ah[0])

if total_ah_trend < 0:
    i_np = -i_np     # DC or CD

delta_ah = (i_np * SAMPLING_INTERVAL) / 3600.0
cumulative_ah = np.cumsum(delta_ah)
soc_np = 1.0 - (cumulative_ah[max_v_idx] - cumulative_ah) / CURRENT_CAPACITY_AH
soc_np = np.clip(soc_np, 0.0, 1.0)

# (Gated dI/dt）GPT
i_diff = np.zeros_like(i_np)
raw_diff = i_np[1:] - i_np[:-1]
i_diff[1:] = np.where(np.abs(raw_diff) > 0.5, raw_diff, 0.0)

loss_weights = np.ones_like(v_np, dtype=np.float32)
loss_weights[0:300] = 5.0        # C
loss_weights[1595:1750] = 10.0   # M
loss_weights[2700:3000] = 10.0   # D

i_min, i_max = i_np.min(), i_np.max()
i_norm_np = (i_np - i_min) / (i_max - i_min) * 2.0 - 1.0

soc_min, soc_max = soc_np.min(), soc_np.max()
soc_norm_np = (soc_np - soc_min) / (soc_max - soc_min) * 2.0 - 1.0

idiff_min, idiff_max = i_diff.min(), i_diff.max()
if idiff_max == idiff_min: idiff_max += 1.0
idiff_norm_np = (i_diff - idiff_min) / (idiff_max - idiff_min) * 2.0 - 1.0

# PyTorch
v_true = torch.tensor(v_np, dtype=torch.float32)
i_norm = torch.tensor(i_norm_np, dtype=torch.float32)
soc_norm = torch.tensor(soc_norm_np, dtype=torch.float32)
idiff_norm = torch.tensor(idiff_norm_np, dtype=torch.float32)
weight_mask = torch.tensor(loss_weights, dtype=torch.float32)


# STEP 2: Mish 

class UltimateBatteryModel(nn.Module):
    def __init__(self, init_v_ocv):
        super(UltimateBatteryModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Mish(),  
            nn.Linear(64, 64),
            nn.Mish(),
            nn.Linear(64, 32),
            nn.Mish(),
            nn.Linear(32, 1)
        )
        with torch.no_grad():
            self.net[-1].bias.fill_(init_v_ocv)

    def forward(self, soc_arr, current_arr, idiff_arr):
        inputs = torch.stack([soc_arr, current_arr], dim=-1)
        inputs = torch.cat([inputs, idiff_arr.unsqueeze(-1)], dim=-1)
        return self.net(inputs).squeeze(-1)

# STEP 3: Train DMC

# [0] Scalar 
init_voltage = float(v_true[0].item())

model = UltimateBatteryModel(init_voltage)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

epochs = 120 


for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    v_pred = model(soc_norm, i_norm, idiff_norm)
    weighted_loss = torch.mean(((v_pred - v_true) ** 2) * weight_mask)
    weighted_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    if epoch % 20 == 0 or epoch == 1:
         print(f" Epoch [{epoch:03d}/{epochs}] -> Fine-grained loss function: {weighted_loss.item():.6f}")

print("Done")


# STEP 4: Fig.

v_final_pred = v_pred.detach().cpu().numpy()

test_errors = v_np - v_final_pred
mae = np.mean(np.abs(test_errors))
rmse = np.sqrt(np.mean(test_errors ** 2))

print("="*60)
print(f"📉 RMSE: {rmse*1000:.2f} mV")
print(f"📉 MAE:     {mae*1000:.2f} mV")
print("="*60)

fig, ax = plt.subplots(num='Ultimate Physics-Informed Neural Battery Fit', figsize=(11, 5), dpi=100)
ax.plot(v_np, label='True Voltage (New Experiment)', color='black', linewidth=1.5)
ax.plot(v_final_pred, label='Fine-Tuned Neural Prediction', color='red', linestyle='--', linewidth=1.5)
ax.set_title('Fine-Tuned Tracking: Eliminating Pre-activation Sag at 1500 points', fontsize=12, pad=15)
ax.set_xlabel('Data Points (Sampling every 2s)', fontsize=10)
ax.set_ylabel('Voltage (V)', fontsize=10)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='lower right', frameon=True, shadow=True)

plt.draw()
plt.show(block=True)


# In[ ]:




