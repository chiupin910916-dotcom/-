import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 設定圖片解析度
figdpi = 100

# ---------------------------------------------------------
# 1. 讀取資料 (或是生成模擬資料)
# ---------------------------------------------------------
try:
    # 嘗試讀取檔案
    hw9_csv = pd.read_csv('hw9.csv').to_numpy(dtype=np.float64)
    t = hw9_csv[:, 0]             # 時間
    flow_velocity = hw9_csv[:, 1] # 氣體流速
    dt = t[1] - t[0]              # 自動偵測取樣間隔 (通常是 0.01)
    print("成功讀取 hw9.csv")
    
except FileNotFoundError:
    print("找不到 hw9.csv，正在生成模擬數據以供演示...")
    # 生成模擬數據 (包含漂移誤差)
    t = np.arange(0, 20, 0.01)
    dt = 0.01
    # 模擬呼吸訊號 (正弦波) + 雜訊 + 致命的固定誤差 (0.5)
    clean_flow = 10 * np.sin(2 * np.pi * 0.2 * t)
    sensor_error = 0.5 # 這就是導致累積誤差的兇手
    flow_velocity = clean_flow + sensor_error + np.random.normal(0, 0.1, len(t))

# ---------------------------------------------------------
# 第一部分：繪製原始流速 (老師題目的第1點)
# ---------------------------------------------------------
plt.figure(figsize=(10, 8), dpi=figdpi)

plt.subplot(3, 1, 1)
plt.plot(t, flow_velocity, 'r')
plt.title('1. Raw Gas Flow Velocity (with sensor bias)')
plt.xlabel('time in seconds')
plt.ylabel('ml/sec')
plt.grid(True, alpha=0.3)

# ---------------------------------------------------------
# 第二部分：直接積分 (老師題目的第2點 - 會有累積誤差)
# ---------------------------------------------------------
# 積分公式: 累積和 * 時間間隔 (dt)
# 老師的程式碼寫 * 0.01，這裡我們用變數 dt 比較保險
net_vol = np.cumsum(flow_velocity) * dt

plt.subplot(3, 1, 2)
plt.plot(t, net_vol, 'r')
plt.title('2. Net Volume (Drifting due to accumulated error)')
plt.xlabel('time in seconds')
plt.ylabel('ml')
plt.grid(True, alpha=0.3)

# ---------------------------------------------------------
# 第三部分：修正累積誤差 (你的作業解答)
# 方法：移除直流偏差 (Remove DC Offset / Bias)
# ---------------------------------------------------------

# 步驟 A: 計算流速的平均值 (這個值通常代表感測器的固定誤差)
bias_offset = np.mean(flow_velocity)

# 步驟 B: 將原始流速減去這個偏差
corrected_velocity = flow_velocity - bias_offset

# 步驟 C: 重新積分
corrected_net_vol = np.cumsum(corrected_velocity) * dt

# ---【新增這行】步驟 D: 將體積去中心化 (修正垂直位置) ---
# 這會把整條線往上拉，讓它圍繞著 0 震盪
corrected_net_vol = corrected_net_vol - np.mean(corrected_net_vol)

plt.subplot(3, 1, 3)
plt.plot(t, corrected_net_vol, 'b') # 用藍色表示修正後
plt.title(f'3. Corrected Net Volume (Bias removed: {bias_offset:.4f})')
plt.xlabel('time in seconds')
plt.ylabel('ml')
plt.grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig('hw9_solution.svg') # 如果需要存檔可取消註解
plt.show()
