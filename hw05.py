import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pandas as pd

# --- 核心演算法 ---
def myeig(A, symmetric=False):
    if symmetric:
        lambdas, V = np.linalg.eigh(A)
    else:
        lambdas, V = np.linalg.eig(A)
    lambdas_real = np.real(lambdas)
    sorted_idx = lambdas_real.argsort()[::-1] 
    return lambdas[sorted_idx], V[:, sorted_idx]

def mysvd(A):
    lambdas, V = myeig(A.T @ A, symmetric=True)
    lambdas, V = np.real(lambdas), np.real(V)
    lambdas = np.append(lambdas, 1e-12)
    rank = np.argwhere(lambdas < 1e-6).min()
    lambdas, V = lambdas[0:rank], V[:, 0:rank]
    U = A @ V / np.sqrt(lambdas)
    Sigma = np.diag(np.sqrt(lambdas))
    return U, Sigma, V

def poly_data_matrix(x: np.ndarray, n: int):
    m = x.shape[0]
    X = np.zeros((m, n + 1))
    X[:, 0] = 1.0
    for deg in range(1, n + 1):
        X[:, deg] = X[:, deg - 1] * x
    return X

def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# --- 主程式 ---

# 1. 讀取資料
try:
    df = pd.read_csv('hw5.csv')
except FileNotFoundError:
    df = pd.read_csv('data/hw5.csv')
    
data = df.to_numpy(dtype=np.float64)
hours = data[:, 0]    # 時間 (x)
sulfate = data[:, 1]  # 濃度 (y)

plt.figure(figsize=(12, 5))

# ==========================================
# 左圖：第 1 & 2 小題 (原始尺度 + 多項式迴歸)
# ==========================================
plt.subplot(1, 2, 1)
plt.plot(hours, sulfate, 'r.', markersize=8, label='True values')

# 建立 3 次多項式模型
degree = 3
X_poly = poly_data_matrix(hours, degree)

# SVD 求解
U, Sigma, V = mysvd(X_poly)
w_poly = V @ la.inv(Sigma) @ U.T @ sulfate

# 產生平滑曲線用
x_smooth = np.linspace(hours.min(), hours.max(), 100)
X_smooth = poly_data_matrix(x_smooth, degree)
y_smooth = X_smooth @ w_poly

# 計算 R^2
y_pred_poly = X_poly @ w_poly
r2_poly = calculate_r2(sulfate, y_pred_poly)

plt.plot(x_smooth, y_smooth, 'b-', linewidth=2, label=f'Poly fit (deg={degree})')
plt.title(f'1&2. Sulfate vs Time\n$R^2$ = {r2_poly:.4f}')
plt.xlabel('Time (hours)')
plt.ylabel('Sulfate Concentration')
plt.legend()
plt.grid(True, alpha=0.3)


# ==========================================
# 右圖：第 3 & 4 小題 (雙對數尺度 + 線性迴歸)
# ==========================================
plt.subplot(1, 2, 2)

# 轉換數據為對數
log_hours = np.log(hours)
log_sulfate = np.log(sulfate)

plt.plot(log_hours, log_sulfate, 'r.', markersize=8, label='Log-Log values')

# 建立線性模型 (對數空間中的直線)
X_log = poly_data_matrix(log_hours, 1) # 1次多項式 = 直線

# SVD 求解
U_l, Sigma_l, V_l = mysvd(X_log)
w_log = V_l @ la.inv(Sigma_l) @ U_l.T @ log_sulfate

# 計算預測值 (這就是預測曲線的數據)
log_y_pred = X_log @ w_log

# 計算 R^2
r2_log = calculate_r2(log_sulfate, log_y_pred)

# [修正點] 畫出預測曲線
# 因為 log_hours 已經是排序過的(隨時間增加)，直接畫線即可
plt.plot(log_hours, log_y_pred, 'b-', linewidth=2, label='Log-Linear fit')

plt.title(f'3&4. Log(Sulfate) vs Log(Time)\n$R^2$ = {r2_log:.4f}')
plt.xlabel('Log(Time)')
plt.ylabel('Log(Concentration)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- 輸出文字解釋 ---
print("=== 模型資訊 ===")
print(f"左圖 (Poly Regression) R^2: {r2_poly:.4f}")
print(f"右圖 (Log-Log Linear)  R^2: {r2_log:.4f}")
print(f"Log-Log 回歸公式: log(y) = {w_log[1]:.4f} * log(x) + {w_log[0]:.4f}")
