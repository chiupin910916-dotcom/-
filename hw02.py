# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 設定繪圖解析度
plt.rcParams['figure.dpi'] = 144 

# --- 防呆機制：生成測試圖 ---
# 如果沒有這張圖，程式會自動產生一張，避免報錯
file_path = 'data/svd_demo1.jpg'
if not os.path.exists(file_path):
    if not os.path.exists('data'):
        os.makedirs('data')
    # 產生 512x512 的隨機雜訊圖
    dummy_array = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    dummy_img = Image.fromarray(dummy_array)
    dummy_img.save(file_path)
    print(f"已生成測試圖片於: {file_path}")

# -------------------------------------------
# 自定義特徵值與 SVD 函式
# -------------------------------------------
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
    
    # 數值穩定性處理，避免極小的浮點數誤差
    lambdas = np.append(lambdas, 1e-12)
    rank = np.argwhere(lambdas < 1e-6).min()
    lambdas, V = lambdas[0:rank], V[:, 0:rank]
    
    U = A @ V / np.sqrt(lambdas)
    Sigma = np.diag(np.sqrt(lambdas))
    return U, Sigma, V

# -------------------------------------------
# 主程式區塊
# -------------------------------------------

# [關鍵] 使用 PIL 讀取圖片，避開 cv2 與中文路徑的問題
try:
    img_pil = Image.open(file_path).convert('L') # 轉為灰階
    A = np.array(img_pil).astype(np.float64)
except Exception as e:
    print(f"讀圖錯誤: {e}")
    # 萬一真的讀不到，生成隨機矩陣讓程式能跑完
    A = np.random.rand(512, 512) * 255

# SVD 分解
U, Sigma, V = mysvd(A)
VT = V.T

# 計算能量函式 (作業要求補完的部分)
def compute_energy(X: np.ndarray):
    # 矩陣能量 = 所有元素的平方和
    return np.sum(X**2)

# 準備計算 SNR
img_h, img_w = A.shape
keep_r = 201
rs = np.arange(1, keep_r)

# 1. 計算原圖能量
energy_A = compute_energy(A)
energy_N = np.zeros(keep_r) 
snr_list = []

print("正在計算 SNR (請稍候)...")

for r in rs:
    if r > len(Sigma): 
        # 如果 r 超過矩陣秩，維持上一個 SNR 值
        snr_list.append(snr_list[-1])
        continue

    # 重建影像 A_bar = U_r * Sigma_r * VT_r
    A_bar = U[:, 0:r] @ Sigma[0:r, 0:r] @ VT[0:r, :] 
    
    # 計算雜訊 (原始 - 重建)
    Noise = A - A_bar 
    e_noise = compute_energy(Noise)
    energy_N[r] = e_noise
    
    # 計算 SNR
    if e_noise > 1e-10:
        snr = 10 * np.log10(energy_A / e_noise)
    else:
        snr = 100 # 極大值代表完美重建
    
    snr_list.append(snr)

# 繪圖
plt.figure(figsize=(8, 6))
plt.plot(rs, snr_list, 'r-', linewidth=2)
plt.title('Signal-to-Noise Ratio (SNR) vs Rank r')
plt.xlabel('r')
plt.ylabel('SNR (dB)')
plt.grid(True)
plt.show()

# --------------------------
# 驗證步驟
# --------------------------
print("\n--- 驗證結果 ---")
# 特徵值 lambda = 奇異值 sigma 的平方
singular_values_sq = np.diag(Sigma)**2

# 隨機抽樣檢查
check_indices = [10, 50, 100, 150]
print(f"{'r':<5} | {'Noise Energy':<20} | {'Sum of Discarded Lambdas':<25}")
print("-" * 60)

for r in check_indices:
    if r < len(energy_N):
        calc_e = energy_N[r]
        # 理論值 = 捨棄掉的特徵值總和
        theo_e = np.sum(singular_values_sq[r:])
        print(f"{r:<5} | {calc_e:<20.4f} | {theo_e:<25.4f}")