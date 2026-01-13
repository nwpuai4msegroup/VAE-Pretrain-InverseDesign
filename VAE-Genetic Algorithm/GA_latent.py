import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
# from vae import VAE  # Commented out as not used in this script

# Set global style
fs = 30
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = fs
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['lines.linewidth'] = 2

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Skip VAE loading since latents are pre-computed in CSV
print("Skipping VAE loading as latents are pre-computed.")

# Load pre-trained latent space from CSV file for background
latent_space_csv_path = 'latent_space_with_features.csv'
if not os.path.exists(latent_space_csv_path):
    raise FileNotFoundError(f"Latent space CSV not found: {latent_space_csv_path}")

df_latent = pd.read_csv(latent_space_csv_path)
print(f"Loaded latent space from {latent_space_csv_path}, shape: {df_latent.shape}")
# Extract latent vectors (latent_0 to latent_19)
latent_dim = 20
latent_columns = [f'latent_{i}' for i in range(latent_dim)]
if all(col in df_latent.columns for col in latent_columns):
    z_pre_np = df_latent[latent_columns].values.astype(float)
    num_samples_pre = len(z_pre_np)
    print(f"Pre-trained latent space shape: {z_pre_np.shape}")
else:
    raise ValueError("Latent columns not found in CSV")

# Load GA best per generation from CSV
target_save_dir = "./target_GA"
ga_csv_path = os.path.join(target_save_dir, 'best_per_generation.csv')
if not os.path.exists(ga_csv_path):
    raise FileNotFoundError(f"GA bests CSV not found: {ga_csv_path}")

df_ga = pd.read_csv(ga_csv_path, dtype={'generation': str})
print(f"Loaded GA bests from {ga_csv_path}, shape: {df_ga.shape}")

# Selected generations as strings - updated to 10 points
selected_gens_str = ['initial', '0', '9', '24', '25', '26', '30']
df_ga_selected = df_ga[df_ga['generation'].isin(selected_gens_str)].copy()
print(f"Selected generations: {selected_gens_str}")
print(f"Selected rows: {len(df_ga_selected)}")

# Use all selected rows (including initial)
ga_latents = df_ga_selected[latent_columns].values.astype(float)
ga_generations_list = df_ga_selected['generation'].values
ga_fitness_raw = df_ga_selected['fitness']
ga_fitness = pd.to_numeric(ga_fitness_raw).values

# Numeric generations for normalization (map 'initial' to 0, others int(g)+1)
ga_gen_numeric = []
for g in ga_generations_list:
    if g == 'initial':
        ga_gen_numeric.append(0)
    else:
        ga_gen_numeric.append(int(g) + 1)
ga_generations = np.array(ga_gen_numeric)

num_ga = len(ga_latents)
if num_ga == 0:
    raise ValueError("No selected generations found.")
print(f"GA selected bests shape: {ga_latents.shape}, generations: {ga_generations.min()} to {ga_generations.max()}")

# To plot together, concatenate all latents: pre + GA
all_z = np.vstack([z_pre_np, ga_latents])
tsne = TSNE(n_components=2, perplexity=min(50, len(all_z)-1), random_state=1234, n_iter=1000)  # Increased perplexity
all_tsne = tsne.fit_transform(all_z)

# Split back
z_pre_tsne = all_tsne[:num_samples_pre]
z_ga_tsne = all_tsne[num_samples_pre :].copy()  # Copy to modify

# Separate GA points by offsetting x to ensure they are distinct
offset_step = 2.0  # Adjust this value to control separation (larger = more spread)
for i in range(num_ga):
    # Offset in x direction, centered around mean
    offset = (i - (num_ga - 1) / 2) * offset_step
    z_ga_tsne[i, 0] += offset

print("t-SNE completed (pre-trained + selected GA points with offsets).")

# Fit Gaussian KDE for density estimation on pre-trained latent space (t-SNE 2D) - use original for background
kde = gaussian_kde(z_pre_tsne.T, bw_method=0.5)

# Create meshgrid based on original all_tsne for background consistency
all_x_orig = all_tsne[:, 0]
all_y_orig = all_tsne[:, 1]
margin = 5.0
x_min, x_max = all_x_orig.min() - margin, all_x_orig.max() + margin
y_min, y_max = all_y_orig.min() - margin, all_y_orig.max() + margin
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
positions = np.vstack([xx.ravel(), yy.ravel()])
density = kde(positions).reshape(xx.shape)
density = np.clip(density, 1e-12, None)
log_density = np.log(density)

# -----------------------------
#      ⭐ 图像 + 颜色条布局
# -----------------------------
fig1 = plt.figure(figsize=(8, 6))
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(
    1, 2,
    width_ratios=[20, 1],   # 左图大，右侧 bar 小
    wspace=0.15             # 左右间距
)

ax1 = fig1.add_subplot(gs[0, 0])
cax = fig1.add_subplot(gs[0, 1])

# Contour (⚠ remove extend to avoid triangular arrow)
cmap = plt.get_cmap('RdBu_r')

# 修改颜色棒范围 - 根据实际数据调整
density_min = log_density.min()
density_max = log_density.max()
print(f"Log density range: {density_min:.2f} to {density_max:.2f}")

# 设置合适的颜色棒范围
vmin = max(density_min, -20)  # 确保最小值合理
vmax = min(density_max, -5)   # 确保最大值合理

contour = ax1.contourf(
    xx, yy, log_density,
    levels=100,
    cmap=cmap,
    alpha=0.65,
    vmin=vmin,
    vmax=vmax,
    antialiased=True
)

# Remove contour edges
for c in contour.collections:
    c.set_edgecolor("face")

# Plot each selected GA best individually for legend
colors = plt.cm.plasma(np.linspace(0, 1, num_ga))
for i in range(num_ga):
    label = f'Gen {ga_gen_numeric[i]}'
    ax1.scatter(z_ga_tsne[i, 0], z_ga_tsne[i, 1], c=[colors[i]], s=150,  # Adjusted size to match
                edgecolor='k', linewidth=0.7, zorder=5, label=label)

# Connect with line to show evolution path using offset positions
# ax1.plot(z_ga_tsne[:, 0], z_ga_tsne[:, 1], color='black', linewidth=3, alpha=0.8, zorder=2)  # Removed as per request

# 只显示右下角1/4区域
x_center = (x_min + x_max) / 2
y_center = (y_min + y_max) / 2
ax1.set_xlim(x_center, x_max)
ax1.set_ylim(y_min, y_center)

# 添加横纵坐标刻度（只保留整数）- 基于新的显示范围计算4个整数刻度
# 计算x轴整数范围（右下角区域）
x_display_min = x_center
x_display_max = x_max
x_int_min = int(np.ceil(x_display_min))
x_int_max = int(np.floor(x_display_max))
# 确保有4个整数刻度
if x_int_max > x_int_min:
    x_ticks = np.linspace(x_int_min, x_int_max, 4).astype(int)
    # 确保刻度唯一且为整数
    x_ticks = np.unique(x_ticks)
    if len(x_ticks) < 4:
        # 如果刻度不足4个，扩展范围
        x_ticks = np.linspace(x_int_min, x_int_max + (4 - len(x_ticks)), 4).astype(int)
else:
    # 如果范围太小，使用显示范围边界附近的整数
    x_ticks = np.linspace(x_display_min, x_display_max, 4).astype(int)

# 计算y轴整数范围（右下角区域）
y_display_min = y_min
y_display_max = y_center
y_int_min = int(np.floor(y_display_min))
y_int_max = int(np.ceil(y_display_max))
# 确保有4个整数刻度
if y_int_max > y_int_min:
    y_ticks = np.linspace(y_int_min, y_int_max, 4).astype(int)
    # 确保刻度唯一且为整数
    y_ticks = np.unique(y_ticks)
    if len(y_ticks) < 4:
        # 如果刻度不足4个，扩展范围
        y_ticks = np.linspace(y_int_min, y_int_max + (4 - len(y_ticks)), 4).astype(int)
else:
    # 如果范围太小，使用显示范围边界附近的整数
    y_ticks = np.linspace(y_display_min, y_display_max, 4).astype(int)

# 设置刻度
ax1.set_xticks(x_ticks)
ax1.set_yticks(y_ticks)

# 确保刻度标签显示为整数（避免科学计数法）
ax1.ticklabel_format(useOffset=False, style='plain')
ax1.tick_params(axis='both', which='major', labelsize=fs * 0.8, width=2, length=6)

ax1.set_xlabel("Tsne-dimension 1", fontsize=fs+1)
ax1.set_ylabel("Tsne-dimension 2", fontsize=fs+1)

# ---------- Colorbar ----------
cbar1 = fig1.colorbar(contour, cax=cax)
cbar1.set_label("Log Likelihood", fontsize=fs, rotation=270, labelpad=35)
cbar1.ax.tick_params(labelsize=fs * 0.8)

# 修改颜色棒刻度 - 根据实际范围动态设置
cbar_ticks = np.linspace(vmin, vmax, 4)  # 在范围内均匀设置4个刻度
cbar1.set_ticks(cbar_ticks)
# 格式化刻度标签，保留整数
cbar1.set_ticklabels([f'{tick:.0f}' for tick in cbar_ticks])

# Legend for the points
ax1.legend(fontsize=fs*0.8, loc='upper left', frameon=False, fancybox=True, ncol=2 if num_ga > 4 else 1)

plt.tight_layout()
plt.savefig('tsne_selected_ga_evolution_in_pretrained_space.png', dpi=300, bbox_inches='tight')
plt.show()

print("t-SNE plot with selected GA evolution saved as 'tsne_selected_ga_evolution_in_pretrained_space.png'")

# Plot fitness vs selected generations to show convergence
fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
ax2.plot(ga_generations, ga_fitness, color='blue', linewidth=3, marker='o', markersize=10)
ax2.set_xlabel('Generation', fontsize=fs)
ax2.set_ylabel('Best Min Modulus (GPa)', fontsize=fs)
ax2.tick_params(axis='both', which='major', labelsize=fs * 0.7)
ax2.grid(False)

# 确保世代数（x轴）显示为整数
ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# Add a dummy legend entry or remove if not needed
ax2.plot([], [], color='blue', label='Best Fitness')
ax2.legend(fontsize=fs * 0.8, loc='upper right')

plt.tight_layout()
plt.savefig('selected_ga_fitness_convergence.png', dpi=300, bbox_inches='tight')
plt.show()

print("Selected fitness convergence plot saved as 'selected_ga_fitness_convergence.png'")