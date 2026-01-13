import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset
from vae import VAE  # 使用VAE模型
from scipy.stats import gaussian_kde
import os

# 设置全局样式
fs = 24
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = fs
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['lines.linewidth'] = 2

# ----------------------------
# 设置文件路径和超参数
# ----------------------------
model_path = "saved_models/vae_model_final.pth"  # 使用VAE模型
output_csv_path = "./latent_space_with_features.csv"

# Hyperparameters (match training)
input_dim = 6
latent_dim = 20
hidden_dim = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 检查CSV文件是否存在，如果存在则加载，否则生成
# ----------------------------
if os.path.exists(output_csv_path):
    print(f"检测到本地文件 {output_csv_path}，直接加载...")
    output_df = pd.read_csv(output_csv_path)

    # 从CSV中提取所需数组
    element_names = ['Ti', 'Mo', 'Nb', 'Zr', 'Sn', 'Ta']
    comp_data = output_df[[f'original_{name}' for name in element_names]].values
    all_recon = output_df[[f'recon_{name}' for name in element_names]].values

    # 提取t-SNE坐标
    z_2d = output_df[['tsne_1', 'tsne_2']].values

    # 提取潜在空间数据
    all_z = np.zeros((len(output_df), latent_dim))
    all_mu = np.zeros((len(output_df), latent_dim))
    all_log_var = np.zeros((len(output_df), latent_dim))
    for i in range(latent_dim):
        all_z[:, i] = output_df[f'latent_{i}']
        all_mu[:, i] = output_df[f'mu_{i}']
        all_log_var[:, i] = output_df[f'log_var_{i}']

    # 提取原始数据
    original_df = output_df.copy()

    print("数据加载完成！")
else:
    print(f"本地文件 {output_csv_path} 不存在，开始生成...")

    # ----------------------------
    # 加载VAE模型
    # ----------------------------
    # Initialize VAE model
    model = VAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim
    ).to(device)

    # Load the model state dict
    checkpoint = torch.load(model_path, map_location=device)

    # 健壮的模型加载方式
    if 'model_state_dict' in checkpoint:
        # 直接加载整个模型
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'encoder_state_dict' in checkpoint and 'decoder_state_dict' in checkpoint:
        # 分别加载编码器和解码器
        model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        # 加载fc_mu和fc_var（如果单独保存）
        if 'fc_mu_state_dict' in checkpoint:
            model.fc_mu.load_state_dict(checkpoint['fc_mu_state_dict'])
        if 'fc_var_state_dict' in checkpoint:
            model.fc_var.load_state_dict(checkpoint['fc_var_state_dict'])
    else:
        # 尝试直接加载整个模型
        model.load_state_dict(checkpoint)

    model.eval()
    print("VAE模型加载成功!")

    # 加载数据
    file_path = 'filtered_titanium_alloy_compositions_1percent_step.csv'
    data = pd.read_csv(file_path)
    comp_data = data.iloc[:, :6].values
    original_df = data.copy()  # 保存原始数据

    # 转换为Tensor，不打乱数据顺序
    X_tensor = torch.tensor(comp_data, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)  # 设置shuffle=False

    # ----------------------------
    # 获取完整潜在空间表示和重构结果
    # ----------------------------
    all_mu = []
    all_log_var = []
    all_z = []
    all_recon = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)

            # VAE前向传播返回 (recons, mu, log_var)
            recons, mu, log_var = model(inputs)

            # 获取潜在空间样本
            z = model.reparameterize(mu, log_var)

            all_mu.append(mu.cpu().numpy())
            all_log_var.append(log_var.cpu().numpy())
            all_z.append(z.cpu().numpy())
            all_recon.append(recons.cpu().numpy())

    # 合并所有批次的结果
    all_mu = np.vstack(all_mu)
    all_log_var = np.vstack(all_log_var)
    all_z = np.vstack(all_z)
    all_recon = np.vstack(all_recon)

    # ----------------------------
    # 检查是否需要使用 t-SNE
    # ----------------------------
    if latent_dim > 2:
        tsne = TSNE(n_components=2, random_state=0, perplexity=30)
        z_2d = tsne.fit_transform(all_z)
        print("使用 t-SNE 将潜在空间降维至 2D")
    else:
        z_2d = all_z
        print("潜在空间已经是 2D，无需使用 t-SNE")

    # ----------------------------
    # 构建完整输出DataFrame
    # ----------------------------
    # 创建基础DataFrame
    output_df = pd.DataFrame({
        'tsne_1': z_2d[:, 0],  # Renamed for clarity
        'tsne_2': z_2d[:, 1],  # Renamed for clarity
    })

    # 添加原始成分数据
    element_names = ['Ti', 'Mo', 'Nb', 'Zr', 'Sn', 'Ta']
    for i, name in enumerate(element_names):
        output_df[f'original_{name}'] = comp_data[:, i]

    # 添加潜在空间数据
    for i in range(latent_dim):
        output_df[f'latent_{i}'] = all_z[:, i]
        output_df[f'mu_{i}'] = all_mu[:, i]
        output_df[f'log_var_{i}'] = all_log_var[:, i]

    # 添加重构成分数据
    for i, name in enumerate(element_names):
        output_df[f'recon_{name}'] = all_recon[:, i]

    # 添加重构误差
    original_errors = np.abs(comp_data - all_recon)
    for i, name in enumerate(element_names):
        output_df[f'error_{name}'] = original_errors[:, i]

    output_df['total_error'] = np.sum(original_errors, axis=1)

    # 合并原始数据的所有列
    output_df = pd.concat([original_df, output_df], axis=1)

    # ----------------------------
    # 保存完整结果到 CSV
    # ----------------------------
    output_df.to_csv(output_csv_path, index=False)
    print(f"完整结果已保存至 {output_csv_path}")

# ----------------------------
# 1. 基础t-SNE可视化
# ----------------------------
plt.figure(figsize=(8, 6))
plt.scatter(z_2d[:, 0], z_2d[:, 1], s=50, alpha=0.7, edgecolor='k', linewidth=0.5)
# plt.title('Latent Space Distribution (t-SNE)', fontsize=fs)
plt.xlabel("t-SNE Dimension 1", fontsize=fs)
plt.ylabel("t-SNE Dimension 2", fontsize=fs)

# 设置刻度字体大小
plt.xticks(fontsize=fs * 0.8)
plt.yticks(fontsize=fs * 0.8)

# 移除网格线
plt.grid(False)

plt.tight_layout()
plt.savefig('tSNE_Latent_Space.png', dpi=300, bbox_inches='tight')
plt.close()

# ----------------------------
# 2. 每个元素在潜在空间中的分布（单独图表）
# ----------------------------
element_names = ['Ti', 'Mo', 'Nb', 'Zr', 'Sn', 'Ta']

for element in element_names:
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(z_2d[:, 0], z_2d[:, 1],
                     c=output_df[f'original_{element}'],
                     cmap='viridis', s=50, alpha=0.8, edgecolor='k', linewidth=0.5)

    cbar = plt.colorbar(sc, pad=0.01)
    cbar.set_label(f'{element} Concentration (%)', fontsize=fs * 0.9)
    cbar.ax.tick_params(labelsize=fs * 0.7)

    plt.title(f'Latent Space Colored by {element} Concentration', fontsize=fs)
    plt.xlabel("t-SNE Dimension 1", fontsize=fs)
    plt.ylabel("t-SNE Dimension 2", fontsize=fs)

    plt.xticks(fontsize=fs * 0.8)
    plt.yticks(fontsize=fs * 0.8)

    # 移除网格线
    plt.grid(False)

    plt.tight_layout()
    plt.savefig(f'Latent_Space_{element}.png', dpi=300, bbox_inches='tight')
    plt.close()

# ----------------------------
# 3. 重构损失分布图
# ----------------------------
plt.figure(figsize=(8, 6))
total_errors = output_df['total_error']

# 使用提供的配色之一（稍深的淡蓝）
error_color = '#6B9BB3'

# 绘制直方图
n, bins, patches = plt.hist(total_errors, bins=30, alpha=0.7, density=True,
                            color=error_color, edgecolor='k', linewidth=1.5)

# 添加密度曲线（使用红色）
kde = gaussian_kde(total_errors)
x = np.linspace(min(total_errors), max(total_errors), 300)
plt.plot(x, kde(x), color='red', linewidth=3)
#
# plt.title('Reconstruction Error Distribution', fontsize=fs)
plt.xlabel('Total Reconstruction Error', fontsize=fs)
plt.ylabel('Density', fontsize=fs)

plt.xticks(fontsize=fs * 0.8)
plt.yticks(fontsize=fs * 0.8)

# 移除网格线
plt.grid(False)

plt.tight_layout()
plt.savefig('Reconstruction_Error_Distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# ----------------------------
# 4. 潜在变量分布 (随机挑选6个维度，叠加在一张图上，只使用曲线)
# ----------------------------
# 挑选6个维度
selected_dims = [0, 5, 8, 9, 15, 19]
print(f"挑选的潜在维度: {selected_dims}")

# 定义提供的配色变体（稍深版本）
image_color_variants = [
    "#6B9BB3",  # 稍深的淡蓝
    "#87B395",  # 稍深的薄荷绿
    "#A497C6",  # 稍深的柔紫
    "#C8B687",  # 稍深的沙黄
    "#B78981",  # 稍深的玫瑰灰
    "#848993"  # 稍深的石板灰
]

plt.figure(figsize=(8, 6))

for idx, dim in enumerate(selected_dims):
    latent_values = all_z[:, dim]
    curve_color = image_color_variants[idx % len(image_color_variants)]

    # 添加KDE曲线（使用不同颜色）
    kde = gaussian_kde(latent_values)
    x = np.linspace(min(latent_values), max(latent_values), 300)
    plt.plot(x, kde(x), color=curve_color, linewidth=5, linestyle='-', label=f'z_{dim}')

# plt.title('Selected Latent Dimensions Distributions', fontsize=fs)
plt.xlabel('Value', fontsize=fs)
plt.ylabel('Density', fontsize=fs)
plt.legend(fontsize=fs * 0.7, frameon=False)
plt.xticks(fontsize=fs * 0.8)
plt.yticks(fontsize=fs * 0.8)

# 移除网格线
plt.grid(False)

plt.tight_layout()
plt.savefig('Selected_Latent_Dimensions_Overlaid.png', dpi=300, bbox_inches='tight')
plt.close()

# ----------------------------
# 5. 新增：t-SNE降维后的重构损失图
# ----------------------------
plt.figure(figsize=(8, 6))
sc = plt.scatter(z_2d[:, 0], z_2d[:, 1],
                 c=output_df['total_error'],
                 cmap='viridis', s=50, alpha=0.8, edgecolor='k', linewidth=0.5)

cbar = plt.colorbar(sc, pad=0.01)
cbar.set_label('Total Reconstruction Error', fontsize=fs * 0.9)
cbar.ax.tick_params(labelsize=fs * 0.7)

# plt.title('Latent Space Colored by Reconstruction Error', fontsize=fs)
plt.xlabel("t-SNE Dimension 1", fontsize=fs)
plt.ylabel("t-SNE Dimension 2", fontsize=fs)

plt.xticks(fontsize=fs * 0.8)
plt.yticks(fontsize=fs * 0.8)

# 移除网格线
plt.grid(False)

plt.tight_layout()
plt.savefig('Latent_Space_Reconstruction_Error.png', dpi=300, bbox_inches='tight')
plt.close()

print("所有图表生成完成!")