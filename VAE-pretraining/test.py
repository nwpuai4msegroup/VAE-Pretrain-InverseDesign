import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from betavae import BetaVAE  # 请确保 betavae 库已正确安装

# 字体和样式设置
fs = 35
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = fs

# ----------------------------
# 设置文件路径和超参数
# ----------------------------
model_path = "./model_parameters/beta_vae_model_epoch_15000.pth"
output_train_csv_path = "./train_data_with_predictions.csv"
output_test_csv_path = "./test_data_with_predictions.csv"

# 超参数设置（确保与训练时相同）
input_dim = 6
latent_dim = 10  # 自动判断是否使用 t-SNE
hidden_dim = 300  # 根据您的模型代码设置
beta = 4
gamma = 10.0
max_capacity = 20

Capacity_max_iter = 500
loss_type = 'H'
n_heads = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 加载模型
# ----------------------------
model = BetaVAE(
    input_dim=input_dim,
    latent_dim=latent_dim,
    hidden_dim=hidden_dim,
    beta=beta,
    gamma=gamma,
    max_capacity=max_capacity,
    Capacity_max_iter=Capacity_max_iter,
    loss_type=loss_type,
    n_heads=n_heads
).to(device)

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("模型加载成功。")
except FileNotFoundError:
    print(f"未找到模型文件: {model_path}")
    exit(1)
except Exception as e:
    print(f"加载模型时出错: {e}")
    exit(1)

# ----------------------------
# 加载数据并进行划分
# ----------------------------

file_path = 'ms.xlsx'
data = pd.read_excel(file_path)

comp_data = data.iloc[:112, 1:7].values  # 成分
performance = data.iloc[:112, 7].values  # MS

# 划分训练集和测试集（80% 训练，20% 测试）
X_train, X_test, y_train, y_test = train_test_split(
    comp_data,
    performance,
    test_size=0.2,
    random_state=42
)
print("数据集划分完成。")
print(f"训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

# ----------------------------
# 归一化处理
# ----------------------------
com_scaler = MinMaxScaler()
performance_scaler = MinMaxScaler()

# 对训练集进行归一化
X_train_normalized = com_scaler.fit_transform(X_train)
y_train_normalized = performance_scaler.fit_transform(y_train.reshape(-1, 1))

# 对测试集进行归一化（使用训练集的 scaler 进行变换）
X_test_normalized = com_scaler.transform(X_test)
y_test_normalized = performance_scaler.transform(y_test.reshape(-1, 1))
print("数据归一化完成。")
print(f"X_train_normalized shape: {X_train_normalized.shape}")
print(f"y_train_normalized shape: {y_train_normalized.shape}")

# ----------------------------
# 转换为 Tensor
# ----------------------------
X_train_tensor = torch.tensor(X_train_normalized, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_normalized, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_normalized, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test_normalized, dtype=torch.float32).to(device)
print("数据转换为Tensor完成。")

# ----------------------------
# 获取潜在空间表示
# ----------------------------
with torch.no_grad():
    # 获取训练集的潜在空间表示
    mu_train, log_var_train = model.encode(X_train_tensor)
    z_train = model.reparameterize(mu_train, log_var_train).cpu().numpy()

    # 获取测试集的潜在空间表示
    mu_test, log_var_test = model.encode(X_test_tensor)
    z_test = model.reparameterize(mu_test, log_var_test).cpu().numpy()
print("潜在空间表示提取完成。")
print(f"z_train shape: {z_train.shape}")
print(f"z_test shape: {z_test.shape}")

# ----------------------------
# 检查是否需要使用 t-SNE
# ----------------------------
if latent_dim > 2:
    tsne = TSNE(n_components=2, random_state=42)
    z_combined = np.concatenate((z_train, z_test), axis=0)
    z_combined_2d = tsne.fit_transform(z_combined)
    z_train_2d, z_test_2d = np.split(z_combined_2d, [len(z_train)])
    print("使用 t-SNE 将潜在空间降维至 2D")
else:
    z_train_2d = z_train
    z_test_2d = z_test
    print("潜在空间已经是 2D，无需使用 t-SNE")

# 确认 z_train_2d 已正确定义
print(f"z_train_2d shape: {z_train_2d.shape}")
print(f"z_test_2d shape: {z_test_2d.shape}")

# ----------------------------
# 使用模型进行性能预测
# ----------------------------
with torch.no_grad():
    # 获取训练集的性能预测
    performance_pred_train = model.performance_predictor(torch.tensor(z_train, dtype=torch.float32).to(device))
    # 获取测试集的性能预测
    performance_pred_test = model.performance_predictor(torch.tensor(z_test, dtype=torch.float32).to(device))

# 将性能预测结果转换为 numpy 数组并展平
performance_pred_train = performance_pred_train.cpu().numpy().flatten()
performance_pred_test = performance_pred_test.cpu().numpy().flatten()

# 对预测结果进行逆归一化
performance_pred_train = performance_scaler.inverse_transform(performance_pred_train.reshape(-1, 1)).flatten()
performance_pred_test = performance_scaler.inverse_transform(performance_pred_test.reshape(-1, 1)).flatten()

# ----------------------------
# 计算预测精度（使用 R²）
# ----------------------------
r2_train = r2_score(y_train, performance_pred_train)
r2_test = r2_score(y_test, performance_pred_test)
print(f"训练集 R² 得分: {r2_train}")
print(f"测试集 R² 得分: {r2_test}")

# ----------------------------
# 保存训练集和测试集的数据
# ----------------------------
# 训练集
train_output_df = pd.DataFrame({
    **{f'Comp_{i + 1}': X_train[:, i] for i in range(input_dim)},
    'z_2d_x': z_train_2d[:, 0],
    'z_2d_y': z_train_2d[:, 1],
    'Performance': y_train,
    'Predicted_Performance': performance_pred_train,
    'R2_Score': [r2_train] * len(y_train)  # 所有行的 R² 得分相同
})
train_output_df.to_csv(output_train_csv_path, index=False)
print(f"训练集数据已保存至 {output_train_csv_path}")

# 测试集
test_output_df = pd.DataFrame({
    **{f'Comp_{i + 1}': X_test[:, i] for i in range(input_dim)},
    'z_2d_x': z_test_2d[:, 0],
    'z_2d_y': z_test_2d[:, 1],
    'Performance': y_test,
    'Predicted_Performance': performance_pred_test,
    'R2_Score': [r2_test] * len(y_test)  # 所有行的 R² 得分相同
})
test_output_df.to_csv(output_test_csv_path, index=False)
print(f"测试集数据已保存至 {output_test_csv_path}")

# ----------------------------
# 1. 定义潜在空间的网格并计算预测性能
# ----------------------------
# 设置网格分辨率
num_bins = 1000  # 可以根据需要调整

# 获取潜在空间的边界，并添加一些填充以覆盖所有数据点
padding = 0.05  # 5%的填充
x_min, x_max = z_train_2d[:, 0].min(), z_train_2d[:, 0].max()
y_min, y_max = z_train_2d[:, 1].min(), z_train_2d[:, 1].max()
x_range = x_max - x_min
y_range = y_max - y_min
x_min -= padding * x_range
x_max += padding * x_range
y_min -= padding * y_range
y_max += padding * y_range

# 创建网格边界
x_edges = np.linspace(x_min, x_max, num_bins + 1)
y_edges = np.linspace(y_min, y_max, num_bins + 1)
X_mesh, Y_mesh = np.meshgrid(x_edges, y_edges)

# 生成网格中心点作为潜在向量
x_centers = (x_edges[:-1] + x_edges[1:]) / 2
y_centers = (y_edges[:-1] + y_edges[1:]) / 2
X_centers, Y_centers = np.meshgrid(x_centers, y_centers)
grid_z = np.vstack([X_centers.ravel(), Y_centers.ravel()]).T  # 形状: (2500, 2)

# 如果 latent_dim > 2，需要将 grid_z 扩展到原始潜在维度
if latent_dim > 2:
    # 假设剩余维度设置为0（可以根据需要调整）
    additional_dims = latent_dim - 2
    grid_z = np.hstack([grid_z, np.zeros((grid_z.shape[0], additional_dims))])

# 转换为 Tensor
grid_z_tensor = torch.tensor(grid_z, dtype=torch.float32).to(device)

# ----------------------------
# 2. 使用模型预测性能
# ----------------------------
with torch.no_grad():
    # 获取性能预测
    performance_pred_grid = performance_scaler.inverse_transform(model.performance_predictor(grid_z_tensor).cpu().numpy()).flatten()

# 确认 performance_pred_grid 的大小
print(f"performance_pred_grid size: {performance_pred_grid.size}")  # 应输出 2500

if performance_pred_grid.size != num_bins * num_bins:
    raise ValueError(f"性能预测数量 ({performance_pred_grid.size}) 与网格数量 ({num_bins * num_bins}) 不匹配。请检查模型输出。")

# ----------------------------
# 3. 创建性能映射
# ----------------------------
performance_grid = performance_pred_grid.reshape(num_bins, num_bins)

# ----------------------------
# 4. 绘制热图和叠加测试集散点图
# ----------------------------
plt.figure(figsize=(12, 10))

# 绘制性能热图
pcm = plt.pcolormesh(
    X_mesh,
    Y_mesh,
    performance_grid,
    cmap='viridis',
    shading='auto',
    edgecolors='none'      # 不显示边缘颜色
)

# 添加颜色条
cbar = plt.colorbar(pcm, label="Predicted Ms (K)")
cbar.ax.tick_params(labelsize=fs * 0.8)  # 颜色条刻度字体大小

# 叠加测试集的散点图
sc_test = plt.scatter(
    z_test_2d[:, 0],
    z_test_2d[:, 1],
    c=y_test,
    cmap='viridis',
    s=400,
    marker='^',
    label="Test",
    linewidth=1.0,
    edgecolors="k"
)

# 设置轴标签
plt.xlabel("Latent Dimension 1", fontsize=fs)
plt.ylabel("Latent Dimension 2", fontsize=fs)

# 设置刻度字体大小
plt.tick_params(axis='both', which='major', labelsize=fs * 0.8)

# 设置图例
plt.legend(fontsize=fs * 0.8, frameon=False, loc="upper left")

# 调整边框线宽
for spine in plt.gca().spines.values():
    spine.set_linewidth(2)

# 设置图表尺寸
plt.gcf().set_size_inches(12, 10)

# 保存并显示图表
plt.grid(False)
plt.tight_layout()
plt.savefig('Latent_Space_test_pcolormesh.png', dpi=500, bbox_inches='tight')
plt.show()
print("潜在空间分布图已保存为 'Latent_Space_test_pcolormesh.png'")
