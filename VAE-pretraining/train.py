import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd
from vae import VAE  # 导入纯VAE模型
import time
from tqdm import tqdm
import numpy as np

# 超参数设置
input_dim = 6
latent_dim = 20
hidden_dim = 100
learning_rate = 5*1e-6
batch_size = 64
epochs = 500
save_interval = 10

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建保存目录
model_save_dir = "./saved_models"
loss_save_dir = "./loss_records"
os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(loss_save_dir, exist_ok=True)

# 初始化模型
model = VAE(
    input_dim=input_dim,
    latent_dim=latent_dim,
    hidden_dim=hidden_dim
).to(device)

# 优化器（仅需生成器优化器）
optimizer = optim.Adam(
    list(model.encoder.parameters()) + list(model.decoder.parameters()),
    lr=learning_rate
)

# 加载数据
file_path = 'filtered_titanium_alloy_compositions_1percent_step.csv'
data = pd.read_csv(file_path)
comp_data = data.iloc[:, :6].values

# 转换为Tensor
X_tensor = torch.tensor(comp_data, dtype=torch.float32)
dataset = TensorDataset(X_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# 训练函数
def train(model, dataloader, optimizer, epochs, save_interval, model_save_dir):
    model.train()

    # 损失记录
    loss_records = {
        'epoch': [],
        'total_loss': [],
        'Recon_Loss': [],
        'KLD': [],
        'Range_Loss': [],
        'Sum_Loss': [],
        'learning_rate': [],
        'epoch_time': []
    }

    best_loss = float('inf')

    for epoch in range(epochs):
        epoch_start_time = time.time()

        # 初始化 epoch 损失
        total_epoch_loss = 0
        recon_epoch_loss = 0
        kld_epoch_loss = 0
        range_epoch_loss = 0
        sum_epoch_loss = 0

        # 使用tqdm显示进度条
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch in progress_bar:

            inputs = batch[0].to(device)

            optimizer.zero_grad()

            # 前向传播
            recons, mu, log_var = model(inputs)

            # 计算损失 - 确保generator_loss方法存在并返回正确的损失项
            loss_dict = model.generator_loss(
                recons,
                inputs,  # 使用inputs而不是batch_data
                mu,
                log_var,
                epoch,
                epochs
            )
            loss = loss_dict['loss']

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Recon': f"{loss_dict['Recon_Loss'].item():.4f}"
            })

            # 累积损失
            total_epoch_loss += loss.item()
            recon_epoch_loss += loss_dict['Recon_Loss'].item()
            kld_epoch_loss += loss_dict['KLD'].item()
            range_epoch_loss += loss_dict['Range_Loss'].item() if 'Range_Loss' in loss_dict else 0
            sum_epoch_loss += loss_dict['Sum_Loss'].item() if 'Sum_Loss' in loss_dict else 0

        # 计算平均损失
        num_batches = len(dataloader)
        loss_records['epoch'].append(epoch + 1)
        loss_records['total_loss'].append(total_epoch_loss / num_batches)
        loss_records['Recon_Loss'].append(recon_epoch_loss / num_batches)
        loss_records['KLD'].append(kld_epoch_loss / num_batches)
        loss_records['Range_Loss'].append(range_epoch_loss / num_batches)
        loss_records['Sum_Loss'].append(sum_epoch_loss / num_batches)
        loss_records['learning_rate'].append(optimizer.param_groups[0]['lr'])

        # 计算 epoch 时间
        epoch_time = time.time() - epoch_start_time
        loss_records['epoch_time'].append(epoch_time)

        # 打印 epoch 摘要
        print(f"\nEpoch [{epoch + 1}/{epochs}] - Time: {epoch_time:.2f}s")
        print(f"Total Loss: {loss_records['total_loss'][-1]:.4f}")
        print(f"  Reconstruction: {loss_records['Recon_Loss'][-1]:.4f}")
        print(f"  KLD: {loss_records['KLD'][-1]:.4f}")
        print(f"  Range Constraint: {loss_records['Range_Loss'][-1]:.4f}")
        print(f"  Sum Constraint: {loss_records['Sum_Loss'][-1]:.4f}")

        # 定期保存模型
        if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
            model_path = os.path.join(model_save_dir, f"vae_model_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),  # 保存整个模型状态
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_records['total_loss'][-1]
            }, model_path)
            print(f"Model saved to {model_path}")

        # 保存最佳模型
        current_loss = loss_records['total_loss'][-1]
        if current_loss < best_loss:
            best_loss = current_loss
            best_model_path = os.path.join(model_save_dir, "vae_model_best.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'loss': current_loss
            }, best_model_path)
            print(f"New best model saved to {best_model_path}")

    return loss_records

# 保存最终结果
def save_results(model, loss_records, model_save_dir, loss_save_dir, optimizer):
    # 保存最终模型
    final_model_path = os.path.join(model_save_dir, "vae_model_final.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")

    # 保存损失记录
    loss_df = pd.DataFrame(loss_records)
    loss_csv_path = os.path.join(loss_save_dir, "training_losses.csv")
    loss_df.to_csv(loss_csv_path, index=False)
    print(f"Loss records saved to {loss_csv_path}")

    # 保存训练配置
    config = {
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'hidden_dim': hidden_dim,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs
    }
    config_df = pd.DataFrame([config])
    config_df.to_csv(os.path.join(loss_save_dir, "training_config.csv"), index=False)
    print(f"Training configuration saved")


# 运行训练
print("Starting VAE training...")
start_time = time.time()
loss_records = train(model, dataloader, optimizer, epochs, save_interval, model_save_dir)
total_time = time.time() - start_time
print(f"\nTraining completed! Total time: {total_time // 3600:.0f}h {(total_time % 3600) // 60:.0f}m {total_time % 60:.2f}s")

save_results(model, loss_records, model_save_dir, loss_save_dir, optimizer)