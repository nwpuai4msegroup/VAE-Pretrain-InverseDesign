import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple

class VAE(nn.Module):
    def __init__(self, input_dim: int = 6,
                 latent_dim: int = 20, hidden_dim: int = 100) -> None:
        """
        变分自编码器(VAE)实现，包含合金成分约束
        参数:
            input_dim: 输入数据的维度（合金成分维度）
            latent_dim: 潜在空间的维度
            hidden_dim: 隐藏层的维度
        """
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        # 编码器部分：将输入数据映射到潜在空间
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 输出潜在空间的均值和对数方差
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

        # 解码器部分：将潜在变量重构为原始数据
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, input_dim)
        )

        # 元素范围约束（提前定义，避免每次计算）
        self.register_buffer('element_ranges', torch.tensor([
            [60, 80],  # Ti
            [0, 6],  # Mo
            [15, 30],  # Nb
            [0, 14],  # Zr
            [0, 14],  # Sn
            [0, 2]  # Ta
        ], dtype=torch.float32))

    def encode(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码过程：将输入数据转换为潜在分布的参数"""
        h = self.encoder(input)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """解码过程：将潜在变量重构为原始数据"""
        return self.decoder(z)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """重参数化技巧：从潜在分布中采样"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播过程"""
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def generator_loss(self, recons: torch.Tensor, input: torch.Tensor,
                       mu: torch.Tensor, log_var: torch.Tensor,
                       epoch: int, total_epochs: int) -> dict:
        """
        计算生成器(VAE)的损失函数（移除GAN对抗损失）
        参数:
            recons: 重构的数据
            input: 原始输入数据
            mu: 潜在空间的均值
            log_var: 潜在空间的对数方差
            epoch: 当前训练周期
            total_epochs: 总训练周期数
        返回:
            包含各项损失的字典
        """
        # 1. 重构损失（均方误差）
        recon_loss = F.mse_loss(recons, input)

        # 2. KL散度（逐步增加权重）
        kld_weight = min(0.01 + (epoch / total_epochs) * 0.1, 0.1)
        kld_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        # 3. 添加经验损失（合金成分约束）
        # 优化后的范围损失计算（更高效）
        lower_bounds = self.element_ranges[:, 0]
        upper_bounds = self.element_ranges[:, 1]

        # 计算低于下界的损失
        below_loss = torch.clamp_min(lower_bounds - recons, 0) ** 2
        # 计算高于上界的损失
        above_loss = torch.clamp_min(recons - upper_bounds, 0) ** 2
        # 组合范围损失
        range_loss = torch.mean(below_loss + above_loss)

        # 总和约束损失（惩罚总和≠100%）
        sum_loss = torch.mean((recons.sum(dim=1) - 100.0) ** 2)

        # 组合所有损失
        total_loss = recon_loss + kld_weight * kld_loss + range_loss + sum_loss

        return {
            'loss': total_loss,
            'Recon_Loss': recon_loss,
            'KLD': kld_loss,
            'Range_Loss': range_loss,
            'Sum_Loss': sum_loss
        }