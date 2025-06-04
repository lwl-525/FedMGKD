# model.py
import torch
import torch.nn as nn


class GFE(nn.Module):
    """Global Feature Extractor，支持中间层输出用于多层次蒸馏"""
    def __init__(self, in_channels=1, hidden_size=1024, embed_dim=512):
        super().__init__()
        # 第一卷积块
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
        )
        # 第二卷积块
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
        )
        # 最终全连接
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x, return_intermediate=False):
        """
        如果 return_intermediate=True，返回 (embed, conv1_feat, conv2_feat)
        否则只返回最终嵌入 embed
        """
        conv1_out = self.conv1(x)                   # [B,32,H1,W1]
        conv2_out = self.conv2(conv1_out)           # [B,64,H2,W2]
        flat = conv2_out.flatten(1)                 # [B, hidden_size]
        embed = self.fc(flat)                       # [B, embed_dim]

        if return_intermediate:
            return embed, conv1_out, conv2_out
        return embed


class CSFE(nn.Module):
    """Client-Specific Feature Extractor"""
    def __init__(self, in_channels=1, hidden_size=1024, embed_dim=512):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            nn.Conv2d(32, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )

    def forward(self, x):
        z = self.convs(x)           # [B,64,H2,W2]
        z = z.flatten(1)            # [B, hidden_size]
        return self.fc(z)           # [B, 2*embed_dim]


class FedMGKDModel(nn.Module):
    """联邦学习模型，集成全局/客户端提取器与分类头"""
    def __init__(self, in_channels=1, num_classes=10, hidden_size=1024, embed_dim=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.gfe = GFE(in_channels, hidden_size, embed_dim)
        self.csfe = CSFE(in_channels, hidden_size, embed_dim)
        self.phead = nn.Linear(embed_dim * 2, num_classes)

    def forward(self, x, reparam=True, return_intermediate=False):
        # 全局特征 + 中间输出
        if return_intermediate:
            gf, conv1, conv2 = self.gfe(x, return_intermediate=True)
        else:
            gf = self.gfe(x)
            conv1 = conv2 = None

        # 客户端特征重参数化
        z = self.csfe(x)
        mean, logvar = z.chunk(2, dim=1)
        if reparam and self.training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            csf = mean + eps * std
        else:
            csf = mean

        # 分类
        logits = self.phead(torch.cat([gf, csf], dim=1))

        if return_intermediate:
            return logits, gf, csf, conv1, conv2
        return logits, gf, csf


def create_model(dataset: str) -> FedMGKDModel:
    specs = {
        "MNIST": (1, 3136, 10),
        "FashionMNIST": (1, 3136, 10),
        "Cifar10": (3, 4096, 10),
        "Cifar100": (3, 4096, 100),
    }
    if dataset not in specs:
        raise ValueError(f"Unsupported dataset: {dataset}")
    in_ch, hs, nc = specs[dataset]
    return FedMGKDModel(in_ch, nc, hs, embed_dim=512)

