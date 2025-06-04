import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset.utils import read_client_data
from model import create_model


class FedMGKDClient:
    """支持多层次蒸馏、动态可信度评估和特征正交约束的客户端"""
    def __init__(self, args, client_id: int):
        self.client_id = client_id
        self.device = args.device
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size

        # 蒸馏相关超参
        # self.distill_weight   = args.distill_weight
        # 可学习的蒸馏权重 logits
        self.distill_logits   = nn.Parameter(torch.zeros(3, device=self.device))  # [logit_gf, logit_c1, logit_c2]

        # 动态可信度 EMA
        self.trust_ema           = getattr(args, "trust_ema", 0.9)
        self.teacher_confidence  = 1.0

        # 模型 & 教师模型
        self.model = create_model(args.dataset).to(self.device)
        self.global_model = copy.deepcopy(self.model).to(self.device)
        self.global_model.eval()
        for p in self.global_model.parameters():
            p.requires_grad = False

        # 损失 & 优化器
        self.cls_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        # 将 distill_logits 加入优化器
        self.opt = torch.optim.Adam(
            list(self.model.gfe.parameters()) +
            list(self.model.phead.parameters()) +
            [self.distill_logits],
            lr=args.lr, weight_decay=args.weight_decay
        )

        # 数据加载
        train_ds = read_client_data(
            args.base_data_dir, args.dataset, args.experiment_name,
            client_id, is_train=True
        )
        test_ds  = read_client_data(
            args.base_data_dir, args.dataset, args.experiment_name,
            client_id, is_train=False
        )
        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                       shuffle=True, drop_last=True)
        self.test_loader  = DataLoader(test_ds,  batch_size=self.batch_size,
                                       shuffle=False, drop_last=False)
        self.train_samples = len(train_ds)

    def set_model(self, global_gfe: nn.Module):
        """同步服务器下发的全局 GFE"""
        for src, tgt in zip(global_gfe.parameters(), self.model.gfe.parameters()):
            tgt.data.copy_(src.data)
        for src, tgt in zip(global_gfe.parameters(), self.global_model.gfe.parameters()):
            tgt.data.copy_(src.data)

    def compute_teacher_confidence(self, x: torch.Tensor) -> float:
        """基于教师模型输出置信度做 EMA 更新"""
        with torch.no_grad():
            logits, _, _ = self.global_model(x, reparam=False)
            probs = F.softmax(logits, dim=1)
            conf = probs.max(dim=1)[0].mean().item()
        new_conf = self.trust_ema * self.teacher_confidence + (1 - self.trust_ema) * conf
        self.teacher_confidence = max(0.1, min(new_conf, 1.0))
        return self.teacher_confidence

    def train(self):
        """执行多层次蒸馏 + 动态可信度 + 可学习权重蒸馏 + 正交约束的本地训练"""
        self.model.train()
        for _ in range(self.local_epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.opt.zero_grad()

                # —— 学生前向，获取多层次特征 —— #
                logits, gf_s, csf_s, c1_s, c2_s = self.model(
                    x, reparam=True, return_intermediate=True
                )
                loss_cls = self.cls_loss(logits, y)

                # —— 教师前向 —— #
                with torch.no_grad():
                    _, gf_t, _, c1_t, c2_t = self.global_model(
                        x, reparam=False, return_intermediate=True
                    )

                # —— 动态可信度 —— #
                trust = self.compute_teacher_confidence(x)

                # —— 多层次蒸馏损失（可学习权重方法） —— #
                loss_gf = self.mse_loss(gf_s, gf_t)
                loss_c1 = self.mse_loss(c1_s, c1_t)
                loss_c2 = self.mse_loss(c2_s, c2_t)
                # Softmax 将 logits 转为归一化权重
                # w_gf, w_c1, w_c2 = F.softmax(self.distill_logits, dim=0)
                w_gf, w_c1, w_c2 = 1, 0, 0
                loss_dist = trust * (w_gf * loss_gf + w_c1 * loss_c1 + w_c2 * loss_c2)

                # —— 特征正交约束 —— #
                gf_n  = F.normalize(gf_s, dim=1)
                csf_n = F.normalize(csf_s, dim=1)
                ortho = (gf_n * csf_n).sum(dim=1).abs().mean()

                # —— 总损失 —— #
                loss = loss_cls + loss_dist + 1 * ortho
                loss.backward()
                self.opt.step()

        torch.cuda.empty_cache()

    def train_metrics(self):
        """返回本地训练集上的 (样本数, 正确数, 总损失)"""
        self.model.eval()
        total, correct, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits, _, _ = self.model(x, reparam=False)
                loss = self.cls_loss(logits, y)
                loss_sum += loss.item() * y.size(0)
                correct  += (logits.argmax(1) == y).sum().item()
                total    += y.size(0)
        return {
            "train_num_samples": total,
            "train_corrects":    correct,
            "train_cls_loss":    loss_sum
        }

    def test_metrics(self):
        """返回本地测试集上的 (样本数, 正确数, 总损失)"""
        self.model.eval()
        total, correct, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits, _, _ = self.model(x, reparam=False)
                loss = self.cls_loss(logits, y)
                loss_sum += loss.item() * y.size(0)
                correct  += (logits.argmax(1) == y).sum().item()
                total    += y.size(0)
        return {
            "test_num_samples": total,
            "test_corrects":    correct,
            "test_cls_loss":    loss_sum
        }
