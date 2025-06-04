import argparse
import copy
import torch
import time
import numpy as np
import os
import csv  # 用于将结果保存到 CSV 文件中
from client import FedMGKDClient

class FedMGKDServer:
    def __init__(self, args):
        self.args = args  # 保存参数对象
        self.global_epochs = args.global_epochs
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.eval_interval = args.eval_interval
        self.args.partition = args.partition

        # 初始化所有客户端
        self.clients = []
        for i in range(self.num_clients):
            client = FedMGKDClient(args, client_id=i)
            self.clients.append(client)

        # 用于存储全局特征提取器（GFE）
        self.gfe = None
        self.uploaded_weights = []
        self.uploaded_models = []
        self.selected_clients = []
        self.best_test_acc = 0
        self.Budget = []  # 记录每一轮的时间消耗

        # 记录每轮评估结果，后续保存为 CSV 文件
        self.results = []

    def send_models(self):
        """分发全局模型到所有客户端"""
        assert len(self.clients) > 0
        for client in self.clients:
            client.set_model(copy.deepcopy(self.gfe))

    def add_parameters(self, w, client_model):
        """按比例累加客户端模型参数（用于加权平均）"""
        for server_param, client_param in zip(self.gfe.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def aggregate_parameters(self):
        """聚合客户端上传的全局特征提取器参数"""
        assert len(self.uploaded_models) > 0

        # 用其中一个模型副本初始化全局模型，并将其所有参数置零
        self.gfe = copy.deepcopy(self.uploaded_models[0])
        for param in self.gfe.parameters():
            param.data.zero_()

        # 按照上传样本数量进行加权平均
        total_samples = sum(self.uploaded_weights)
        for w, model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w / total_samples, model)

    def run(self):
        """运行联邦学习的主流程"""
        print(f"\n▶ Starting Federal Training with {self.num_clients} clients...")

        for epoch in range(self.global_epochs):
            start_time = time.time()
            # 选择本轮参与训练的客户端
            self.selected_clients = self.select_clients()

            # 定时评估模型性能
            if epoch % self.eval_interval == 0:
                print(f"\n🔁 Round {epoch + 1}/{self.global_epochs} {'-' * 30}")
                self.evaluate(epoch)

            # 执行选中客户端本地训练
            for client in self.selected_clients:
                client.train()

            # 接收客户端上传的模型参数
            self.receive_models()
            # 聚合全局特征提取器参数
            self.aggregate_parameters()
            # 将最新的全局模型发送到所有客户端
            self.send_models()

            # 记录每轮消耗的时间
            round_time = time.time() - start_time
            self.Budget.append(round_time)
            print(f'⏱️ Round time cost: {round_time:.2f}s')

        # 全部训练轮次结束后打印最终结果
        print("\n✅ Training completed!")
        print(f"🏆 Best test accuracy: {self.best_test_acc:.2%}")
        print(f"🕰️ Average time per round: {np.mean(self.Budget):.2f}s")

        # 保存评估结果到 CSV 文件中
        self.save_results_to_csv()

    def receive_models(self):
        """接收参与训练的客户端上传的全局特征提取器参数"""
        assert len(self.selected_clients) > 0

        total_samples = sum([c.train_samples for c in self.selected_clients])
        self.uploaded_weights = []
        self.uploaded_models = []

        for client in self.selected_clients:
            # 权重依据客户端训练样本数量计算
            self.uploaded_weights.append(client.train_samples)
            # 上传的是客户端模型中全局特征提取器部分
            self.uploaded_models.append(copy.deepcopy(client.model.gfe))

    def test_metrics(self):
        """收集所有客户端在测试集上的指标数据"""
        num_samples = []
        corrects = []
        losses = []

        for client in self.clients:
            stats = client.test_metrics()
            num_samples.append(stats["test_num_samples"])
            corrects.append(stats["test_corrects"])
            losses.append(stats["test_cls_loss"])
        return num_samples, corrects, losses

    def train_metrics(self):
        """收集所有客户端在训练集上的指标数据"""
        num_samples = []
        corrects = []
        losses = []

        for client in self.clients:
            stats = client.train_metrics()
            num_samples.append(stats["train_num_samples"])
            corrects.append(stats["train_corrects"])
            losses.append(stats["train_cls_loss"])
        return num_samples, corrects, losses

    def evaluate(self, epoch):
        """评估全局模型在所有客户端上的性能，并记录结果"""
        test_samples, test_corrects, test_losses = self.test_metrics()
        train_samples, train_corrects, train_losses = self.train_metrics()

        test_acc = sum(test_corrects) / sum(test_samples)
        test_loss = sum(test_losses) / sum(test_samples)
        train_acc = sum(train_corrects) / sum(train_samples)
        train_loss = sum(train_losses) / sum(train_samples)

        if test_acc > self.best_test_acc:
            self.best_test_acc = test_acc

        # 计算各客户端的准确率标准差
        client_accs = [c / s for c, s in zip(test_corrects, test_samples)]
        acc_std = np.std(client_accs)

        print(f"\n📊 Evaluation results:")
        print(f"Train › Loss: {train_loss:.4f}  Acc: {train_acc:.2%}")
        print(f"Test  › Loss: {test_loss:.4f}  Acc: {test_acc:.2%} ")

        formatted_train_acc = f"{train_acc * 100:.2f}%"
        formatted_test_acc_with_std = f"{test_acc * 100:.2f}% "
        self.results.append({
            "epoch": epoch,
            "train_acc": formatted_train_acc,
            "train_loss": f"{train_loss:.4f}",
            "test_acc": formatted_test_acc_with_std,
            "test_loss": f"{test_loss:.4f}"
        })

    def save_results_to_csv(self):
        """将每轮评估结果保存到 CSV 文件中"""
        filename = (
            f"{self.args.dataset}_"
            f"clients{self.args.num_clients}_"
            f"ratio{self.args.join_ratio}_"
            f"alpha{self.args.alpha}_"
            f"partition{self.args.partition}_"
            f"test{self.args.test}.csv"
        )

        # 指定结果保存目录，可根据需要调整
        save_dir = os.path.join(os.getcwd(), "result")
        os.makedirs(save_dir, exist_ok=True)

        filepath = os.path.join(save_dir, filename)
        with open(filepath, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.results[0].keys())
            writer.writeheader()
            writer.writerows(self.results)

        print(f"💾 Saved results to: {filepath}")

    def select_clients(self):
        """随机选择本轮参与训练的客户端"""
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.randint(
                self.num_join_clients, self.num_clients + 1
            )
        else:
            self.current_num_join_clients = self.num_join_clients

        return np.random.choice(self.clients, self.current_num_join_clients, replace=False).tolist()

